# fast_moco_mem_eff.py  (Py ≥3.8, Torch ≥1.12, NCCL backend)

from __future__ import annotations
import torch, torch.distributed as dist
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


# ------------------------ 配置 ------------------------
@dataclass
class MoCoCfg:
    beta0: float = .9
    beta_sigma: float = .5
    gamma0: float = 10.
    gamma_sigma: float = .5
    rho: float = 0.
    stat_interval: int = 0
    eps: float = 1e-8
    dtype: torch.dtype = torch.float32


@dataclass
class MemCfg:
    low_dtype: torch.dtype = torch.float16     # y / grads 存储精度
    chunk_mb: int = 32                         # 单块显存上限 (MiB)
    offload: bool = True                       # y 是否常驻 CPU(pinned)


# ======================================================
class FastMoCoMemEff:
    """
    内存友好的 MoCo 梯度矫正 (多任务 + DDP)
    论文: Mitigating Gradient Bias in Multi-objective Learning (ICLR 2023)
    """

    # ------------------------------------------------------------
    def __init__(self,
                 shared_module: torch.nn.Module,
                 world_task_num: int,
                 device: str | torch.device = "cuda",
                 multi_gpu: bool = False,
                 cfg: MoCoCfg | None = None,
                 mem_cfg: MemCfg | None = None,
                 pg: dist.ProcessGroup | None = None) -> None:

        self.module = shared_module
        self.T = world_task_num
        self.dev = torch.device(device)
        self.pg = pg or dist.group.WORLD
        self.multi = multi_gpu and dist.is_initialized()
        self.rank = dist.get_rank(self.pg) if self.multi else 0
        self.cfg = cfg or MoCoCfg()
        self.mem_cfg = mem_cfg or MemCfg()

        self.grad_dim = sum(p.numel() for p in self.module.parameters())
        self._init_state()

    # ------------------------------------------------------------
    def _init_state(self):
        D, T, ld = self.grad_dim, self.T, self.mem_cfg.low_dtype
        pin_opt = dict(device='cpu', pin_memory=True) if self.mem_cfg.offload else {}
        self.y = torch.zeros((T, D), dtype=ld, **pin_opt)          # 状态矩阵
        self.lambd = torch.full((T,), 1 / T, device=self.dev, dtype=self.cfg.dtype)
        self.step = 0

        bytes_per_el = torch.finfo(ld).bits // 8
        cols_per_chunk = (self.mem_cfg.chunk_mb * 1024 ** 2) // (T * bytes_per_el)
        self.chunk_size = max(1, cols_per_chunk)

    # ---------------- helper ----------------
    def _grad2vec(self) -> torch.Tensor:
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p))
                          .reshape(-1) for p in self.module.parameters()]).to(self.cfg.dtype)

    def _write_back(self, flat: torch.Tensor):
        off = 0
        for p in self.module.parameters():
            n = p.numel()
            p.grad = flat[off: off + n].view_as(p)
            off += n

    # ================== 主函数 ==================
    @torch.no_grad()
    def backward(self,
                 losses: List[torch.Tensor],
                 retain_graph: bool = False
                 ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:

        # ---------- 0. 本地梯度 ----------
        self.module.zero_grad(set_to_none=False)
        L = len(losses)
        g_local = torch.zeros(L, self.grad_dim, device=self.dev, dtype=self.cfg.dtype)

        for i, loss in enumerate(losses):
            loss.backward(retain_graph=retain_graph or i < L - 1)
            g_local[i] = self._grad2vec()
            self.module.zero_grad(set_to_none=False)

        # ---------- 1. all-gather ----------
        if self.multi:
            W = dist.get_world_size(self.pg)
            n_local = torch.tensor([L], dtype=torch.int32, device=self.dev)
            n_all = [torch.zeros_like(n_local) for _ in range(W)]
            dist.all_gather(n_all, n_local, group=self.pg)
            n_all = [int(x.item()) for x in n_all]
            max_L = max(n_all)

            if L < max_L:
                pad = torch.zeros(max_L - L, self.grad_dim, device=self.dev, dtype=self.cfg.dtype)
                g_local = torch.cat([g_local, pad], 0)

            g_half = g_local.to(self.mem_cfg.low_dtype)
            g_buf = [torch.empty_like(g_half) for _ in range(W)]
            dist.all_gather(g_buf, g_half, group=self.pg)

            loss_local = torch.zeros(max_L, dtype=self.mem_cfg.low_dtype, device=self.dev)
            loss_local[:L] = torch.stack([l.detach() for l in losses]).to(self.mem_cfg.low_dtype)
            loss_buf = [torch.empty_like(loss_local) for _ in range(W)]
            dist.all_gather(loss_buf, loss_local, group=self.pg)

            if self.rank == 0:
                grads = torch.cat([g_buf[r][:n_all[r]] for r in range(W)], 0)
                loss_world = torch.cat([loss_buf[r][:n_all[r]] for r in range(W)], 0)
                if self.mem_cfg.offload:
                    grads = grads.cpu().pin_memory()
                    loss_world = loss_world.cpu().pin_memory()
            else:
                grads, loss_world = None, None
        else:
            grads = g_local.to(self.mem_cfg.low_dtype)
            loss_world = torch.stack([l.detach() for l in losses]).to(self.mem_cfg.low_dtype)

        # ---------- 2. rank-0 更新 ----------
        new_g = torch.empty(self.grad_dim, device=self.dev, dtype=self.cfg.dtype)

        if self.rank == 0:
            grads = grads.to(self.dev, non_blocking=True)
            loss_world = loss_world.to(self.dev, non_blocking=True)

            norms = grads.norm(dim=1, keepdim=True).clamp_min(self.cfg.eps)
            grads.div_(norms)
            grads.mul_(loss_world.unsqueeze(1))
            torch.nan_to_num(grads, nan=0., posinf=1e4, neginf=-1e4, out=grads)

            T, D, C = *grads.shape, self.chunk_size
            y_mat = self.y if self.mem_cfg.offload else self.y.to(self.dev)

            g_accum = torch.zeros(D, dtype=self.cfg.dtype, device=self.dev)

            for c0 in range(0, D, C):
                c1 = min(D, c0 + C)

                g_chunk = grads[:, c0:c1].to(torch.float32)
                y_chunk = (y_mat[:, c0:c1].cuda(non_blocking=True)
                           if self.mem_cfg.offload else y_mat[:, c0:c1]).to(torch.float32)

                y_chunk.mul_(0.01).add_(g_chunk, alpha=0.99)
                y_mat[:, c0:c1].copy_(y_chunk.to(self.mem_cfg.low_dtype), non_blocking=True)
                g_accum[c0:c1] = (y_chunk.T @ self.lambd)

            # ---- λ 更新（确保同设备） ----
            yy = (self.y.cuda(non_blocking=True)
                  if self.mem_cfg.offload else self.y).to(torch.float32)
            logits = -(yy @ yy.T) @ self.lambd
            logits = logits.clamp(-1e3, 1e3)
            self.lambd = torch.softmax(self.lambd + logits, dim=-1)
            torch.nan_to_num(self.lambd, nan=1 / self.T, posinf=1., neginf=0., out=self.lambd)

            new_g.copy_(g_accum)

        # ---------- 3. broadcast & 写回 ----------
        if self.multi:
            dist.broadcast(new_g,     src=0, group=self.pg)
            dist.broadcast(self.lambd, src=0, group=self.pg)

        torch.nan_to_num(new_g, nan=0., posinf=1e4, neginf=-1e4, out=new_g)
        self._write_back(new_g)
        self.step += 1

        # ---------- 4. 统计(可选) ----------
        stats: Optional[Dict[str, Any]] = None
        if (self.rank == 0 and self.cfg.stat_interval > 0
                and self.step % self.cfg.stat_interval == 0):
            norms = grads.float().norm(dim=1)
            cos = grads.float() @ grads.float().T
            cos.div_(norms[:, None] * norms[None, :] + self.cfg.eps)
            avg_cos = cos.masked_fill(torch.eye(self.T, device=cos.device).bool(), 0).mean()
            stats = dict(step=self.step,
                         lambd=self.lambd.cpu().numpy(),
                         grad_norms=norms.cpu().numpy(),
                         avg_cos_sim=avg_cos.item())
            print(f'[MoCo] step={self.step:<8} stats={stats}')

        return self.lambd.cpu().numpy(), stats