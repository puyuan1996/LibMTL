# fast_moco.py   (Py ≥3.8, Torch ≥1.12, NCCL backend)

from __future__ import annotations
import torch, torch.distributed as dist, torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class MoCoCfg:
    beta0: float = .9
    beta_sigma: float = .5
    gamma0: float = 10.
    gamma_sigma: float = .5
    rho: float = 0.
    stat_interval: int = 0          # 0 = 不统计
    eps: float = 1e-8
    dtype: torch.dtype = torch.float32


class FastMoCo:
    """
    本方法参考论文 `Mitigating Gradient Bias in Multi-objective Learning: A Provably Convergent Approach (ICLR 2023) <https://openreview.net/forum?id=dLAYGdKTi2>`_  
    基于作者分享的代码 (Heshan Fernando: fernah@rpi.edu) 实现。

    纯 GPU、支持可变任务数的 MoCo 梯度矫正器
    """
    # ------------------------------------------------------------
    def __init__(self,
                 shared_module: torch.nn.Module,
                 world_task_num: int,
                 device: str | torch.device = "cuda",
                 multi_gpu: bool = False,
                 cfg: MoCoCfg | None = None,
                 pg: dist.ProcessGroup | None = None) -> None:

        self.module = shared_module
        self.T = world_task_num
        self.dev = torch.device(device)
        self.pg = pg or dist.group.WORLD
        self.multi = multi_gpu and dist.is_initialized()
        self.rank = dist.get_rank(self.pg) if self.multi else 0
        self.cfg = cfg or MoCoCfg()

        self.grad_dim = self._compute_grad_dim()
        self._init_state()

    # ------------------------------------------------------------
    # helpers
    def _compute_grad_dim(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def _init_state(self):
        D = self.grad_dim
        T = self.T
        self.y = torch.zeros(T, D, device=self.dev, dtype=self.cfg.dtype)
        self.lambd = torch.full((T,), 1 / T, device=self.dev,
                                dtype=self.cfg.dtype)
        self.step = 0

    def _grad2vec(self) -> torch.Tensor:
        chunks = []
        for p in self.module.parameters():
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            chunks.append(g.reshape(-1))
        return torch.cat(chunks).to(self.cfg.dtype)

    def _write_back(self, flat: torch.Tensor):
        off = 0
        for p in self.module.parameters():
            n = p.numel()
            p.grad = flat[off: off + n].view_as(p)
            off += n
    # ------------------------------------------------------------
    @torch.no_grad()
    def backward(self,
                 losses: List[torch.Tensor],
                 retain_graph: bool = False
                 ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:

        # -------- 0. 本地梯度
        self.module.zero_grad(set_to_none=False)
        L = len(losses)                              # 本 rank 任务数
        g_local = torch.zeros(L, self.grad_dim, device=self.dev,
                              dtype=self.cfg.dtype)

        for i, loss in enumerate(losses):
            loss.backward(retain_graph=retain_graph or i < L - 1)
            g_local[i] = self._grad2vec()
            self.module.zero_grad(set_to_none=False)

        # -------- 1. all_gather (padding 到 max_L)
        if self.multi:
            world = dist.get_world_size(self.pg)
            n_tensor = torch.tensor([L], device=self.dev, dtype=torch.int64)
            n_all = [torch.zeros_like(n_tensor) for _ in range(world)]
            dist.all_gather(n_all, n_tensor, group=self.pg)
            n_all = [int(x.item()) for x in n_all]
            max_L = max(n_all)

            if L < max_L:
                pad = torch.zeros(max_L - L, self.grad_dim,
                                  device=self.dev, dtype=self.cfg.dtype)
                g_local = torch.cat([g_local, pad], 0)

            g_buf = [torch.empty_like(g_local) for _ in range(world)]
            dist.all_gather(g_buf, g_local, group=self.pg)

            if self.rank == 0:
                grads = torch.cat([g_buf[r][:n_all[r]] for r in range(world)], 0)
            else:
                grads = torch.empty(self.T, self.grad_dim,
                                    device=self.dev, dtype=self.cfg.dtype)

            # -------- gather losses
            loss_t = torch.stack([l.detach().to(self.dev, self.cfg.dtype)
                                  for l in losses])   # (L,)
            if L < max_L:
                loss_t = torch.cat([loss_t,
                                    torch.zeros(max_L - L, device=self.dev,
                                                dtype=self.cfg.dtype)])
            loss_buf = [torch.empty_like(loss_t) for _ in range(world)]
            dist.all_gather(loss_buf, loss_t, group=self.pg)

            if self.rank == 0:
                loss_world = torch.cat([loss_buf[r][:n_all[r]]
                                        for r in range(world)], 0)
            else:
                loss_world = None
        else:               # 单 GPU
            grads = g_local
            loss_world = torch.stack([l.to(self.dev, self.cfg.dtype)
                                      for l in losses])

        # -------- 2. rank0 做 MoCo 更新
        new_g = torch.empty(self.grad_dim, device=self.dev,
                            dtype=self.cfg.dtype)

        if self.rank == 0:
            # 归一化 + 损失缩放
            grads = grads / (grads.norm(dim=1, keepdim=True) + self.cfg.eps)
            scale = loss_world.unsqueeze(1)          # (T,1)
            grads.mul_(scale)

            # 固定学习率 v2
            # 平滑更新 y（采用固定平滑系数，也可根据步数调整）
            self.y = self.y - 0.99 * (self.y - grads)
            # # 更新 λ（基于正则化 MGDA 子问题）
            self.lambd = F.softmax(self.lambd - 10 * (self.y @ self.y.t()) @ self.lambd, dim=-1)

            # 自适应学习率 v3
            # t = self.step
            # beta = self.cfg.beta0 * (self.cfg.beta_sigma ** t)
            # gamma = self.cfg.gamma0 * (self.cfg.gamma_sigma ** t)
            # self.y.mul_(beta).add_(grads, alpha=1 - beta)
            # yy = self.y @ self.y.t()
            # grad_l = yy @ self.lambd + self.cfg.rho * self.lambd
            # self.lambd = F.softmax(self.lambd - gamma * grad_l, dim=-1)

            new_g.copy_(self.y.t() @ self.lambd)
        # ------------------------------------------------ broadcast
        if self.multi:
            dist.broadcast(new_g,     src=0, group=self.pg)
            dist.broadcast(self.lambd, src=0, group=self.pg)

        # 写回共享参数梯度
        self._write_back(new_g)
        self.step += 1

        # -------- 3. 统计（可选）
        stats: Optional[Dict[str, Any]] = None
        if (self.rank == 0 and self.cfg.stat_interval > 0
                and self.step % self.cfg.stat_interval == 0):
            norms = grads.norm(dim=1)
            cos = (grads @ grads.t()) / (norms[:, None] * norms[None, :] + self.cfg.eps)
            avg_cos = cos.masked_fill(torch.eye(self.T, device=self.dev).bool(), 0).mean()
            stats = dict(step=self.step,
                         lambd=self.lambd.cpu().numpy(),
                         grad_norms=norms.cpu().numpy(),
                         avg_cos_sim=avg_cos.item())

            print(f'[MoCo] step={self.step:<8} stats={stats}')
            print(f'[MoCo] step={self.step:<8} avg_cos={avg_cos.item():.4f}')

        return self.lambd.cpu().numpy(), stats