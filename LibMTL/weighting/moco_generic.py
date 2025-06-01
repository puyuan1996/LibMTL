# moco_generic_fix.py  (Python ≥3.8, PyTorch ≥1.12)

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist


@dataclass
class MoCoCfg:
    beta0:     float = 0.9     # β_0
    beta_sigma: float = 0.5    # β 衰减
    gamma0:    float = 10.0    # γ_0
    gamma_sigma: float = 0.5   # γ 衰减
    rho:       float = 0.0     # λ L2 正则
    stat_interval: int = 10000 # 统计步长
    stat_sample: int = 128     # 余弦相似度抽样任务数
    eps:       float = 1e-8


class GenericMoCo:
    """
    通用 MoCo 梯度矫正器 (ICLR'23)
    """

    # ------------------------------------------------------------------ #
    #                              init                                   #
    # ------------------------------------------------------------------ #
    def __init__(self,
                 shared_module: torch.nn.Module,
                 world_task_num: int,
                 device: str | torch.device = 'cpu',
                 multi_gpu: bool = False,
                 cfg: MoCoCfg | None = None,
                 pg: dist.ProcessGroup | None = None):

        self.pg = pg or dist.group.WORLD                    # ★ use it
        self.module = shared_module
        self.world_task_num = world_task_num
        self.device = torch.device(device)
        self.multi_gpu = multi_gpu and dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.multi_gpu else 0
        self.cfg = cfg or MoCoCfg()

        # will be set in init_param()
        self.grad_dim: int | None = None
        self.y:      torch.Tensor | None = None   # (T,D)
        self.lambd:  torch.Tensor | None = None   # (T,)
        self.step:   int = 0
        self._param_num: int = 0                  # detect parameter change

    # ------------------------------------------------------------------ #
    #                         helper function                             #
    # ------------------------------------------------------------------ #
    def _share_params(self):
        return self.module.parameters()

    def _grad2vec(self) -> torch.Tensor:
        chunks = []
        for p in self._share_params():
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            chunks.append(g.reshape(-1))
        return torch.cat(chunks)

    def _reset_grad(self, flat_grad: torch.Tensor) -> None:
        offset = 0
        for p in self._share_params():
            n = p.numel()
            p.grad = flat_grad[offset: offset + n].view_as(p)
            offset += n

    def _compute_grad_dim(self):
        return sum(p.numel() for p in self._share_params())

    # ------------------------------------------------------------------ #
    #                         public  API                                 #
    # ------------------------------------------------------------------ #
    def init_param(self) -> None:
        self.grad_dim = self._compute_grad_dim()
        self.y = torch.zeros(self.world_task_num, self.grad_dim,
                             device=self.device, dtype=torch.float32)
        self.lambd = torch.full((self.world_task_num,),
                                1.0 / self.world_task_num,
                                device=self.device, dtype=torch.float32)
        self.step = 0
        self._param_num = self.grad_dim
        if self.rank == 0:
            print(f'[MoCo] init: task={self.world_task_num}, '
                  f'grad_dim={self.grad_dim}, device={self.device}')

    # ------------------------------------------------------------------ #
    #                              core                                   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def backward(self,
                 losses: List[torch.Tensor],
                 retain_graph: bool = False
                 ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:

        print(f'[MoCo] rank{self.rank} line 101')

        # =========== ① 新增：确保所有 rank 的 grad_dim 一致 ===========
        if self.multi_gpu:
            cur_dim = torch.tensor([self._compute_grad_dim()],
                                device=self.device, dtype=torch.long)
            max_dim = cur_dim.clone()
            dist.all_reduce(max_dim, op=dist.ReduceOp.MAX, group=self.pg)
            if int(max_dim.item()) != self.grad_dim:
                if self.rank == 0:
                    print(f'[MoCo] detect parameter number change: '
                        f'{self.grad_dim} → {int(max_dim.item())}  •  re-init')
                dist.barrier(self.pg)        # 保证所有人一起进入 re-init
                self.init_param()
                dist.barrier(self.pg)
        # ===============================================================

        # # 1. 若模型参数改变，则重新初始化
        # if self.grad_dim != self._compute_grad_dim():
        #     # if self.rank == 0:
        #     #     print('[MoCo] parameter number changed, re-init')
        #     print(f'[MoCo] rank{self.rank} parameter number changed, re-init')
        #     self.init_param()

        # 2. 计算本 rank 各任务梯度
        self.module.zero_grad(set_to_none=False)
        local_n = len(losses)
        grads_local = torch.zeros(local_n, self.grad_dim,
                                  device=self.device, dtype=torch.float32)

        for i, loss in enumerate(losses):
            loss.backward(retain_graph=retain_graph or i < local_n - 1)
            grads_local[i] = self._grad2vec().to(torch.float32)
            self.module.zero_grad(set_to_none=False)

        # 3. 多 GPU 汇总可变任务数梯度
        if self.multi_gpu:
            world_size = dist.get_world_size()
            # gather local_n
            n_tensor = torch.tensor([local_n], device=self.device)
            nums = [torch.zeros_like(n_tensor) for _ in range(world_size)]
            print(f'[MoCo] rank{self.rank} dist.all_gather(nums, n_tensor)')
            # dist.all_gather(nums, n_tensor)
            print(f'[MoCo] rank{self.rank} nums:{nums}, n_tensor:{n_tensor})')

            dist.all_gather(nums, n_tensor, group=self.pg)
            nums = [int(x.item()) for x in nums]
            max_n = max(nums)

            # pad
            if local_n < max_n:
                pad = torch.zeros(max_n - local_n, self.grad_dim,
                                  device=self.device, dtype=torch.float32)
                grads_local = torch.cat([grads_local, pad], 0)

            # gather
            grads_buf = [torch.zeros_like(grads_local) for _ in range(world_size)]
            # dist.all_gather(grads_buf, grads_local)
            print(f'[MoCo] rank{self.rank} grads_buf:{grads_buf}, grads_local:{grads_local})')

            dist.all_gather(grads_buf, grads_local, group=self.pg)
            print(f'[MoCo] rank{self.rank} dist.all_gather(grads_buf, grads_local)')

            if self.rank == 0:
                grads = torch.cat([g[:n] for g, n in zip(grads_buf, nums)], 0)
            else:
                grads = torch.empty(self.world_task_num, self.grad_dim,
                                    device=self.device, dtype=torch.float32)

            # gather global losses (仅 rank0 真正需要)
            losses_send = [float(l.detach()) for l in losses]
            losses_recv: List[List[float]] = [None] * world_size  # type: ignore
            # dist.all_gather_object(losses_recv, losses_send)
            # gather object（backend 仍然 NCCL，但必须显式带 group）
            dist.all_gather_object(losses_recv, losses_send, group=self.pg)
            if self.rank == 0:
                losses_world = [torch.tensor(x, device=self.device, dtype=torch.float32)
                                for x in sum(losses_recv, [])]
            else:
                losses_world = None
        else:
            grads = grads_local
            losses_world = [l.to(self.device, dtype=torch.float32) for l in losses]


        print(f'[MoCo] rank{self.rank} line 159 rank0 做 MoCo')

        # ------------------------------------------------ rank0 做 MoCo 
        if self.rank == 0:
            t = self.step
            beta  = self.cfg.beta0  * (self.cfg.beta_sigma  ** t)
            gamma = self.cfg.gamma0 * (self.cfg.gamma_sigma ** t)

            # 归一化 + 按损失缩放
            grads = grads / (grads.norm(dim=1, keepdim=True) + self.cfg.eps)
            # === 关键修复点 ====================================================
            scale = torch.tensor(losses_world,  # type: ignore
                                 device=self.device,
                                 dtype=torch.float32).unsqueeze(1)      # (T,1)
            grads.mul_(scale)                                            # (T,D)
            # ===================================================================

            # EMA 更新 y
            self.y.mul_(beta).add_(grads, alpha=1 - beta)

            # λ 更新
            yy = self.y @ self.y.t()
            grad_lambd = yy @ self.lambd + self.cfg.rho * self.lambd
            self.lambd = F.softmax(self.lambd - gamma * grad_lambd, dim=-1)

            new_grad = (self.y.t() @ self.lambd).contiguous()
        else:
            new_grad = torch.empty(self.grad_dim, device=self.device, dtype=torch.float32)

        print(f'[MoCo] rank{self.rank} line 187 dist.broadcast(new_grad, src=0)')

        # 4. 广播 ḡ 与 λ
        if self.multi_gpu:
            # dist.broadcast(new_grad, src=0)
            # dist.broadcast(self.lambd, src=0)

            dist.broadcast(new_grad, src=0, group=self.pg)
            dist.broadcast(self.lambd, src=0, group=self.pg)
        
        # barrier（如果你想保守同步）
        dist.barrier(self.pg)

        # 5. 把新梯度写回
        self._reset_grad(new_grad)

        self.step += 1

        # 6. 可选统计（仅 rank0）
        stats: Optional[Dict[str, Any]] = None
        if self.rank == 0 and self.cfg.stat_interval > 0 \
           and self.step % self.cfg.stat_interval == 0:

            # 仅随机采样 K 个任务，避免 O(T²) 内存
            T = self.world_task_num
            K = min(self.cfg.stat_sample, T)
            idx = torch.randperm(T, device=self.device)[:K]
            g_sample = grads[idx]
            nrm = g_sample.norm(dim=1)
            cos = (g_sample @ g_sample.t()) / (nrm[:, None] * nrm[None, :] + self.cfg.eps)
            avg_cos = cos.masked_fill(torch.eye(K, device=self.device).bool(), 0).mean()

            stats = dict(
                step=self.step,
                beta=beta, gamma=gamma,
                lambd=self.lambd.cpu().numpy(),
                grad_norms=grads.norm(dim=1).cpu().numpy(),
                avg_cos_sim=avg_cos.item(),
            )
            print(f'[MoCo] step={self.step:<8} avg_cos={avg_cos.item():.4f}')

        return self.lambd.cpu().numpy(), stats

