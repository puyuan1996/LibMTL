import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from LibMTL.weighting.abstract_weighting import AbsWeighting

class MoCo(AbsWeighting):
    r"""MoCo.

    本方法参考论文 `Mitigating Gradient Bias in Multi-objective Learning: A Provably Convergent Approach (ICLR 2023) <https://openreview.net/forum?id=dLAYGdKTi2>`_  
    基于作者分享的代码 (Heshan Fernando: fernah@rpi.edu) 实现。

    Args:
        MoCo_beta (float, default=0.5): y 的学习率.
        MoCo_beta_sigma (float, default=0.5): MoCo_beta 的衰减率.
        MoCo_gamma (float, default=0.1): λ 的学习率.
        MoCo_gamma_sigma (float, default=0.5): MoCo_gamma 的衰减率.
        MoCo_rho (float, default=0): λ 更新的 \ell_2 正则化参数.
        MoCo_stat_interval (int, default=100): 统计量计算的间隔步数.

    .. warning::
            MoCo 不支持 representation gradients，即 ``rep_grad`` 必须为 ``False``.
    """
    def __init__(self, share_model, task_num, device, multi_gpu=False):
        super(MoCo, self).__init__()
        self.share_model = share_model
        self.task_num = task_num  # 全局任务数（所有 rank 上的任务求和需等于该值）
        self.device = device
        self.multi_gpu = multi_gpu  # 多 GPU 标志
        self.step = 0
        self.grad_dim = None
        self.y = None
        self.lambd = None
        # 如果有 rep_grad 相关检查，默认 False
        self.rep_grad = False

    def zero_grad_share_params(self):
        """将共享参数的梯度置为零"""
        self.share_model.zero_grad(set_to_none=False)

    def get_share_params(self):
        return self.share_model.parameters()

    def _compute_grad_dim(self):
        """
        计算共享参数梯度拼接后的维度，
        假设 get_share_params 返回的是一个可迭代对象，
        每个参数 view(-1) 后拼接在一起。
        """
        params = list(self.get_share_params())
        self.grad_dim = sum(p.numel() for p in params if p.requires_grad)

    def init_param(self):
        self._compute_grad_dim()
        self.step = 0
        # y: [task_num, grad_dim]
        self.y = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        # λ: [task_num, ]
        self.lambd = (torch.ones([self.task_num, ]) / self.task_num).to(self.device)

    def _compute_statistics(self, grads):
        """
        计算各任务梯度之间的统计量，包括：
          - 每个任务梯度的范数 grad_norms
          - 任务间两两 cosine similarity 构成的矩阵 cos_sim_matrix
          - 非对角项的平均 cosine similarity avg_cos_sim
        输入:
          grads: [task_num, grad_dim] 的梯度张量
        返回:
          stats: 包含上述统计量的字典
        """
        stats = {}
        # 计算每个任务梯度的范数
        grad_norms = torch.norm(grads, dim=1)
        stats['grad_norms'] = grad_norms.detach().cpu().numpy()

        # 计算任务之间的 cosine similarity 矩阵
        # 先计算内积矩阵
        dot_matrix = grads @ grads.t()
        # 计算范数的外积矩        norm_matrix = grad_norms.unsqueeze(1) * grad_norms.unsqueeze(0)
        norm_matrix = grad_norms.unsqueeze(1) * grad_norms.unsqueeze(0)
        # 防止除 0
        cos_sim_matrix = dot_matrix / (norm_matrix + 1e-8)
        # 限制数值范围在 [-1, 1]
        cos_sim_matrix = torch.clamp(cos_sim_matrix, -1, 1)
        stats['cos_sim_matrix'] = cos_sim_matrix.detach().cpu().numpy()
        # 计算除对角线外的平均 cosine similarity
        task_num = grads.shape[0]
        mask = ~torch.eye(task_num, dtype=torch.bool, device=self.device)
        avg_cos_sim = cos_sim_matrix[mask].mean().item()
        stats['avg_cos_sim'] = avg_cos_sim

        return stats

    def backward(self, losses, **kwargs):
        """
        输入:
          losses: 当前 rank 上各个任务的 loss list，
                  每个 loss 均为标量且 requires_grad=True；
                  当前 rank 上任务数量 = len(losses)（不同 rank 可能不同）
        逻辑流程：
          1. 每个 rank 对本地所有任务分别采用 torch.autograd.grad 得到梯度（local_grads: [local_task_num, grad_dim]）；
          2. 利用 dist.all_gather_object 汇总各 rank 上的 local_grads，拼接成总任务数的 grads（[self.task_num, grad_dim]）；
          3. 同样汇总各 rank 上的 loss，得到 aggregated_losses（列表中每个元素为 scalar tensor）；
          4. 仅在 rank0 上进行 MoCo 梯正：归一化每个任务梯度、利用对应 loss 缩放、平滑更新 y 与 λ，然后计算新的共享梯度 new_grads；
             同时在指定步数间隔内计算各任务梯度统计量（如 cosine similarity 等）并打印；
          5. 将 new_grads 广播给所有 rank，调用 _reset_grad 更新共享模型梯度；
          6. 同步 λ 并返回（numpy 格式）以及统计量（若本次未计算则返回 None）。
        """
        # 判断多 GPU 环境
        multi_gpu = self.multi_gpu and dist.is_initialized()
        rank = dist.get_rank() if multi_gpu else 0
        self.device = f'cuda:{rank}'

        self.step += 1  # 更新迭代步数

        # 参数设置，可通过 kwargs 覆盖默认值
        beta = kwargs.get('MoCo_beta', 0.5)
        beta_sigma = kwargs.get('MoCo_beta_sigma', 0.5)
        gamma = kwargs.get('MoCo_gamma', 0.1)
        gamma_sigma = kwargs.get('MoCo_gamma_sigma', 0.5)
        rho = kwargs.get('MoCo_rho', 0)
        stat_interval = kwargs.get('MoCo_stat_interval', 10000)
        # stat_interval = kwargs.get('MoCo_stat_interval', 1)

        if self.rep_grad:
            raise ValueError('不支持方法 MoCo 使用 representation gradients (rep_grad=True)')

        self._compute_grad_dim()

        # 1. 每个 rank 先计算本地任务的梯度
        local_task_num = len(losses)
        local_grads = torch.zeros(local_task_num, self.grad_dim, device=self.device)
        for i in range(local_task_num):
            # 保证 retain_graph=True 以便多次 backward 计算不同任务的梯度
            grad_list = torch.autograd.grad(
                losses[i],
                list(self.get_share_params()),
                retain_graph=True,
                allow_unused=True
            )
            grad_list = [
                g if g is not None else torch.zeros_like(p)
                for p, g in zip(list(self.get_share_params()), grad_list)
            ]
            local_grads[i] = torch.cat([g.view(-1) for g in grad_list])
            self.zero_grad_share_params()

        # 2. 汇总各 rank 上的梯度
        if multi_gpu:
            world_size = dist.get_world_size()
            # 利用 all_gather_object 收集每个 rank 的 local_grads
            all_local_grads = [None for _ in range(world_size)]
            dist.all_gather_object(all_local_grads, local_grads)
            if rank == 0:
                # 拼接得到 all_task_grads, 总行数须等于 self.task_num
                all_task_grads = torch.cat([g.to(self.device) for g in all_local_grads], dim=0)
                if all_task_grads.shape[0] != self.task_num:
                    raise ValueError(f"Aggregated tasks mismatch: got {all_task_grads.shape[0]} tasks, expected {self.task_num}.")
                grads = all_task_grads
            else:
                # 非 rank0 设置占位张量
                grads = torch.empty(self.task_num, self.grad_dim, device=self.device)
        else:
            # 单 GPU 下，本地任务数应即为全局任务数
            if local_task_num != self.task_num:
                raise ValueError("在单 GPU 模式下，loss 数量应等于 self.task_num.")
            grads = local_grads

        # 3. 同步各 rank 上的 loss 值（每个 loss 是标量）
        if multi_gpu:
            local_losses_tensor = torch.stack(losses, dim=0).to(self.device)
            all_losses_list = [torch.empty_like(local_losses_tensor) for _ in range(world_size)]
            dist.all_gather(all_losses_list, local_losses_tensor)
            if rank == 0:
                all_losses_tensor = torch.cat(all_losses_list, dim=0)
                aggregated_losses = list(all_losses_tensor.unbind(dim=0))
            else:
                aggregated_losses = None
        else:
            aggregated_losses = losses

        stats = None  # 若本次未计算统计量则返回 None
        # 4. 仅在 rank0 上进行 MoCo 梯正逻辑及统计量计算
        if rank == 0:
            with torch.no_grad():
                # 针对每个任务：归一化梯度并乘以对应 loss 值（loss_value 为标量）
                for tn in range(self.task_num):
                    loss_val = aggregated_losses[tn].to(self.device)
                    norm = grads[tn].norm()
                    if norm.item() == 0:
                        print(f"Warning: 任务 {tn} 的梯度范数为 0")
                    else:
                        if self.step % 1000 == 0:
                            print(f"任务 {tn} 的梯度范数为 {norm.item()}")
                    grads[tn] = grads[tn] / (norm + 1e-8) * loss_val

                # 若满足指定步数间隔，则计算并打印统计量
                if self.step % stat_interval == 0:
                    stats = self._compute_statistics(grads)
                    print(f"Step {self.step} 梯度统计量:")
                    print(f"  每任务梯度范数: {stats['grad_norms']}")
                    print(f"  平均任务间 cosine similarity: {stats['avg_cos_sim']}")
                    # 如有需要，可打印完整 cos_sim 矩阵
                    # print(f"  Task cosine similarity 矩阵: \n{stats['cos_sim_matrix']}")

                # 平滑更新 y，beta 衰减可能与迭代步数有关
                self.y = self.y - (beta / (self.step ** beta_sigma)) * (self.y - grads)
                # 更新任务权重 λ（基于正则化 MGDA 子问题）
                self.lambd = F.softmax(
                    self.lambd - (gamma / (self.step ** gamma_sigma)) *
                    (self.y @ self.y.t() + rho * torch.eye(self.task_num, device=self.device)) @ self.lambd,
                    dim=-1
                )
                # 根据更新后的 λ 与 y，计算新的共享梯度
                new_grads = self.y.t() @ self.lambd
        else:
            new_grads = torch.empty(self.grad_dim, device=self.device)

        # 5. 将新的共享梯度广播给所有 rank，并更新共享模型的梯度
        if multi_gpu:
            if rank == 0:
                new_grads_tensor = new_grads
            else:
                new_grads_tensor = torch.empty(self.grad_dim, device=self.device)
            dist.broadcast(new_grads_tensor, src=0)
            self._reset_grad(new_grads_tensor)
        else:
            self._reset_grad(new_grads)

        # 6. 同步 λ
        if rank == 0:
            lambd_val = self.lambd
        else:
            lambd_val = torch.zeros_like(self.lambd)
        if multi_gpu:
            dist.broadcast(lambd_val, src=0)
        # 返回更新后的 lambda（numpy 格式）以及统计量（若本次未计算则为 None）
        return lambd_val.detach().cpu().numpy(), stats
