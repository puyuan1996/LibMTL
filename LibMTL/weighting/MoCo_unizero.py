import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from LibMTL.weighting.abstract_weighting import AbsWeighting
# 导入必要库，用于生成热力图
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def init_param(self, **kwargs):
        self._compute_grad_dim()
        self.step = 0
        # y: [task_num, grad_dim]
        self.y = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        # λ: [task_num, ]
        self.lambd = (torch.ones([self.task_num, ]) / self.task_num).to(self.device)

        # 参数设置，可通过 kwargs 覆盖默认值

        # moco param-v2
        self.beta = kwargs.get('MoCo_beta', 0.99)
        self.beta_sigma = kwargs.get('MoCo_beta_sigma', 0.3)
        self.gamma = kwargs.get('MoCo_gamma', 10)
        self.gamma_sigma = kwargs.get('MoCo_gamma_sigma', 0.3)

        # moco param-v3（如果需要可切换参数）
        self.beta = kwargs.get('MoCo_beta', 0.99)
        self.beta_sigma = kwargs.get('MoCo_beta_sigma', 0.5)
        self.gamma = kwargs.get('MoCo_gamma', 10)
        self.gamma_sigma = kwargs.get('MoCo_gamma_sigma', 0.5)

        self.rho = kwargs.get('MoCo_rho', 0)
        self.stat_interval = kwargs.get('MoCo_stat_interval', 1000)
        # stat_interval = kwargs.get('MoCo_stat_interval', 1)

    def _grad2vec(self):
        """
        将共享模型中各个参数的梯度以 vector 的形式拼接起来
        """
        vec = []
        for param in self.get_share_params():
            if param.grad is not None:
                vec.append(param.grad.view(-1))
            else:
                vec.append(torch.zeros_like(param).view(-1))
        return torch.cat(vec)

    def _reset_grad(self, new_grad):
        r"""
        根据 new_grad 的值，重新赋值给共享模型中各个参数的梯度
        """
        offset = 0
        for param in self.get_share_params():
            numel = param.data.numel()
            grad_segment = new_grad[offset: offset + numel].view_as(param)
            # param.grad = grad_segment.clone().to(param.device) # TODO: .to(param.device) is important for atari share encoder
            param.grad = grad_segment.clone()# TODO: .to(param.device) is important for atari share encoder
            
            offset += numel

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
        stats['grad_norms'] = grad_norms.detach().numpy()

        # 计算任务之间的 cosine similarity 矩阵
        dot_matrix = grads @ grads.t()
        norm_matrix = grad_norms.unsqueeze(1) * grad_norms.unsqueeze(0)
        cos_sim_matrix = dot_matrix / (norm_matrix + 1e-8)
        cos_sim_matrix = torch.clamp(cos_sim_matrix, -1, 1)
        stats['cos_sim_matrix'] = cos_sim_matrix.detach().cpu().numpy()
        # 计算除对角线外的平均 cosine similarity
        task_num = grads.shape[0]
        mask = ~torch.eye(task_num, dtype=torch.bool, device=cos_sim_matrix.device)
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
            # 判断是否为多 GPU 环境
            multi_gpu = self.multi_gpu and dist.is_initialized()
            rank = dist.get_rank() if multi_gpu else 0
            self.device = f'cuda:{rank}'

            # 将 self.y 与 self.lambd 转移到当前设备，避免设备不一致问题
            self.y = self.y.to(self.device)
            self.lambd = self.lambd.to(self.device)

            self.step += 1  # 更新迭代步数

            if self.rep_grad:
                raise ValueError('不支持方法 MoCo 使用 representation gradients (rep_grad=True)')

            self._compute_grad_dim()

            # 1. 每个 rank 计算本地任务的梯度
            local_task_num = len(losses)
            local_grads = torch.zeros(local_task_num, self.grad_dim, device=self.device)
            for i in range(local_task_num):
                # 对每个任务的 loss 调用 backward 计算全网络梯度
                if i != local_task_num - 1:
                    losses[i].backward(retain_graph=True)
                else:
                    losses[i].backward()
                # 提取共享参数的梯度
                local_grads[i] = self._grad2vec()
                # 清零共享参数梯度，防止梯度累加
                self.zero_grad_share_params()

            # 2. 汇总各 rank 上的梯度
            if multi_gpu:
                world_size = dist.get_world_size()
                # 收集每个 rank 上真实的任务数
                all_local_task_nums = [None for _ in range(world_size)]
                dist.all_gather_object(all_local_task_nums, local_task_num)

                # if rank == 0:
                #     print("=" * 20)
                #     print(f"all_local_task_nums:{all_local_task_nums}")
                #     print("=" * 20)

                max_local_task_num = max(all_local_task_nums)
                # 若当前 rank 任务数不足，则 pad 0，以达到一致形状
                if local_task_num < max_local_task_num:
                    pad_tensor = torch.zeros(max_local_task_num - local_task_num, self.grad_dim, device=self.device)
                    local_grads = torch.cat([local_grads, pad_tensor], dim=0)
                # 将 local_grads 移至 CPU
                local_grads_cpu = local_grads.cpu()
                all_local_grads = [None for _ in range(world_size)]
                dist.all_gather_object(all_local_grads, local_grads_cpu)
                if rank == 0:
                    valid_grad_list = []
                    for i, tensor_cpu in enumerate(all_local_grads):
                        valid_count = all_local_task_nums[i]
                        tensor_valid = tensor_cpu[:valid_count, :].to(self.device)
                        valid_grad_list.append(tensor_valid)
                    all_task_grads = torch.cat(valid_grad_list, dim=0)
                    if all_task_grads.shape[0] != self.task_num:
                        raise ValueError(f"Aggregated tasks mismatch: got {all_task_grads.shape[0]} tasks, expected {self.task_num}.")
                    grads = all_task_grads
                else:
                    grads = torch.empty(self.task_num, self.grad_dim, device=self.device)
            else:
                if local_task_num != self.task_num:
                    raise ValueError("在单 GPU 模式下，loss 数量应等于 self.task_num.")
                grads = local_grads

            # 3. 同步各 rank 上的 loss 值
            if multi_gpu:
                local_losses_list = [loss.detach() for loss in losses]
                all_losses_lists = [None for _ in range(world_size)]
                dist.all_gather_object(all_losses_lists, local_losses_list)
                if rank == 0:
                    aggregated_losses = []
                    for loss_list in all_losses_lists:
                        aggregated_losses.extend(loss_list)
                    if len(aggregated_losses) != self.task_num:
                        raise ValueError(f"Aggregated losses mismatch: got {len(aggregated_losses)} losses, expected {self.task_num}.")
                else:
                    aggregated_losses = None
            else:
                aggregated_losses = losses

            # 可选：保存原始 grads 用于统计
            raw_grads = grads.clone().cpu()
            stats = None  # 默认不计算统计量
            if rank == 0:
                # print("=" * 20)
                # print("we are in moco")
                # print(f"len(aggregated_losses):{len(aggregated_losses)}, aggregated_losses:{aggregated_losses}")
                # print("=" * 20)
                with torch.no_grad():
                    # 归一化梯度并乘以对应 loss 值
                    for tn in range(self.task_num):
                        loss_val = aggregated_losses[tn].to(self.device)
                        norm = grads[tn].norm()
                        if norm.item() == 0:
                            print(f"Warning: 任务 {tn} 的梯度范数为 0")
                        grads[tn] = grads[tn] / (norm + 1e-8) * loss_val
                    # 平滑更新 y（采用固定平滑系数，也可根据步数调整）
                    self.y = self.y - 0.99 * (self.y - grads)
                    # 更新 λ（基于正则化 MGDA 子问题）
                    self.lambd = F.softmax(self.lambd - 10 * (self.y @ self.y.t()) @ self.lambd, dim=-1)
                    new_grads = self.y.t() @ self.lambd
            else:
                new_grads = torch.empty(self.grad_dim, device=self.device)

            # 4. 广播新的共享梯度并更新模型参数梯度
            if multi_gpu:
                if rank == 0:
                    new_grads_tensor = new_grads
                else:
                    new_grads_tensor = torch.empty(self.grad_dim, device=self.device)
                dist.broadcast(new_grads_tensor, src=0)
                self._reset_grad(new_grads_tensor)
            else:
                self._reset_grad(new_grads)

            # 5. 同步 λ
            if rank == 0:
                lambd_val = self.lambd
            else:
                lambd_val = torch.zeros_like(self.lambd)
            if multi_gpu:
                dist.broadcast(lambd_val, src=0)
            return lambd_val.detach().cpu().numpy(), stats