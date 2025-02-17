import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from LibMTL.weighting.abstract_weighting import AbsWeighting
# 导入必要库，用于生成热力图
import matplotlib.pyplot as plt
import seaborn as sns
import math

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

        self._compute_group_indices()   # 预计算各组切片索引

        self.step = 0
        # y: [task_num, grad_dim]
        self.y = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        # λ: [task_num, ]
        self.lambd = (torch.ones([self.task_num, ]) / self.task_num).to(self.device)

        # 参数设置，可通过 kwargs 覆盖默认值
        # self.beta = kwargs.get('MoCo_beta', 0.5)
        # self.beta_sigma = kwargs.get('MoCo_beta_sigma', 0.5)
        # self.gamma = kwargs.get('MoCo_gamma', 0.1)
        # self.gamma_sigma = kwargs.get('MoCo_gamma_sigma', 0.5)

        # moco param-v2
        self.beta = kwargs.get('MoCo_beta', 0.99)
        self.beta_sigma = kwargs.get('MoCo_beta_sigma', 0.3)
        self.gamma = kwargs.get('MoCo_gamma', 10)
        self.gamma_sigma = kwargs.get('MoCo_gamma_sigma', 0.3)

        # moco param-v3
        self.beta = kwargs.get('MoCo_beta', 0.99)
        self.beta_sigma = kwargs.get('MoCo_beta_sigma', 0.5)
        self.gamma = kwargs.get('MoCo_gamma', 10)
        self.gamma_sigma = kwargs.get('MoCo_gamma_sigma', 0.5)

        self.rho = kwargs.get('MoCo_rho', 0)
        self.stat_interval = kwargs.get('MoCo_stat_interval', 1000)
        # stat_interval = kwargs.get('MoCo_stat_interval', 1)

    def _compute_group_indices(self):
        """
        利用 share_model.get_group_parameters() 计算各组在全局梯度中对应的起始与终止索引，
        要求 share_model.parameters() 返回的参数顺序与 get_group_parameters() 中各组参数顺序一致。
        """
        group_params = self.share_model.get_group_parameters()
        # 下面按照字典的插入顺序遍历（Python3.7+ dict 保持插入顺序）
        self.group_indices = {}
        offset = 0
        for group, params in group_params.items():
            numel = sum([p.data.numel() for p in params])
            self.group_indices[group] = (offset, offset + numel)
            offset += numel

    def _compute_statistics_from_slice(self, raw_grads, start, end):
        """
        针对 raw_grads 的切片部分计算统计量，
        raw_grads: [task_num, grad_dim] 整体梯度矩阵，
        返回该部分的统计信息。
        """
        slice_grads = raw_grads[:, start:end]
        stats = {}
        grad_norms = torch.norm(slice_grads, dim=1)
        stats['grad_norms'] = grad_norms.detach().cpu().numpy()

        dot_matrix = slice_grads @ slice_grads.t()
        norm_matrix = grad_norms.unsqueeze(1) * grad_norms.unsqueeze(0)
        cos_sim_matrix = dot_matrix / (norm_matrix + 1e-8)
        cos_sim_matrix = torch.clamp(cos_sim_matrix, -1, 1)
        stats['cos_sim_matrix'] = cos_sim_matrix.detach().cpu().numpy()

        task_num = slice_grads.shape[0]
        mask = ~torch.eye(task_num, dtype=torch.bool, device=slice_grads.device)
        avg_cos_sim = cos_sim_matrix[mask].mean().item()
        stats['avg_cos_sim'] = avg_cos_sim
        return stats

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
        # import ipdb;ipdb.set_trace()
        stats = {}
        # 计算每个任务梯度的范数
        grad_norms = torch.norm(grads, dim=1)
        stats['grad_norms'] = grad_norms.detach().numpy()

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
        device = cos_sim_matrix.device  # 获取 cos_sim_matrix 的设备
        mask = ~torch.eye(task_num, dtype=torch.bool, device=device)

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

        if self.rep_grad:
            raise ValueError('不支持方法 MoCo 使用 representation gradients (rep_grad=True)')

        self._compute_grad_dim()

        # 1. 每个 rank 先计算本地任务的梯度
        local_task_num = len(losses)
        local_grads = torch.zeros(local_task_num, self.grad_dim, device=self.device)

        for i in range(local_task_num):
            # 对每个任务的 loss 调用 backward 计算全网络梯度（共享部分和 head 部分）
            # 注意：由于后续需要多次计算共享部分梯度，非最后一次调用 retain_graph=True
            if i != local_task_num - 1:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            # 提取共享参数的梯度（encoder/backbone），head 部分的梯度不受影响
            local_grads[i] = self._grad2vec()

            # 清零共享参数的梯度，避免后续任务的共享梯度发生累加
            self.zero_grad_share_params()

        # 2. 汇总各 rank 上的梯度
        if multi_gpu:
            world_size = dist.get_world_size()
            # 利用 all_gather_object 收集每个 rank 的 local_grads
            # all_local_grads = [None for _ in range(world_size)]
            # dist.all_gather_object(all_local_grads, local_grads)

            # 先将本地梯度转到 CPU，释放 GPU 显存占用
            all_local_grads = [None for _ in range(world_size)]
            local_grads_cpu = local_grads.cpu()
            dist.all_gather_object(all_local_grads, local_grads_cpu)

            if rank == 0:
                # 拼接得到 all_task_grads, 总行数须等于 self.task_num
                # all_task_grads = torch.cat([g.to(self.device) for g in all_local_grads], dim=0)
                
                # 拼接得到所有任务的梯度，并转回 self.device
                all_task_grads = torch.cat([torch.as_tensor(g).to(self.device) for g in all_local_grads], dim=0)

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

        
        # 在循环之前保存原始 grads 用于统计（可选）
        raw_grads = grads.clone().cpu()

        stats = None  # 若本次未计算统计量则返回 None
        group_stats = {}

        # 4. 仅在 rank0 上进行 MoCo 梯正逻辑及统计量计算
        if rank == 0:
            with torch.no_grad():
                # 针对每个任务：归一化梯度并乘以对应 loss 值（loss_value 为标量）
                for tn in range(self.task_num):
                    loss_val = aggregated_losses[tn].to(self.device)
                    norm = grads[tn].norm()
                    if norm.item() == 0:
                        print(f"Warning: 任务 {tn} 的梯度范数为 0")
                    grads[tn] = grads[tn] / (norm + 1e-8) * loss_val

                # 先对各个分组利用预先计算的切片索引计算统计
                for group, (start, end) in self.group_indices.items():
                    group_stats[group] = self._compute_statistics_from_slice(raw_grads, start, end)
                    print(f"Step {self.step} 分组 {group} 平均 cosine similarity: {group_stats[group]['avg_cos_sim']:.4f}")

                # 若满足指定步数间隔，则计算并打印整体统计量，并绘制与保存热力图
                if self.step == 1 or self.step % self.stat_interval == 0:
                    stats = self._compute_statistics(raw_grads)  # 使用整体 raw_grads 统计
                    print(f"Step {self.step} 整体梯度统计量:")
                    print(f"  每任务梯度范数: {stats['grad_norms']}")
                    print(f"  平均任务间 cosine similarity: {stats['avg_cos_sim']}")
                    
                    # 组合各热力图：整体及各分组，每个子图显示相应的 cosine similarity 矩阵
                    n_plots = 1 + len(self.group_indices)  # 第一个为整体热力图，其后为各分组
                    # 为了显示清晰，统一采用垂直子图布局；也可根据需要改为横向或者网格布局
                    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 6 * n_plots))
                    # 如果只有一个子图，确保 axes 是列表
                    if n_plots == 1:
                        axes = [axes]
                    
                    # 绘制整体热力图
                    ax = axes[0]
                    sns.heatmap(
                        stats['cos_sim_matrix'],
                        annot=True,
                        fmt=".2f",
                        cmap='RdBu',
                        cbar=True,
                        square=True,
                        vmin=-1,
                        vmax=1,
                        ax=ax
                    )
                    ax.set_title(f"Step {self.step} - Overall Cosine Similarity", fontsize=16)
                    ax.set_xlabel("Task Index", fontsize=14)
                    ax.set_ylabel("Task Index", fontsize=14)
                    
                    # 按各分组绘制热力图
                    for i, (group, (start, end)) in enumerate(self.group_indices.items(), start=1):
                        ax = axes[i]
                        group_matrix = group_stats[group]['cos_sim_matrix']
                        sns.heatmap(
                            group_matrix,
                            annot=True,
                            fmt=".2f",
                            cmap='RdBu',
                            cbar=True,
                            square=True,
                            vmin=-1,
                            vmax=1,
                            ax=ax
                        )
                        ax.set_title(f"Step {self.step} - {group} Cosine Similarity", fontsize=16)
                        ax.set_xlabel("Task Index", fontsize=14)
                        ax.set_ylabel("Task Index", fontsize=14)
                    
                    # 构造合理的目录结构和命名方式
                    # 设置 base_dir 路径，请根据实际项目目录进行调整
                    # base_dir = "/mnt/afs/niuyazhe/code/LightZero/dmc_uz_cos_sim_heatmap_layer/8games_notaskembed"
                    base_dir = "/mnt/afs/niuyazhe/code/LightZero/dmc_uz_cos_sim_heatmap_layer/8games_concattaskembed"

                    # 如果目录不存在，可提前创建（示例中略去创建代码）
                    overall_save_path = f"{base_dir}/combined_cos_sim_heatmap_step_{self.step}.png"
                    
                    plt.tight_layout()
                    plt.savefig(overall_save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Combined Task cosine similarity 热力图已保存至 {overall_save_path}")

                # 平滑更新 y，beta 衰减可能与迭代步数有关
                # self.y = self.y - (self.beta / (self.step ** self.beta_sigma)) * (self.y - grads)
                # # 更新任务权重 λ（基于正则化 MGDA 子问题）
                # self.lambd = F.softmax(
                #     self.lambd - (self.gamma / (self.step ** self.gamma_sigma)) *
                #     (self.y @ self.y.t() + self.rho * torch.eye(self.task_num, device=self.device)) @ self.lambd,
                #     dim=-1
                # )

                # moco param-v0
                # 使用平滑公式，更新跟踪变量 y
                self.y = self.y - 0.99 * (self.y - grads)
                # 更新任务权重 λ（基于正则化 MGDA 子问题）
                self.lambd = F.softmax(self.lambd - 10 * (self.y@self.y.t())@self.lambd, -1)

                # 根据更新后的 λ 与 y，计算新的共享梯度
                new_grads = self.y.t() @ self.lambd
        else:
            new_grads = torch.empty(self.grad_dim, device=self.device)

        # 5. 将新的共享梯度广播给所有 rank，并更新共享模型的梯度
        # if multi_gpu:
        #     if rank == 0:
        #         new_grads_tensor = new_grads
        #     else:
        #         new_grads_tensor = torch.empty(self.grad_dim, device=self.device)
        #     dist.broadcast(new_grads_tensor, src=0)
        #     self._reset_grad(new_grads_tensor)
        # else:
        #     self._reset_grad(new_grads)

        # 6. 同步 λ
        if rank == 0:
            lambd_val = self.lambd
        else:
            lambd_val = torch.zeros_like(self.lambd)
        if multi_gpu:
            dist.broadcast(lambd_val, src=0)
        # 返回更新后的 lambda（numpy 格式）以及统计量（若本次未计算则为 None）
        return lambd_val.detach().cpu().numpy(), stats
