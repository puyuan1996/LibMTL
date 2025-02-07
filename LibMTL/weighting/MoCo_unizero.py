import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
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
    
    .. warning::
            MoCo 不支持 representation gradients，即 ``rep_grad`` 必须为 ``False``.
    """
    def __init__(self, share_model, task_num, device, multi_gpu=False):
        super(MoCo, self).__init__()
        self.share_model = share_model
        self.task_num = task_num
        self.device = device
        # 添加 multi_gpu 标志，用以区分单 GPU 和多 GPU 环境
        self.multi_gpu = multi_gpu
        # self.multi_gpu = True # TODO

    def zero_grad_share_params(self):
        r"""将共享参数的梯度置为零
        """
        self.share_model.zero_grad(set_to_none=False)

    def get_share_params(self):
        return self.share_model.parameters()

    def init_param(self):
        self._compute_grad_dim()
        self.step = 0
        self.y = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        self.lambd = (torch.ones([self.task_num, ]) / self.task_num).to(self.device)
        
    def backward(self, losses, **kwargs):
        """
        losses 为包含各个任务 loss 的列表（仅在 rank0 非空，其他 rank 为 None）
        """
        # 判断当前是否多 GPU（以及是否已初始化分布式通信）
        multi_gpu = self.multi_gpu and dist.is_initialized()
        rank = dist.get_rank() if multi_gpu else 0
        self.device = f'cuda:{rank}'

        self.step += 1  # 增加迭代步数

        # 参数设定，允许通过 kwargs 自定义更新参数
        beta = kwargs.get('MoCo_beta', 0.5)
        beta_sigma = kwargs.get('MoCo_beta_sigma', 0.5)
        gamma = kwargs.get('MoCo_gamma', 0.1)
        gamma_sigma = kwargs.get('MoCo_gamma_sigma', 0.5)
        rho = kwargs.get('MoCo_rho', 0)

        # 如果使用 representation gradients（不支持 MoCo），则报错
        if self.rep_grad:
            raise ValueError('不支持方法 MoCo 使用 representation gradients (rep_grad=True)')

        # rank0 负责真正的梯度计算和变量更新
        if rank == 0:
            self._compute_grad_dim()
            # 计算每个任务的梯度（假设 _compute_grad 已实现各任务梯度计算）
            grads = self._compute_grad(losses, mode='backward')

            # 对各任务梯度先归一化，再乘以对应 loss 进行缩放，同时对内部变量 y 进行平滑更新
            # 转换 losses 列表中所有 loss 到指定设备
            losses = [loss.to(self.device) for loss in losses]
            with torch.no_grad():
                for tn in range(self.task_num):
                    norm = grads[tn].norm()
                    if norm.item() == 0:
                        print(f"Warning: 任务 {tn} 的梯度范数为 0")
                    grads[tn] = grads[tn] / (norm + 1e-8) * losses[tn]
                # 平滑更新 y（beta 可能随 step 衰减）
                self.y = self.y - (beta / self.step**beta_sigma) * (self.y - grads)
                # 更新任务权重 λ（基于正则化 MGDA 子问题）
                self.lambd = F.softmax(self.lambd - (gamma / self.step**gamma_sigma) *
                                    (self.y @ self.y.t() + rho * torch.eye(self.task_num, device=self.device)) @ self.lambd, dim=-1)
                # 根据更新后的 λ 和 y 计算新的共享梯度
                new_grads = self.y.t() @ self.lambd
        else:
            # 非 rank0 不接收 loss，不计算梯度，创建占位 tensor
            new_grads = torch.empty(self.grad_dim, device=self.device)

        # 分布式环境下，仅由 rank0 更新共享模型的梯度，然后广播到其他 GPU
        if multi_gpu:
            # 无论是 rank0 还是其他 rank，都准备一个 tensor 用于梯度更新
            if rank == 0:
                new_grads_tensor = new_grads
            else:
                new_grads_tensor = torch.empty(self.grad_dim, device=self.device)
                
            # 让所有进程都调用广播，广播时 rank0 将 new_grads_tensor 的内容发送出去
            dist.broadcast(new_grads_tensor, src=0)
            
            # 将接收到或本身生成的 new_grads_tensor 传入 _reset_grad，更新共享模型的梯度
            self._reset_grad(new_grads_tensor)
            
            # 接下来，为确保所有 GPU 模型参数梯度一致，
            # 让所有 rank 对每个共享模型参数的 .grad 进行同步
            for p in self.share_model.parameters():
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.broadcast(p.grad, src=0)
        else:
            # 单 GPU 情况下直接更新共享模型梯度
            self._reset_grad(new_grads)

        # 同理，只在 rank0 内获得当前更新后的 λ，其他 rank 从 rank0 接收 λ
        if rank == 0:
            lambd_val = self.lambd
            print(f'lambd_val:{lambd_val}')
        else:
            # 创建一个空的 numpy 数组，然后转换为 torch.Tensor
            lambd_val = torch.zeros_like(torch.tensor(self.lambd))

        if multi_gpu:
            # 将 lambd_val 转换为 torch.Tensor
            lambd_val = torch.tensor(lambd_val) if not isinstance(lambd_val, torch.Tensor) else lambd_val
            # broadcast 是一个原地操作，不需要赋值给 lambd_val
            dist.broadcast(lambd_val, src=0)

        return lambd_val.detach().cpu().numpy()
