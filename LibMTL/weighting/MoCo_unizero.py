import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class MoCo(AbsWeighting):
    r"""MoCo.
    
    This method is proposed in `Mitigating Gradient Bias in Multi-objective Learning: A Provably Convergent Approach (ICLR 2023) <https://openreview.net/forum?id=dLAYGdKTi2>`_ \
    and implemented based on the author' sharing code (Heshan Fernando: fernah@rpi.edu). 

    Args:
        MoCo_beta (float, default=0.5): The learning rate of y.
        MoCo_beta_sigma (float, default=0.5): The decay rate of MoCo_beta.
        MoCo_gamma (float, default=0.1): The learning rate of lambd.
        MoCo_gamma_sigma (float, default=0.5): The decay rate of MoCo_gamma.
        MoCo_rho (float, default=0): The \ell_2 regularization parameter of lambda's update.

    .. warning::
            MoCo is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self, share_model, task_num, device):
        super(MoCo, self).__init__()
        self.share_model = share_model

        self.task_num = task_num
        self.device = device
    
    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        # TODO
        self.share_model.zero_grad(set_to_none=False)

    def get_share_params(self):
        return self.share_model.parameters()

    def init_param(self):
        self._compute_grad_dim()
        self.step = 0
        self.y = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        self.lambd = (torch.ones([self.task_num, ]) / self.task_num).to(self.device)
        
    def backward(self, losses, **kwargs):
        self.step += 1 # 增加迭代步数
        # beta, beta_sigma = kwargs['MoCo_beta'], kwargs['MoCo_beta_sigma']
        # gamma, gamma_sigma = kwargs['MoCo_gamma'], kwargs['MoCo_gamma_sigma']
        # rho = kwargs['MoCo_rho']

        if self.rep_grad:
            raise ValueError('No support method MoCo with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()

            # Debugging: print losses and device
            # print(f"losses: {[loss.item() for loss in losses]}, device: {losses[0].device}")
            
            # 计算任务的梯度
            grads = self._compute_grad(losses, mode='backward')
            
            # Debugging: print gradients' shape and device
            # for i, grad in enumerate(grads):
            #     print(f"grad[{i}]: shape: {grad.shape}, device: {grad.device}")
        
        # 使用 no_grad 块，避免记录梯度计算记录
        with torch.no_grad():
            for tn in range(self.task_num):
                # grads[tn] = grads[tn]/(grads[tn].norm()+1e-8)*losses[tn]
                # 检查并处理梯度中的 NaN 值
                if torch.isnan(grads[tn]).any() or torch.isnan(losses[tn]): # TODO
                    print(f"="*20)
                    print(f"Warning: NaN detected in task {tn}")
                    print(f"="*20)
                    # 将梯度设置为 0，并对损失值进行处理
                    grads[tn].zero_()  # 或者采取其他处理措施，例如跳过该梯度
                    losses[tn] = torch.tensor(0.0, device=self.device)  # 或者设置为一个合理的默认值

                # Debugging:
                norm = grads[tn].norm()
                # print(f"grads[{tn}]: {norm}")
                if norm.item() == 0:
                    print(f"Warning: Zero norm detected for task {tn}")
                grads[tn] = grads[tn] / (norm + 1e-8) * losses[tn]

        # self.y = self.y - (beta/self.step**beta_sigma) * (self.y - grads)
        # self.lambd = F.softmax(self.lambd - (gamma/self.step**gamma_sigma) * (self.y@self.y.t()+rho*torch.eye(self.task_num).to(self.device))@self.lambd, -1)
        
        # 更新跟踪变量 y，使用平滑公式
        # y_{k+1,m} = y_{k,m} - β (y_{k,m} - grads)
        self.y = self.y - 0.99 * (self.y - grads)

        # 更新任务权重 λ，根据正则化 MGDA 子问题
        # λ_{k+1} = softmax(λ_k - γ * (y^T y + ρI) λ_k)
        self.lambd = F.softmax(self.lambd - 10 * (self.y@self.y.t())@self.lambd, -1)
        
        # 根据更新后的 λ 和 y 计算新的梯度
        new_grads = self.y.t()@self.lambd
        
        # 重置共享模型的梯度为 new_grads
        self._reset_grad(new_grads)

        # 返回当前任务的权重 λ
        return self.lambd.detach().cpu().numpy()
