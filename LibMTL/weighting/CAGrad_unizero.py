import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
from LibMTL.weighting.abstract_weighting import AbsWeighting

class CAGrad(AbsWeighting):
    r"""Conflict-Averse Gradient descent (CAGrad).
    
    This method is proposed in `Conflict-Averse Gradient Descent for Multi-task learning (NeurIPS 2021) <https://openreview.net/forum?id=_61Qh8tULj_>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/Cranial-XIX/CAGrad>`_. 

    Args:
        calpha (float, default=0.5): A hyperparameter that controls the convergence rate.
        rescale ({0, 1, 2}, default=1): The type of the gradient rescaling.

    .. warning::
            CAGrad is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self, share_model, task_num, device):
        super(CAGrad, self).__init__()
        # self.learn_model = learn_model
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
        calpha, rescale = kwargs['calpha'], kwargs['rescale']
        if self.rep_grad:
            raise ValueError('No support method CAGrad with representation gradients (rep_grad=True)')
#             per_grads = self._compute_grad(losses, mode='backward', rep_grad=True)
#             grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward')
        
        GG = torch.matmul(grads, grads.t()).cpu() # [num_tasks, num_tasks]
        g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient

        x_start = np.ones(self.task_num) / self.task_num
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (calpha*g0_norm+1e-8).item()
        def objfn(x):
            return (x.reshape(1,-1).dot(A).dot(b.reshape(-1,1))+c*np.sqrt(x.reshape(1,-1).dot(A).dot(x.reshape(-1,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(self.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-8)
        g = grads.mean(0) + lmbda * gw
        if rescale == 0:
            new_grads = g
        elif rescale == 1:
            new_grads = g / (1+calpha**2)
        elif rescale == 2:
            new_grads = g / (1 + calpha)
        else:
            raise ValueError('No support rescale type {}'.format(rescale))
        self._reset_grad(new_grads)
        return w_cpu
