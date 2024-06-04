import torch
import torch.nn.functional as F
from LibMTL.weighting.abstract_weighting import AbsWeighting

class FAMO(AbsWeighting):
    r"""FAMO: A method for multi-task learning that adjusts task weights dynamically.
    
    This method is implemented based on the official implementation.

    Args:
        n_tasks (int): Number of tasks.
        device (torch.device): Device to use for computation.
        gamma (float, default=1e-5): Weight decay factor for the optimizer.
        w_lr (float, default=0.025): Learning rate for the optimizer.
    
    """

    def __init__(self, share_model, task_num, device, gamma=1e-5, w_lr=0.025):
        super(FAMO, self).__init__()
        self.share_model = share_model
        self.task_num = task_num
        self.device = device
        self.gamma = gamma
        self.w_lr = w_lr
        self.min_losses = torch.zeros(task_num).to(device)
        self.w = torch.tensor([0.0] * task_num, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = 1

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        self.share_model.zero_grad(set_to_none=False)

    def get_share_params(self):
        return self.share_model.parameters()

    def init_param(self):
        self.step = 0
        self.min_losses = torch.zeros(self.task_num).to(self.device)
        self.w = torch.tensor([0.0] * self.task_num, device=self.device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=self.w_lr, weight_decay=self.gamma)

    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses, **kwargs):
        z = F.softmax(self.w, dim=-1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z, "logits": self.w.detach().clone()}

    def backward(self, losses, **kwargs):
        loss, extra_outputs = self.get_weighted_loss(losses)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.get_share_params(), self.max_norm)
        loss.backward()
        return loss, extra_outputs

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, dim=-1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()