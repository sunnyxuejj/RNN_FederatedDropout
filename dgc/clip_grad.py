import torch
from torch._six import inf

__all__ = ['clip_grad_norm_', 'clip_grad_value_']


# code modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/clip_grad.py
def clip_grad_norm_(grad, max_norm, norm_type=2):
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = grad.data.abs().max()
    else:
        total_norm = grad.data.norm(norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grad.data.mul_(clip_coef)
    return grad


def clip_grad_value_(grad, clip_value):
    clip_value = float(clip_value)
    grad.data.clamp_(min=-clip_value, max=clip_value)

