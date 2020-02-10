import torch
import torch.nn as nn


class NCC(nn.Module):
    """ Defines 1-ncc loss of two volumes
    Args:
        fixed_patch (tensor): fixed patch with shape [B, C, D, H, W]
        moving_patch (tensor): moving patch with shape [B, C, D, H, W]
        reduction (string, optional): reduction method for loss-function. Default: 'mean' (opt: 'mean', 'sum')
    """

    def __init__(self):
        super(NCC, self).__init__()

    def forward(self, fixed_patch, moving_patch, reduction):
        # Creates a forward pass for the loss function
        return normalized_cross_correlation(fixed_patch, moving_patch, reduction)


def normalized_cross_correlation(fixed_patch, moving_patch, reduction='mean'):
    """Compute ncc and return 1 - ncc as loss-function
    """

    fixed = (fixed_patch[:] - torch.mean(fixed_patch, (2, 3, 4), keepdim=True))
    moving = (moving_patch[:] - torch.mean(moving_patch, (2, 3, 4), keepdim=True))

    fixed_variance = torch.sqrt(torch.sum(torch.pow(fixed, 2), (2, 3, 4)))
    moving_variance = torch.sqrt(torch.sum(torch.pow(moving, 2), (2, 3, 4)))

    num = torch.sum(torch.mul(fixed, moving), (2, 3, 4))
    den = torch.mul(fixed_variance, moving_variance)

    alpha = 1.0e-16  # small number to prevent zero-division
    ncc = torch.div(num, (den + alpha))

    if reduction == 'mean':
        ncc = torch.mean(ncc, dim=0)
    elif reduction == 'sum':
        ncc = torch.sum(ncc, dim=0)

    return 1 - ncc


def regularizer_loss():
    pass
