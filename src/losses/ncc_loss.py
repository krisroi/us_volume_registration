import torch
import torch.nn as nn


class NCC(nn.Module):
    r""" Creates a criterion that uses zero-normalized cross-correlation between two input volumes together 
        with a regularization function to compute the loss.

    Args:
        fixed_patch (tensor): fixed patch with shape [B, C, D, H, W]
        moving_patch (tensor): moving patch with shape [B, C, D, H, W]
        predicted_theta (tensor): predicted theta from network output
        weight (float): epoch dependent weight factor
        reduction (string, optional): reduction method for loss-function. Default: 'mean' (opt: 'mean', 'sum')

    Examples::
        >>> criterion = NCC(useRegularization)
        >>> fixed_patch = torch.randn(B, C, D, H, W)
        >>> moving_patch = torch.randn(B, C, D, H, W)
        >>> predicted_theta = net(fixed_patch, moving_patch)
        >>> predicted_deform = affine_transform(..)
        >>> loss = criterion(fixed_patch, predicted_deform, predicted_theta, weight, reduction(opt))
    """

    def __init__(self, useRegularization, device):
        super(NCC, self).__init__()

        self.useRegularization = useRegularization
        self.device = device

    def forward(self, fixed_patch, moving_patch, predicted_theta, weight, reduction='mean'):
        ncc = normalized_cross_correlation(fixed_patch, moving_patch, reduction)
        if not self.useRegularization:
            weight = 0
        L_reg = regularization_loss(predicted_theta, weight, self.device)
        return (1 - ncc) + L_reg


def normalized_cross_correlation(fixed_patch, moving_patch, reduction):
    """Compute and return zncc ([0, 1])
        Note:
            Reduction option 'None' should only be used when computing zero-ncc
            and not backpropagating loss. For backpropagation, the loss-matrix
            needs to be reduced to a single item.
    """
    fixed = (fixed_patch[:])
    moving = (moving_patch[:])

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
    elif reduction == None:
        ncc = ncc

    return ncc


def extract(predicted_theta):
    """Extract rotation (A) and translation (b) part from predicted theta
        and return together with an instance of the identity matrix.
    """
    IDT = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32)
    IDT = IDT.view(-1, 3, 3)

    A = predicted_theta[:, :, 0:3]
    b = predicted_theta[:, :, 3]

    return IDT, A, b


def regularization_loss(predicted_theta, weight, device):
    """Regularization functions that computes the Frobenius norm.
    """
    IDT, A, b = extract(predicted_theta)
    return weight * ((torch.norm((A - IDT.to(device)), p='fro'))**2 + (torch.norm(b, p=2))**2)


def determinant_loss(predicted_theta):
    IDT, A, _ = extract(predicted_theta)
    return (-1 + torch.det(A + IDT))**2
