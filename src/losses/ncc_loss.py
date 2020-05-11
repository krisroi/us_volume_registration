import torch
import torch.nn as nn


class MaskedNCC(nn.Module):
    r""" Creates a criterion that uses zero-normalized cross-correlation between two input volumes together
        with a regularization function to compute the loss.

    Args:
        fixed_patch (tensor): fixed patch with shape [B, C, D, H, W]
        moving_patch (tensor): moving patch with shape [B, C, D, H, W]
        predicted_theta (tensor): predicted theta from network output
        weight (float): epoch dependent weight factor
        reduction (string, optional): reduction method for loss-function. Default: 'mean' (opt: 'mean', 'sum', 'none')

    Examples::
        >>> criterion = NCC(useRegularization)
        >>> fixed_patch = torch.randn(B, C, D, H, W)
        >>> moving_patch = torch.randn(B, C, D, H, W)
        >>> predicted_theta = net(fixed_patch, moving_patch)
        >>> predicted_deform = affine_transform(..)
        >>> loss = criterion(fixed_patch, predicted_deform, predicted_theta, weight, reduction(opt))
    """

    def __init__(self, useRegularization, device):
        super(MaskedNCC, self).__init__()

        self.useRegularization = useRegularization
        self.device = device

    def forward(self, fixed_patch, moving_patch, predicted_theta, weight, reduction='mean'):
        ncc, mask = normalized_cross_correlation(fixed_patch, moving_patch, reduction)
        if not self.useRegularization:
            weight = 0
        L_reg = regularization_loss(predicted_theta, weight, self.device)
        return ((1 - ncc) + L_reg), mask


def normalized_cross_correlation(fixed_patch, moving_patch, reduction):
    """Compute and return masked zero-ncc ([0, 1]).
       The function creates a mask and computes zero-ncc ONLY where the
       volumes overlap.
        Note:
            Reduction option 'None' should only be used when computing zero-ncc
            and not backpropagating loss. For backpropagation, the loss-matrix
            needs to be reduced to a single item.
    """
    fixed = (fixed_patch[:])
    moving = (moving_patch[:])

    mask = create_mask(fixed, moving)

    N = torch.sum(torch.ones_like(mask), (2, 3, 4), keepdim=True)

    masked_fixed_mean = torch.div(torch.mean(fixed, axis=(2, 3, 4), keepdim=True), N)
    masked_moving_mean = torch.div(torch.mean(moving, axis=(2, 3, 4), keepdim=True), N)

    fixed_variance = torch.div(torch.sum(torch.pow((fixed - masked_fixed_mean), 2),
                                         axis=(2, 3, 4), keepdim=True), N)
    moving_variance = torch.div(torch.sum(torch.pow((moving - masked_moving_mean), 2),
                                          axis=(2, 3, 4), keepdim=True), N)

    numerator = torch.mul((fixed - masked_fixed_mean) * mask, (moving - masked_moving_mean) * mask)
    denominator = torch.sqrt(fixed_variance * moving_variance)

    epsilon = 1e-08

    pixel_ncc = torch.div(numerator, (denominator + epsilon))
    ncc = torch.mean(pixel_ncc, axis=(2, 3, 4))

    if reduction == 'mean':
        ncc = torch.mean(ncc, dim=0)
    elif reduction == 'sum':
        ncc = torch.sum(ncc, dim=0)
    elif reduction == None:
        ncc = ncc

    return ncc, mask


def create_mask(fixed_patch, moving_patch):
    '''Creates sector-mask for ultrasound volumes
    '''
    fix_mask = torch.ne(fixed_patch, 0)
    mov_mask = torch.ne(moving_patch, 0)

    fix_mask = fix_mask.float()
    mov_mask = mov_mask.float()

    fix_mask = fix_mask.masked_fill(fix_mask == 0, 2)
    mov_mask = mov_mask.masked_fill(mov_mask == 0, 3)

    mask = torch.eq(fix_mask, mov_mask)
    mask_f = mask.float()

    return mask_f


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
