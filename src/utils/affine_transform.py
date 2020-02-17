import torch
import torch.nn.functional as F


def affine_grid_3d(size, theta):
    """ Defines an affine grid in 3 dimensions.
    Args:
        size (tuple): tuple of ints containing the dimensions to the moving patch
            B = batch size
            C = number of channels
            D, H, W = dimensions of the input volume in depth, height and width
        theta (tensor): predicted deformation matrix
    Returns:
        A 3d affine grid that is used for transformation
        Return size: [B, D, H, W, 3]
    """
    B, C, D, H, W = size  # Extract dimensions of the input

    theta = theta.expand(B, 3, 4)  # expand to the number of batches you need

    grid = F.affine_grid(theta, size=(B, C, D, H, W))
    grid = grid.view(B, D, H, W, 3)

    return grid


def affine_transform(moving_patch, theta):
    """ Performs an affine transform of some input data with a transformation matrix theta
    Args:
        moving_patch (tensor): input data to transform
        theta (tensor): predicted deformation matrix
    Returns:
        Transformed input data that is transformed with the transformation matrix.
    """
    B, C, D, H, W = moving_patch.shape  # Extracting the dimensions

    grid_3d = affine_grid_3d((B, C, D, H, W), theta)
    warped_patches = F.grid_sample(moving_patch, grid_3d, mode='bilinear',
                                   padding_mode='border', align_corners=True)  #(mode opt: 'nearest', bilinear' - padding_mode opt: 'border', 'zeros', 'reflection')

    return warped_patches
