import torch
import torch.nn as nn

from utils.affine_transform import affine_transform


class USARNet(nn.Module):
    r"""Proposed network class for affine ultrasound to ultrasound image registration.
    Args:
        encoder (_Encoder class): configured encoder
        affineRegression (_AffineRegression class): configured affine regressor
    """

    def __init__(self, encoder, affineRegression):
        super(USARNet, self).__init__()

        self._encoder = encoder
        self.affineRegression = affineRegression
        self.flatten = nn.Flatten()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fixed_batch, moving_batch):
        fixed_loc = self._encoder(fixed_batch)
        moving_loc = self._encoder(moving_batch)
        fixed_loc = self.flatten(fixed_loc)
        moving_loc = self.flatten(moving_loc)
        concated = torch.cat((fixed_loc, moving_loc), 1)

        theta = self.affineRegression(concated)
        theta = theta.view(-1, 3, 4)
        
        return theta