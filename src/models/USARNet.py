import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class USARNet(nn.Module):
    r"""Proposed network class for affine ultrasound to ultrasound image registration.
    Args:
        encoder (_Encoder class): configured encoder
        affineRegression (_AffineRegression class): configures affine regressor.
    """

    def __init__(self, encoder, affineRegression):
        super(USARNet, self).__init__()

        self.fixedEncoder = encoder
        self.movingEncoder = encoder
        self.affineRegression = affineRegression

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

    def forward(self, fixed, moving):
        fixed_loc = torch.flatten(self.fixedEncoder(fixed), 1)
        moving_loc = torch.flatten(self.movingEncoder(moving), 1)
        print(fixed_loc == moving_loc)
        concated = torch.cat((fixed_loc, moving_loc), 1)

        theta = self.affineRegression(concated)
        theta = theta.view(-1, 3, 4)
        return theta


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

    from Encoder import _Encoder
    from AffineRegression import _AffineRegression

    ENCODER_CONFIG = (4, 4, 4, 4)
    DIM = 64
    BATCH_SIZE = 4
    INPUT_BATCH = torch.randn(BATCH_SIZE, 1, DIM, DIM, DIM)

    encoder = _Encoder(encoder_config=ENCODER_CONFIG, growth_rate=12, num_init_features=8)

    INPUT_SHAPE = (DIM // ((2**len(ENCODER_CONFIG))))**3 * 65 * 2

    affineRegression = _AffineRegression(
        num_input_parameters=INPUT_SHAPE,
        num_init_parameters=512,
        affine_config=(512, 256, 128, 64),
        drop_rate=0
    )

    net = USARNet(encoder, affineRegression)

    #mov = torch.randn(1, 1, 300, 300, 300)

    print(net(INPUT_BATCH, INPUT_BATCH))
