import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from Encoder import _Encoder
from AffineRegression import _AffineRegression


class USARNet(nn.Module):
    def __init__(self, encoder, affineRegression):
        super(USARNet, self).__init__()

        self.encoder = encoder
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
        fixed_loc = torch.flatten(self.encoder(fixed), 1)
        moving_loc = torch.flatten(self.encoder(moving), 1)
        concated = torch.cat((fixed_loc, moving_loc), 1)

        theta = self.affineRegression(concated)
        theta = theta.view(-1, 3, 4)
        return theta


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

    ENCODER_CONFIG = (4, 4, 4, 4)
    DIM = 64
    BATCH_SIZE = 4
    INPUT_BATCH = torch.randn(BATCH_SIZE, 1, DIM, DIM, DIM)

    encoder = _Encoder(ENCODER_CONFIG=ENCODER_CONFIG, growth_rate=12, num_init_features=8)

    INPUT_SHAPE = (DIM // ((2**len(ENCODER_CONFIG))))**3 * 65 * 2

    affineRegression = _AffineRegression(
        num_input_parameters=INPUT_SHAPE,
        num_init_parameters=512,
        affine_config=(512, 256, 128, 64),
        reduction_rate=2,
        drop_rate=0
    )

    net = USARNet(encoder, affineRegression)

    #mov = torch.randn(1, 1, 300, 300, 300)

    print(net(INPUT_BATCH, INPUT_BATCH))
