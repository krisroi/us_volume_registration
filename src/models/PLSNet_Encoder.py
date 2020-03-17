import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools


class _DSConv(nn.Module):
    def __init__(self, num_input_features, growth_rate, padding, dilation):
        super(_DSConv, self).__init__()

        self.add_module('dwConv', nn.Conv3d(num_input_features, num_input_features, kernel_size=3,
                                            stride=1, padding=padding, dilation=dilation,
                                            groups=num_input_features, bias=False))
        self.add_module('conv', nn.Conv3d(num_input_features, growth_rate, kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('norm', nn.BatchNorm3d(growth_rate))
        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = torch.cat(x, 1)
        out = self.dwConv(out)
        out = self.relu(self.norm(self.conv(out)))
        return out


class _StridedDSConv(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_StridedDSConv, self).__init__()

        self.add_module('dwConv', nn.Conv3d(num_input_features, num_input_features, kernel_size=3,
                                            stride=1, padding=1, groups=num_input_features,
                                            bias=False))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1,
                                          stride=2, bias=False))
        self.add_module('norm', nn.BatchNorm3d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.dwConv(x)
        out = self.relu(self.norm(self.conv(out)))
        return out


class _DilatedResidualDenseBlock(nn.ModuleDict):

    __constants__ = ['intermediate']

    def __init__(self, num_layers, num_input_features, growth_rate):
        super(_DilatedResidualDenseBlock, self).__init__()

        self.intermediate = nn.ModuleDict()

        for i in range(num_layers):
            layer = _DSConv(
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                padding=(i + 1),
                dilation=(i + 1)
            )
            self.intermediate.add_module('DSConvLayer%d' % (i + 1), layer)

        self.add_module('conv', nn.Conv3d(num_input_features + num_layers * growth_rate, num_input_features,
                                          kernel_size=1, stride=1, bias=False))

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.intermediate.items():
            new_features = layer(features)
            features.append(new_features)
        concat = torch.cat(features, 1)
        out = self.conv(concat)
        out = torch.add(out, init_features)
        return out


class _PLSNet(nn.Module):
    r"""PLS-Net model class, based on
    `"Efficient 3D Fully Convolutional Networks for Pulmonary Lobe Segmentation in CT Images"               <https://arxiv.org/pdf/1909.07474v1.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each DRDB block
        num_init_features (int) - the number of filters to learn in the first convolution layer
    NOTE: This inplementation does not include using mixed precision training strategy, which is a big part of the idea of the network.
    """

    __constants__ = ['features']

    def __init__(self, encoder_config, growth_rate, num_init_features):
        super(_PLSNet, self).__init__()

        self.encoder_config = encoder_config

        self.drd_module = nn.ModuleDict()
        self.strided_conv_module = nn.ModuleDict()

        # Initial StridedDSConv
        self.add_module('conv0', nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1,
                                           groups=1, bias=False))
        self.add_module('conv1', nn.Conv3d(1, num_init_features, kernel_size=1,
                                           stride=2, bias=False))
        self.add_module('norm', nn.BatchNorm3d(num_init_features))
        self.add_module('relu', nn.ReLU(inplace=True))

        num_features = num_init_features + 1
        for i, num_layers in enumerate(encoder_config):
            DRD_BLOCK = _DilatedResidualDenseBlock(num_layers=num_layers,
                                                   num_input_features=num_features,
                                                   growth_rate=growth_rate)
            self.drd_module.add_module('DRD_BLOCK%d' % (i + 1), DRD_BLOCK)

            if i != len(encoder_config) - 1:

                STRIDED_CONV_LAYER = _StridedDSConv(num_input_features=num_features,
                                                    num_output_features=num_init_features * (2**(i + 1)))
                self.strided_conv_module.add_module('DS_LAYER%d' % (i + 1), STRIDED_CONV_LAYER)
            num_features = num_init_features * (2**(i + 1)) + 1

    def forward(self, x):
        origInput = x
        out = self.relu(self.norm(self.conv1(self.conv0(x))))

        for i, ((DRD_NAME, DRD_BLOCK), (STRIDED_CONV_NAME, STRIDED_CONV_LAYER)) in \
                enumerate(itertools.zip_longest(self.drd_module.items(), self.strided_conv_module.items(),
                                                fillvalue=(0, 'placeholder'))):
            downsampled_data = F.interpolate(input=origInput,
                                             scale_factor=(1 / (2 ** (i + 1))),
                                             mode='trilinear')
            out = torch.cat((out, downsampled_data), 1)
            out = DRD_BLOCK(out)
            if i != len(self.encoder_config) - 1:
                out = STRIDED_CONV_LAYER(out)
        out = F.relu(out, inplace=True)
        return out
