import torch
import torch.nn as nn

from collections import OrderedDict


class _FullyConnectedBlock(nn.Module):
    def __init__(self, num_input_parameters, reduction_rate, drop_rate):
        super(_FullyConnectedBlock, self).__init__()

        self.add_module('fc', nn.Linear(num_input_parameters, num_input_parameters // reduction_rate))
        self.add_module('norm', nn.BatchNorm1d(num_input_parameters // reduction_rate))
        self.add_module('relu', nn.ReLU(inplace=True))

        self.dropout = nn.Dropout(p=drop_rate)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_parameters = self.relu(self.norm(self.fc(x)))
        if self.drop_rate > 0:
            new_parameters = self.dropout(new_parameters)
        return new_parameters


class _AffineRegression(nn.Module):

    __constants__ = ['params']

    def __init__(self, num_input_parameters, num_init_parameters, affine_config,
                 reduction_rate, drop_rate=0):
        super(_AffineRegression, self).__init__()

        # Initial fully connected layer. Takes num_input_features from the encoder and outputs
        # desired number of output features from the first FC layer.
        self.params = nn.Sequential(OrderedDict([
            ('fc_in', nn.Linear(num_input_parameters, num_init_parameters)),
            ('norm_in', nn.BatchNorm1d(num_init_parameters)),
            ('relu_in', nn.ReLU(inplace=True))
        ]))

        # Adds fully connected block, specified by num_blocks. Enumerate all but
        # last element in config file and use that in the last fc_out layer.
        for i, num_parameters in enumerate(affine_config[:-1]):
            block = _FullyConnectedBlock(
                num_input_parameters=num_parameters,
                reduction_rate=reduction_rate,
                drop_rate=drop_rate
            )
            self.params.add_module('fc_block%d' % (i), block)

        # Final FC layer. This layer produces the 12 output transformation parameters.
        self.params.add_module('fc_out', nn.Linear(affine_config[-1], 12))

        # Initializing last fully connected layer to the identity matrix
        self.params.fc_out.weight.data.zero_()
        self.params.fc_out.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float64))

    def forward(self, x):
        params = self.params(x)
        return params


if __name__ == '__main__':

    affineRegression = _AffineRegression(num_input_parameters=2032,
                                         num_init_parameters=512,
                                         affine_config=(512, 256, 128, 64),
                                         reduction_rate=2,
                                         drop_rate=0
                                         )
    fix = torch.randn(2, 2032)
    print(affineRegression(fix))
