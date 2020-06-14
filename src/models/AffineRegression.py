import torch
import torch.nn as nn

from collections import OrderedDict


class _FullyConnectedBlock(nn.Module):
    """Creates a fully connected "block" consisting of a fully connected layer followed
        by batch normalization and relu activation
    """
    def __init__(self, num_input_parameters, num_output_parameters, drop_rate):
        super(_FullyConnectedBlock, self).__init__()

        self.add_module('fc', nn.Linear(num_input_parameters, num_output_parameters))
        self.add_module('norm', nn.BatchNorm1d(num_output_parameters))
        self.add_module('relu', nn.ReLU(inplace=True))

        self.dropout = nn.Dropout(p=drop_rate)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_parameters = self.relu(self.norm(self.fc(x)))
        if self.drop_rate > 0:
            new_parameters = self.dropout(new_parameters)
        return new_parameters


class _AffineRegression(nn.Module):
    r"""Affine regression model class. Takes a flattened array and regresses an affine
        3D transformation matrix.
    Args:
        num_input_parameters (int): number of input parameters from flattened array
        num_init_parameters (int): number of parameters to produce in the first FC layer
        affine_config (tuple of ints): how many output parameters from each FC layer. Is used to
            define number of FC layers.
        drop_rate (float, optional): drop rate of each fully connected layer. Default: 0
    """

    def __init__(self, num_input_parameters, num_init_parameters,
                 affine_config, drop_rate=0):
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
                num_output_parameters=affine_config[i + 1],
                drop_rate=drop_rate
            )
            self.params.add_module('fc_block%d' % (i), block)

        # Final FC layer. This layer produces the 12 output transformation parameters.
        self.params.add_module('fc_out', nn.Linear(affine_config[-1], 12))

        # Initializing last fully connected layer to the identity matrix
        self.params.fc_out.weight.data.zero_()
        self.params.fc_out.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                                                        dtype=torch.float32))

    def forward(self, x):
        params = self.params(x)
        return params
