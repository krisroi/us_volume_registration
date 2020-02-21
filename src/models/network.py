import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from collections import OrderedDict
from torch.jit.annotations import List

from utils.utility_functions import plot_featuremaps

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def any_requires_grad(self, input):
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, input):
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    @torch.jit._overload_method
    def forward(self, input):
        pass

    @torch.jit._overload_method
    def forward(self, input):
        pass

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class _DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    __constants__ = ['features']

    def __init__(self, growth_rate=24, block_config=(1, 2, 4, 8, 16),
                 num_init_features=8, bn_size=4, drop_rate=0, memory_efficient=True):

        super(_DenseNet, self).__init__()

        # Initial convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=5, stride=1,
                                padding=1, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
        ]))

        # Each denseblock
        # block_config = (1,2,3,4) yields 4 blocks, with 1, 2, 3 and 4 layers respectively
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config):
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        print(out.shape)
        #out = F.adaptive_avg_pool3d(out, (2, 2, 2))
        #plot_featuremaps(out.cpu().detach().numpy())
        return out


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

        # Initial fully connected layer. Takes num_input_features from _DenseNet and outputs
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


class USAIRNet(nn.Module):
    def __init__(self, denseNet, affineRegression):
        super(USAIRNet, self).__init__()

        self.denseNet = denseNet
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
        fixed_loc = torch.flatten(self.denseNet(fixed), 1)
        moving_loc = torch.flatten(self.denseNet(moving), 1)
        concated = torch.cat((fixed_loc, moving_loc), 1)

        theta = self.affineRegression(concated)
        theta = theta.view(-1, 3, 4)
        return theta


# Utility for counting parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    
    denseNet = _DenseNet(growth_rate=8, block_config=(1, 2, 4, 8, 16, 32),
                         num_init_features=12, bn_size=4, drop_rate=0,
                         memory_efficient=False
                        )

    affineRegression = _AffineRegression(
        num_input_parameters=2032,
        num_init_parameters=512,
        affine_config=(512, 256, 128, 64),
        reduction_rate=2,
        drop_rate=0.2
       ) 
    
    net = USAIRNet(denseNet, affineRegression).to('cuda:1')
    
    fix = torch.randn(1, 1, 320, 320, 320).to('cuda:1')
    #mov = torch.randn(1, 1, 300, 300, 300)
    
    net(fix, fix)