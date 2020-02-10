import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

from torch.jit.annotations import List


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
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    __constants__ = ['features']

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

        super(_DenseNet, self).__init__()

        # Initial convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
        ]))

        # Each denseblock
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
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = _DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model

def _airnet(arch, growth_rate, block_config, num_init_features, pretrained, progress, **kwargs):
    


def densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 12, (6, 10, 8), 1, pretrained, progress,
                     **kwargs)


class _Regressor(nn.Sequential):
    def __init__(self, num_input_features, num_theta_parameters, reduction):
        super(_Regressor, self).__init__()
        self.add_module('fc0', nn.Linear(num_input_features, num_input_features // reduction))
        self.add_module('norm0', nn.BatchNorm1d(num_input_features // reduction))
        self.add_module('relu0', nn.ReLU(inplace=True))
        self.add_module('fc1', nn.Linear(num_input_features // reduction, num_input_features // (2 * reduction)))
        self.add_module('norm1', nn.BatchNorm1d(num_input_features // (2 * reduction)))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('fc2', nn.Linear(num_input_features // (2 * reduction), num_input_features // (8 * reduction)))
        self.add_module('norm2', nn.BatchNorm1d(num_input_features // (8 * reduction)))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('fc3', nn.Linear(num_input_features // (8 * reduction), num_theta_parameters))

        #self.fc3.weight.data.zero_()
        #self.fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float64))

'''
class USAIRNet(nn.Module):

    __constants__ = ['fixed_loc, moving_loc']

    def __init__(self, growth_rate=4, block_config=(2, 6, 12, 8),
                 num_init_features=1, bn_size=2, drop_rate=0, memory_efficient=False,
                 num_input_features=43218, num_theta_parameters=12, reduction=8):

        super(USAIRNet, self).__init__()
        
        self.fixed_loc = nn.Sequential()
        self.moving_loc = nn.Sequential()
        self.regressor = nn.Sequential()
        
        dense = _DenseNet(growth_rate=growth_rate,
                          block_config=block_config,
                          num_init_features=num_init_features,
                          bn_size=bn_size,
                          drop_rate=drop_rate,
                          memory_efficient=memory_efficient
                          )
        
        self.fixed_loc.add_module('densenet%d', dense)
        self.moving_loc.add_module('densenet%d', dense)

        self.fixed_loc.add_module('flatten', nn.Flatten())
        self.moving_loc.add_module('flatten', nn.Flatten())

        regress = _Regressor(num_input_features=num_input_features,
                             num_theta_parameters=num_theta_parameters,
                             reduction=reduction)
        
        self.regressor.add_module('regressor%d', regress)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fixed, moving):
        fixed = self.fixed_loc(fixed)
        moving = self.moving_loc(moving)
        concated_params = torch.cat((fixed, moving), 1)
        theta = self.regressor(concated_params)
        return theta
'''


if __name__ == '__main__':

    net = densenet121()
    
    randn = torch.randn(1, 1, 224, 224, 224)
    
    print(net)
    print(net(randn).shape)
