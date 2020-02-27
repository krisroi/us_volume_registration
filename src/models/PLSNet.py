import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch.jit.annotations import List


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

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        return concated_features

    def forward(self, x):
        bottleneck_output = self.bn_function(x)
        print(bottleneck_output.shape)
        out = self.dwConv(bottleneck_output)
        out = self.relu(self.norm(self.conv(out)))
        return out


class _StridedDSConv(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_StridedDSConv, self).__init__()

        self.add_module('dwConv', nn.Conv3d(num_input_features, num_input_features, kernel_size=3,
                                            stride=1, padding=1, groups=1, bias=False))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1,
                                          stride=2, bias=False))
        self.add_module('norm', nn.BatchNorm3d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        return concated_features

    def forward(self, x):
        print('_StridedDSConv forward().')
        bottleneck_output = self.bn_function(x)
        out = self.dwConv(bottleneck_output)
        out = self.relu(self.norm(self.conv(out)))
        return out


class _DilatedResidualDenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate):
        super(_DilatedResidualDenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DSConv(
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                padding=(i + 1),
                dilation=i + 1
            )
            self.add_module('3x3x3 DS_Conv%d' % (i + 1), layer)

        self.add_module('conv', nn.Conv3d(num_input_features + num_layers * growth_rate, num_input_features,
                                          kernel_size=1, stride=1, bias=False))

    def forward(self, init_features):
        print('_DilatedResidualDenseBlock forward().')
        features = [init_features]
        for name, layer in self.items():
            print(layer)
            new_features = layer(features)
            features.append(new_features)
        concat = torch.cat(features, 1)
        print('Concat shape: ', concat.shape)
        out = self.conv(concat)
        print('Out prior to addition: ', out.shape)
        out = torch.add(out, concat)
        print('Out posterior to addition: ', out.shape)
        return out


class _InputReinforcement(nn.Module):
    def __init__(self, layerNum):
        super(_InputReinforcement, self).__init__()

        self.layerNum = layerNum
        self.sampling_rate = 2**layerNum

    def forward(self, x):
        downsampled_data = F.interpolate(x, scale_factor=(1 / self.sampling_rate), mode='trilinear')
        concated_data = concat(origInput, x)
        return downsampled_data


class _Concatenation(nn.Module):
    def __init__(self, convInput):
        super(_Concatenation, self).__init__()

        self.convInput = convInput

    def forward(self, origInput):
        return torch.cat((conv))


class _PLSNet(nn.Module):

    __constants__ = ['features']

    def __init__(self, PLS_CONFIG, growth_rate, num_init_features):
        super(_PLSNet, self).__init__()

        # Initial StridedDSConv
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False)),
            ('conv1', nn.Conv3d(1, num_init_features, kernel_size=1, stride=2, bias=False)),
            ('norm', nn.BatchNorm3d(num_init_features)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(PLS_CONFIG):

            DRS_BLOCK = _DilatedResidualDenseBlock(num_layers=num_layers,
                                                   num_input_features=num_features,
                                                   growth_rate=growth_rate)
            self.features.add_module('DRS_BLOCK%d' % (i + 1), DRS_BLOCK)

            DS_LAYER = _StridedDSConv(num_input_features=num_features,
                                      num_output_features=num_features * 2)
            self.features.add_module('DS_LAYER%d' % (i + 1), DS_LAYER)
            num_features = num_features * 2

    def forward(self, x):
        print('_PLS forward().')
        features = self.features(x)
        out = F.relu(features, inplace=True)
        return out


if __name__ == '__main__':

    # Supress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

    arr = torch.randn(4, 1, 64, 64, 64)

    #net1 = _DSConv(num_input_features=1, growth_rate=12, dilation=2)
    #net2 = _StridedDSConv(num_input_features=1, num_output_features=12)
    #net3 = _DilatedResidualDenseBlock(num_layers=4, num_input_features=8, growth_rate=12)
    #net4 = _InputReinforcement(layerNum=1)

    #DS = net2(array)
    # print(DS.shape)
    #IR = net4(array)
    # print(IR.shape)
    #out = torch.cat((DS, IR), 1)
    # print(out.shape)

    # print(net1)
    # print(net2)
    # print(net3)

    PLS_CONFIG = (4, 4, 4)

    net5 = _PLSNet(PLS_CONFIG=PLS_CONFIG, growth_rate=8, num_init_features=4)
    net5(arr)
