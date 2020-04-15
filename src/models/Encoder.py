import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from torch.jit.annotations import List


class _Conv(nn.Module):
    def __init__(self, num_input_features, growth_rate, padding, dilation):
        super(_Conv, self).__init__()

        self.add_module('conv', nn.Conv3d(num_input_features, growth_rate, kernel_size=3,
                                          stride=1, padding=padding, dilation=dilation,
                                          groups=1, bias=False))
        self.add_module('norm', nn.BatchNorm3d(growth_rate))
        self.add_module('relu', nn.ReLU(inplace=True))

    
    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (Tensor) -> (Tensor)
        pass
        
        
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x
        out = torch.cat(prev_features, 1)
        out = self.relu(self.norm(self.conv(out)))
        return out


class _StridedConv(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_StridedConv, self).__init__()

        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=3,
                                          stride=2, padding=1, groups=1, bias=False))
        self.add_module('norm', nn.BatchNorm3d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.relu(self.norm(self.conv(x)))
        return out


class _DilatedResidualDenseBlock(nn.ModuleDict):

    def __init__(self, num_layers, num_input_features, growth_rate):
        super(_DilatedResidualDenseBlock, self).__init__()

        self.intermediate = nn.ModuleDict()

        for i in range(num_layers):
            layer = _Conv(
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                padding=(i + 1),
                dilation=(i + 1)
            )
            self.intermediate.add_module('ConvLayer%d' % (i + 1), layer)

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


class _Encoder(nn.Module):
    r"""Encoder model class, largely based on the encoder presented in
    "Efficient 3D Fully Convolutional Networks for Pulmonary Lobe Segmentation in CT Images"
        <https://arxiv.org/pdf/1909.07474v1.pdf>
    Args:
        growth_rate (int): how many filters to add each layer (`k` in paper)
        encoder_config (tuple of 4 ints): how many layers in each DRDB block
        num_init_features (int): the number of filters to learn in the first convolution layer
    """
    
    #__constants__ = ['drd_module']

    def __init__(self, encoder_config, growth_rate, num_init_features, memory_efficient=False):
        super(_Encoder, self).__init__()

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
            self.drd_module.add_module('DRD_BLOCK%d' % ((i + 1)), DRD_BLOCK)

            if i != len(encoder_config) - 1:
                STRIDED_CONV_LAYER = _StridedConv(num_input_features=num_features,
                                                  num_output_features=num_init_features * (2**(i + 1)))
                self.strided_conv_module.add_module('DS_LAYER%d' % (i + 1), STRIDED_CONV_LAYER)
            num_features = num_init_features * (2**(i + 1)) + 1
            
        self.__get_keys()
        
        self.memory_efficient = memory_efficient
        
            
    def __get_keys(self):
        self.drd_keynames = []
        self.strided_keynames = []
            
        for key in self.drd_module.keys():
            self.drd_keynames.append(key)
            
        for key in self.strided_conv_module.keys():
            self.strided_keynames.append(key)
        self.strided_keynames.append('') # Append empty entry to make it correct length
        
        
    @torch.jit.unused
    def call_checkpoint(self, module):
        """Custom checkpointing function. 
           If memory_efficient, trade compute for memory.
        """
        def closure(*inputs):
            inputs = module(inputs[0])
            return inputs
        return closure
            

    def forward(self, x):
        
        # Save original input
        origInput = x
        
        # Initial strided conv layer
        out = self.relu(self.norm(self.conv1(self.conv0(x))))
        
        # Counter for finding correct DRD-block
        drd_key_num = 0
                
        for i, key in enumerate(self.strided_keynames):
            
            # Downsampling original data
            downsampled_data = F.interpolate(input=origInput,
                                             scale_factor=(1 / (2 ** (i + 1))),
                                             mode='trilinear')
            # Input reinforcemet
            out = torch.cat((out, downsampled_data), 1)
            
            # Apply DRD-block
            if self.memory_efficient: # and not torch.jit.is_scripting():
                out = cp.checkpoint(self.drd_module[self.drd_keynames[drd_key_num]], out)
            else:
                out = self.drd_module[self.drd_keynames[drd_key_num]](out)
            
            # Apply strided conv-layer
            if i != len(self.encoder_config) - 1:
                out = self.strided_conv_module[key](out)
            
            drd_key_num += 1
            
        out = self.relu(out)
        return out