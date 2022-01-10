# python 3.7
"""Contains the implementation of discriminator described in PGGAN.

Different from the official tensorflow version in folder `pggan_tf_official`,
this is a simple pytorch version which only contains the discriminator part. This
class is specially used for recomposition and feature extraction.

For more details, please check the original paper:
https://arxiv.org/pdf/1710.10196.pdf
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PGGANDiscriminatorNet']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4


class PGGANDiscriminatorNet(nn.Module):
    """Defines the generator network in PGGAN.

      NOTE: The generated images are with `RGB` color channels and range [-1, 1].
    """

    def __init__(self,num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
                 resolution          = 1024,           # Input resolution. Overridden based on dataset.
                 last_res            = 4,
                 fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
                 fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
                 fmap_max            = 512,          # Maximum number of feature maps in any layer.
                 use_wscale          = True,         # Enable equalized learning rate?
                 dtype               = 'float32',    # Data type to use for activations and outputs.
                 fused_scale         = True,  # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
                 group_size          = 4,
                 **kwargs):
        super().__init__()
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.input_shapes = [None, num_channels, resolution, resolution]
        self.init_res = resolution
        self.log2_init_res = np.int(np.log2(self.init_res))
        self.last_res = last_res
        self.log2_last_res = np.int(np.log2(self.last_res))
        self.num_channels = num_channels
        self.use_wscale = use_wscale
        self.dtype = dtype
        self.fused_scale = fused_scale

        in_channels = nf(self.log2_init_res-1)
        self.lod = nn.Parameter(torch.zeros(()))
        self.pth_to_tf_var_mapping = {'lod': 'lod'}
        self.add_module('down', DownScale(2))
        self.add_module('groupbn', GroupMinibatchNorm(group_size=group_size))

        self.num_layers = (self.log2_init_res - self.log2_last_res + 1) * 2

        for log2_res in range(self.log2_init_res, self.log2_last_res - 1, -1):
            layer_idx = (self.log2_init_res - log2_res) * 2
            res = 2 ** log2_res
            self.add_module('fromRGB%d' % (layer_idx // 2),
                            ConvBlock(down_sample=False, in_channels=3,
                                      out_channels=nf(log2_res - 1),
                                      kernel_size=1,
                                      padding=0))
            self.pth_to_tf_var_mapping[f'fromRGB{layer_idx // 2}.conv.weight'] = f'FromRGB_lod{layer_idx // 2}/weight'
            self.pth_to_tf_var_mapping[f'fromRGB{layer_idx // 2}.wscale.bias'] = f'FromRGB_lod{layer_idx // 2}/bias'
            # larger than 4.
            if log2_res != self.log2_last_res:
                self.add_module('layer%d' % (layer_idx), ConvBlock(down_sample=False, in_channels=in_channels,
                                                                   out_channels=nf(log2_res - 1),
                                                                   kernel_size=3))
                self.pth_to_tf_var_mapping[f'layer{layer_idx}.conv.weight'] = f'{res}x{res}/Conv0/weight'
                self.pth_to_tf_var_mapping[f'layer{layer_idx}.wscale.bias'] = f'{res}x{res}/Conv0/bias'

                in_channels = nf(log2_res - 1)
                if self.fused_scale:
                    pass
                    '''
                    self.add_module('layer%d' % (layer_idx + 1), ConvBlock(down_sample=True, in_channels=in_channels,
                                                                           out_channels=nf(log2_res - 2),
                                                                           kernel_size=3))
                    in_channels = nf(log2_res - 2)
                    self.pth_to_tf_var_mapping[f'layer{layer_idx + 1}.weight'] = f'{res}x{res}/Conv1_down/weight'
                    self.pth_to_tf_var_mapping[f'layer{layer_idx + 1}.wscale.bias'] = f'{res}x{res}/Conv1_down/bias'
                    '''
                else:
                    self.add_module('layer%d' % (layer_idx + 1), ConvBlock(down_sample=False, in_channels=in_channels,
                                                                           out_channels=nf(log2_res - 2),
                                                                           kernel_size=3))
                    in_channels = nf(log2_res - 2)
                    self.pth_to_tf_var_mapping[f'layer{layer_idx + 1}.conv.weight'] = f'{res}x{res}/Conv1/weight'
                    self.pth_to_tf_var_mapping[f'layer{layer_idx + 1}.wscale.bias'] = f'{res}x{res}/Conv1/bias'
            # equal to 4.
            else:
                self.add_module('layer%d' % layer_idx, ConvBlock(down_sample=False, in_channels=in_channels + 1,
                                                                   out_channels=nf(log2_res - 1),
                                                                   kernel_size=3))
                in_channels = nf(log2_res - 1)
                self.pth_to_tf_var_mapping[f'layer{layer_idx}.conv.weight'] = f'{res}x{res}/Conv/weight'
                self.pth_to_tf_var_mapping[f'layer{layer_idx}.wscale.bias'] = f'{res}x{res}/Conv/bias'

                self.add_module('layer%d' % (layer_idx + 1),
                                DenseBlock(in_channels=in_channels * 16,
                                           out_channels=nf(log2_res - 1)))
                self.pth_to_tf_var_mapping[f'layer{layer_idx + 1}.dense.weight'] = f'{res}x{res}/Dense0/weight'

                self.add_module('layer%d' % (layer_idx + 2),
                                DenseBlock(in_channels=nf(log2_res - 1),
                                           out_channels=1, activation_type='linear', wscale_gain=1.0))
                self.pth_to_tf_var_mapping[f'layer{layer_idx + 2}.dense.weight'] = f'{res}x{res}/Dense1/weight'

    def forward(self, image_in, which_block=5, pre_model=False, post_model=False, attain_list=[]):
        def lerp_clip(a, b, t):
            return a + (b - a) * torch.clamp(t, 0.0, 1.0)

        if pre_model:
            image = image_in
            if not (len(image.shape) == 4 and image.shape[1] == self.num_channels and image.shape[2] == self.init_res):
                raise ValueError(f'The input tensor should be with shape [batch_size, '
                                 f'ch, W, H], where `ch` equals to '
                                 f'{self.num_channels}!\n'
                                 f'But {image.shape[1]} received!')

            count = 0
            image = image.view(image.shape[0], self.num_channels, self.init_res, self.init_res)
            lod_in = self.__getattr__(f'lod')

            attain_results = []
            x = self.__getattr__(f'fromRGB{0}')(image)
            count += 1
            if count in attain_list:
                attain_results.append(x)

            if count >= which_block:
                if len(attain_results) == 0:
                    return x
                else:
                    return x, attain_results

            for res_log2 in range(self.log2_init_res, self.log2_last_res, -1):
                block_idx = self.log2_init_res - res_log2
                lod = self.log2_init_res - res_log2

                x = self.__getattr__(f'layer{2 * block_idx}')(x)

                count += 1
                if count in attain_list:
                    attain_results.append(x)

                if count >= which_block:
                    if len(attain_results) == 0:
                        return x
                    else:
                        return x, attain_results

                x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
                if not self.fused_scale:
                    x = self.__getattr__(f'down')(x)

                count += 1
                if count in attain_list:
                    attain_results.append(x)

                if count >= which_block:
                    if len(attain_results) == 0:
                        return x
                    else:
                        return x, attain_results

                image = self.__getattr__(f'down')(image)
                y = self.__getattr__(f'fromRGB{block_idx + 1}')(image)
                x = lerp_clip(x, y, lod_in - lod)

            block_idx = self.log2_init_res - (res_log2 - 1)
            x = self.__getattr__(f'groupbn')(x)
            x = self.__getattr__(f'layer{2 * block_idx}')(x)
            count += 1
            if count in attain_list:
                attain_results.append(x)

            if count >= which_block:
                if len(attain_results) == 0:
                    return x
                else:
                    return x, attain_results
            x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
            x = self.__getattr__(f'layer{2 * block_idx + 2}')(x)

            count += 1
            if count in attain_list:
                attain_results.append(x)

            if count >= which_block:
                if len(attain_results) == 0:
                    return x
                else:
                    return x, attain_results

            return x

        elif post_model:
            image = image_in

            count = 0
            if count < which_block:
                x = image
            else:
                image = image.view(image.shape[0], self.num_channels, self.init_res, self.init_res)
                lod_in = self.__getattr__(f'lod')
                x = self.__getattr__(f'fromRGB{0}')(image)

            count += 1

            for res_log2 in range(self.log2_init_res, self.log2_last_res, -1):
                block_idx = self.log2_init_res - res_log2
                lod = self.log2_init_res - res_log2

                if count >= which_block:
                    x = self.__getattr__(f'layer{2 * block_idx}')(x)
                count += 1

                if count >= which_block:
                    x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
                    if not self.fused_scale:
                        x = self.__getattr__(f'down')(x)
                count += 1

                image = self.__getattr__(f'down')(image)
                y = self.__getattr__(f'fromRGB{block_idx + 1}')(image)
                x = lerp_clip(x, y, lod_in - lod)

            block_idx = self.log2_init_res - (res_log2 - 1)

            if count >= which_block:
                x = self.__getattr__(f'groupbn')(x)
                x = self.__getattr__(f'layer{2 * block_idx}')(x)
            count += 1

            if count >= which_block:
                x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
                x = self.__getattr__(f'layer{2 * block_idx + 2}')(x)
            count += 1
            return x

        else:
            image = image_in
            if not (len(image.shape) == 4 and image.shape[1] == self.num_channels and image.shape[2] == self.init_res):
                raise ValueError(f'The input tensor should be with shape [batch_size, '
                                 f'ch, W, H], where `ch` equals to '
                                 f'{self.num_channels}!\n'
                                 f'But {image.shape[1]} received!')

            image = image.view(image.shape[0], self.num_channels, self.init_res, self.init_res)
            lod_in = self.__getattr__(f'lod')

            x = self.__getattr__(f'fromRGB{0}')(image)
            for res_log2 in range(self.log2_init_res, self.log2_last_res, -1):
                block_idx = self.log2_init_res - res_log2
                lod = self.log2_init_res - res_log2

                x = self.__getattr__(f'layer{2 * block_idx}')(x)
                x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
                if not self.fused_scale:
                    x = self.__getattr__(f'down')(x)
                image = self.__getattr__(f'down')(image)
                y = self.__getattr__(f'fromRGB{block_idx + 1}')(image)
                x = lerp_clip(x, y, lod_in - lod)

            block_idx = self.log2_init_res - (res_log2 - 1)
            x = self.__getattr__(f'groupbn')(x)
            x = self.__getattr__(f'layer{2 * block_idx}')(x)
            x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
            x = self.__getattr__(f'layer{2 * block_idx + 2}')(x)

            return x


class GroupMinibatchNorm(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        group_size = min(self.group_size, x.shape[0])
        s = x.shape
        y = torch.reshape(x, [group_size, -1, s[1], s[2], s[3]])
        y = y - torch.mean(y, dim=0, keepdim=True)
        y = torch.mean(torch.pow(y, 2.0), dim=0)
        y = torch.sqrt(y + 1e-8)
        y = torch.mean(y, dim=[1, 2, 3], keepdim=True)
        y = torch.repeat_interleave(y, group_size, dim=0)
        y = torch.repeat_interleave(y, s[2], dim=2)
        y = torch.repeat_interleave(y, s[3], dim=3)
        return torch.cat([x, y], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, wscale_gain=np.sqrt(2.0), in_channels=512, out_channels=512, bias=False,
                 activation_type='lrelu'):
        super().__init__()
        self.scale = wscale_gain / np.sqrt(in_channels)
        self.dense = torch.nn.Linear(in_features=in_channels, out_features=out_channels, bias=bias)
        if activation_type == 'linear':
            self.activate = nn.Identity()
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation_type == 'tanh':
            self.activate = nn.Hardtanh()
        else:
            raise NotImplementedError(f'Not implemented activation function: '
                                      f'{activation_type}!')

    def forward(self, x):
        x = torch.reshape(x, [x.shape[0], -1])
        return self.activate(self.scale * self.dense(x))


class DownScale(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.down = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.scale == 1:
            return x
        return self.down(x)


class WScaleLayer(nn.Module):
  """Implements the layer to scale weight variable and add bias.

  NOTE: The weight variable is trained in `nn.Conv2d` layer, and only scaled
  with a constant number, which is not trainable in this layer. However, the
  bias variable is trainable in this layer.
  """

  def __init__(self, in_channels, out_channels, kernel_size, gain=np.sqrt(2.0)):
    super().__init__()
    fan_in = in_channels * kernel_size * kernel_size
    self.scale = gain / np.sqrt(fan_in)
    self.bias = nn.Parameter(torch.zeros(out_channels))

  def forward(self, x):
    return x * self.scale + self.bias.view(1, -1, 1, 1)


class ConvBlock(nn.Module):
    def __init__(self, down_sample=False, in_channels=512, out_channels=512, kernel_size=3,
                 wscale_gain=np.sqrt(2.0), stride=1, padding=1, dilation=1, add_bias=False,
                 activation_type='lrelu'):
        super().__init__()

        self.down_sample = down_sample
        if down_sample:
            self.weight = nn.Parameter(torch.randn(kernel_size, kernel_size, in_channels, out_channels))

        else:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=1,
                                  bias=add_bias)

        self.wscale = WScaleLayer(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  gain=wscale_gain)

        if activation_type == 'linear':
            self.activate = nn.Identity()
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation_type == 'tanh':
            self.activate = nn.Hardtanh()
        else:
            raise NotImplementedError(f'Not implemented activation function: '
                                      f'{activation_type}!')

    def forward(self, x):
        if self.down_sample:
            kernel = self.weight
            kernel = F.pad(kernel, (0, 0, 0, 0, 1, 1, 1, 1), 'constant', 0.0)
            kernel = (kernel[1:, 1:] + kernel[:-1, 1:] + kernel[1:, :-1] + kernel[:-1, :-1]) * 0.25
            kernel = kernel.permute(2, 3, 0, 1)
            x = F.conv2d(x, kernel, stride=2)
        else:
            x = self.conv(x)

        x = self.wscale(x)
        x = self.activate(x)
        return x




























