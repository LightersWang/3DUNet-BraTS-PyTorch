import numpy as np
import torch
import torch.nn as nn

from typing import Union
from models.blocks import PlainBlock, ResidualBlock, Upsample


class UNetEncoder(nn.Module):
    """
        U-Net Encoder (include bottleneck)

        input_channels: #channels of input images, e.g. 4 for BraTS multimodal input
        channels_list:  #channels of every levels, e.g. [8, 16, 32, 64, 80, 80]
        block:          Type of conv blocks, choice from PlainBlock and ResidualBlock
    """
    def __init__(self, input_channels, channels_list, 
                 block:Union[PlainBlock, ResidualBlock]=PlainBlock, **block_kwargs):
        super(UNetEncoder, self).__init__()

        self.input_channels = input_channels
        self.channels_list = channels_list    # last is bottleneck
        self.block_type = block

        self.levels = nn.ModuleList()
        for l, num_channels in enumerate(self.channels_list):
            in_channels  = self.input_channels if l == 0 else self.channels_list[l-1]
            out_channels = num_channels
            first_stride = 1 if l == 0 else 2   # level 0 don't downsample

            # 2 blocks per level
            blocks = nn.Sequential(
                block(in_channels,  out_channels, stride=first_stride, **block_kwargs),
                block(out_channels, out_channels, stride=1, **block_kwargs),
            )
            self.levels.append(blocks)

    def forward(self, x, return_skips=False):
        skips = []

        for s in self.levels:
            x = s(x)
            skips.append(x)

        return skips if return_skips else x

class UNetDecoder(nn.Module):
    """
        U-Net Decoder (include bottleneck)

        output_classes:   #classes of final ouput
        channels_list:    #channels of every levels in a bottom-up order, e.g. [320, 320, 256, 128, 64, 32]
        deep_supervision: Whether to use deep supervision
        ds_layer:         Last n layer for deep supervision, default set 0 for turning off
        block:            Type of conv blocks, better be consistent with encoder

        NOTE: Add sigmoid in the end WILL cause numerical unstability.
    """
    def __init__(self, output_classes, channels_list, deep_supervision=False, ds_layer=0,
                 block:Union[PlainBlock, ResidualBlock]=PlainBlock, **block_kwargs):
        super(UNetDecoder, self).__init__()

        self.output_classes = output_classes
        self.channels_list = channels_list                    # first is bottleneck
        self.deep_supervision = deep_supervision
        self.block_type = block
        num_upsample = len(self.channels_list) - 1

        # decoder
        self.levels = nn.ModuleList()
        self.trans_convs = nn.ModuleList()
        for l in range(num_upsample):         # exclude bottleneck
            in_channels  = self.channels_list[l]
            out_channels = self.channels_list[l+1]

            # transpose conv
            trans_conv = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=2, stride=2)
            self.trans_convs.append(trans_conv)

            # 2 blocks per level
            blocks = nn.Sequential(
                block(out_channels * 2, out_channels, stride=1, **block_kwargs),
                block(out_channels, out_channels, stride=1, **block_kwargs),
            )
            self.levels.append(blocks)
        
        # seg output 
        self.seg_output = nn.Conv3d(
            self.channels_list[-1], self.output_classes, kernel_size=1, stride=1)

        # mid-layer deep supervision
        if (self.deep_supervision) and (ds_layer > 1):
            self.ds_layer_list = list(range(num_upsample - ds_layer, num_upsample - 1))
            self.ds = nn.ModuleList()
            for l in range(num_upsample - 1):
                if l in self.ds_layer_list:
                    in_channels = self.channels_list[l+1]
                    up_factor = in_channels // self.channels_list[-1]
                    assert up_factor > 1        # otherwise downsample

                    ds = nn.Sequential(
                        nn.Conv3d(in_channels, self.output_classes, kernel_size=1, stride=1),
                        Upsample(scale_factor=up_factor, mode='trilinear', align_corners=False),
                    )
                else:
                    ds = None     # for easier indexing

                self.ds.append(ds)

    def forward(self, skips):
        skips = skips[::-1]     # reverse so that bottleneck is the first
        x = skips.pop(0)        # bottleneck

        ds_outputs = []
        for l, feat in enumerate(skips):
            x = self.trans_convs[l](x)          # upsample last-level feat
            x = torch.cat([feat, x], dim=1)     # concat upsampled feat and same-level skip feat
            x = self.levels[l](x)               # concated feat to conv

            if (self.training) and (self.deep_supervision) and (l in self.ds_layer_list):
                ds_outputs.append(self.ds[l](x))

        if self.training:
            return [self.seg_output(x)] + ds_outputs[::-1]  # reverse back
        else:
            return self.seg_output(x)


class UNet(nn.Module):
    """
        U-Net

        input_channels:   #channels of input images, e.g. 4 for BraTS multimodal input
        output_classes:   #classes of final ouput
        channels_list:    #channels of every levels in a top-down order, e.g. [32, 64, 128, 256, 320, 320]
        block:            Type of conv blocks, choice from PlainBlock and ResidualBlock
        deep_supervision: Whether to use deep supervision in decoder
        ds_layer:         Last n layer for deep supervision, default set 0 for turning off
    """
    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False, 
                 ds_layer=0, block:Union[PlainBlock, ResidualBlock]=PlainBlock, **block_kwargs):
        super(UNet, self).__init__()

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder(output_classes, channels_list[::-1], block=block, 
            deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

    def forward(self, x):
        return self.decoder(self.encoder(x, return_skips=True))


class MultiEncoderUNet(nn.Module):
    """
        Multi-encoder U-Net for Multimodal Input

        input_channels:   #channels of input images, also is the #encoders
        output_classes:   #classes of final ouput
        channels_list:    #channels of every levels of decoder in a top-down order, e.g. [32, 64, 128, 256, 320, 320]
        block:            Type of conv blocks, choice from PlainBlock and ResidualBlock
        deep_supervision: Whether to use deep supervision in decoder
        ds_layer:         Last n layer for deep supervision, default set 0 for turning off
    """
    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False, 
                 ds_layer=0, block:Union[PlainBlock, ResidualBlock]=PlainBlock, **block_kwargs):
        super(MultiEncoderUNet, self).__init__()

        self.num_skips = len(channels_list)
        self.num_encoders = input_channels
        if isinstance(channels_list, list):
            channels_list = np.array(channels_list)

        # encoders
        self.encoders = nn.ModuleList()
        for _ in range(self.num_encoders):
            self.encoders.append(
                UNetEncoder(1, (channels_list // self.num_encoders), block=block, **block_kwargs))

        # all encoders shared one decoder
        self.decoder = UNetDecoder(output_classes, channels_list[::-1], block=block, 
            deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

    def forward(self, x:torch.Tensor):
        # seperate skips for every encoder
        encoders_skips = []
        for xx, encoder in zip(x.chunk(self.num_encoders, dim=1), self.encoders):
            encoders_skips.append(encoder(xx, return_skips=True))

        # concat same-level skip of different encoders
        encoders_skips = [
            torch.cat([encoders_skips[i][j] for i in range(self.num_encoders)], dim=1) 
        for j in range(self.num_skips)]

        return self.decoder(encoders_skips)


# unit tests
if __name__ == "__main__":
    block_kwargs = {
        "kernel_size": 3, 
        "conv_bias": True, 
        "dropout_prob": None, 
        "norm_key": 'instance'
    }

    # input 
    x = torch.rand(1, 4, 128, 128, 128)
    channels = np.array([32, 64, 128, 256, 320, 320])

    # unet
    unet = UNet(4, 3, channels, deep_supervision=True, ds_layer=4, **block_kwargs)
    # print(unet)
    unet.eval()
    segs = unet(x)
    print(segs.shape)

    # multi-encoder unet
    mencoder_unet = MultiEncoderUNet(4, 3, channels, deep_supervision=True, ds_layer=4, **block_kwargs)
    # print(mencoder_unet)
    segs = mencoder_unet(x)
    for s in segs:
        print(s.shape)
