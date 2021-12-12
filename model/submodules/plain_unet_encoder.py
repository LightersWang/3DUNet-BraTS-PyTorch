import sys
sys.path.append("/home/whr/Code/multi_modal_al/seg/")

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.submodules.blocks import StackedConvLayers


class PlainUNetEncoder(nn.Module):
    """
        Plain UNet Encoder (include bottleneck) 
        base_num_features = 8 (Single Modality)
        base_num_features = 32 (Multi Modality)
    """
    def __init__(self, input_channels, num_downsample, num_blocks_per_stage, layer_args, base_num_features=8, 
                 featmap_mul_downsample=2):
        super(PlainUNetEncoder, self).__init__()

        self.num_blocks_per_stage = num_blocks_per_stage
        self.base_num_features = base_num_features
        self.featmap_mul_downsample = featmap_mul_downsample
        self.num_stages = num_downsample + 1    # include bottleneck
        self.max_num_features = self.base_num_features * 10     # 8 --> 80

        self.output_fearures = [min(self.max_num_features, self.base_num_features * (self.featmap_mul_downsample ** i)) 
            for i in range(self.num_stages)]
        self.input_fearures = [input_channels, *self.output_fearures[:-1]]
        self.first_strides = [1] + [self.featmap_mul_downsample] * (self.num_stages - 1)

        self.stages = []
        for stage in range(self.num_stages):
            current_stage = StackedConvLayers(
                num_convs=self.num_blocks_per_stage,
                input_channels=self.input_fearures[stage],
                output_channels=self.output_fearures[stage],
                first_stride=self.first_strides[stage],
                **layer_args
            )
            self.stages.append(current_stage)
        
        # -1 is bottleneck
        self.stages = nn.ModuleList(self.stages)


    def forward(self, x):
        skips = []

        for s in self.stages:
            x = s(x)
            skips.append(x)

        return skips


if __name__ == "__main__":
    # single modal encoder test
    layer_args = {
        "kernel_size": 3, 
        "conv_bias": True, 
        "dropout_prob": None, 
        "norm": 'instance'
    }
    # single_modal_encoder = PlainUNetEncoder(
    #     input_channels=1, num_downsample=5, num_blocks_per_stage=2, 
    #     layer_args=layer_args, base_num_features=8, featmap_mul_downsample=2)
    # print(single_modal_encoder.stages)
    # print(single_modal_encoder.input_fearures)
    # print(single_modal_encoder.output_fearures)
    # print(single_modal_encoder.first_strides)

    # print("========================================================")
    # x = torch.rand(1, 1, 128, 128, 128)
    # skips = single_modal_encoder(x)
    # for feat in skips:
    #     print(feat.shape)

    # multi modal encoder test
    multi_modal_encoder = PlainUNetEncoder(
        input_channels=4, num_downsample=5, num_blocks_per_stage=2, 
        layer_args=layer_args, base_num_features=32, featmap_mul_downsample=2)
    print(multi_modal_encoder.stages)
    print(multi_modal_encoder.input_fearures)
    print(multi_modal_encoder.output_fearures)
    print(multi_modal_encoder.first_strides)

    print("========================================================")
    x = torch.rand(1, 4, 128, 128, 128)
    skips = multi_modal_encoder(x)
    for feat in skips:
        print(feat.shape)


