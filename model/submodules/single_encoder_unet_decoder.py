import torch
import torch.nn as nn

from model.submodules.blocks import StackedConvLayers, Upsample
from model.submodules.plain_unet_encoder import PlainUNetEncoder


class SingleEncoderUNetDecoder(nn.Module):
    """
    Add sigmoid in the end WILL cause numerical unstability.
    """

    def __init__(self, single_modal_encoder:PlainUNetEncoder,
                 num_classes, layer_args, deep_supervision=True):
        super(SingleEncoderUNetDecoder, self).__init__()

        # assume all encoders have same architecture, so one encoder is enough
        single_modal_output_features = single_modal_encoder.output_fearures
        single_modal_num_stages = single_modal_encoder.num_stages

        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        self.num_stages = single_modal_num_stages - 1                                            # exclude bottleneck
        self.num_blocks_per_stage = single_modal_encoder.num_blocks_per_stage                    # same with encoder
        self.output_features = [f for f in single_modal_output_features[:-1][::-1]]              # concat single modal feature
        self.conv_input_features = [f * 2 for f in self.output_features]                         # concat skip features & upsample feature
        self.transconv_input_features = [f for f in single_modal_output_features[1:][::-1]]      # dim reduction by transpose conv

        self.transpose_convs = []
        self.stages = []
        for stage in range(self.num_stages):
            # upsample & dim reduction by transpose conv
            transconv = nn.ConvTranspose3d(
                in_channels=self.transconv_input_features[stage],
                out_channels=self.output_features[stage],
                kernel_size=2, stride=2, bias=False
            )
            self.transpose_convs.append(transconv)

            # conv
            current_stage = StackedConvLayers(
                num_convs=self.num_blocks_per_stage,
                input_channels=self.conv_input_features[stage],
                output_channels=self.output_features[stage],
                first_stride=1, **layer_args
            )
            self.stages.append(current_stage)

        # features to logits
        self.segmentation_output = nn.Conv3d(
                in_channels=self.output_features[-1],
                out_channels=self.num_classes,
                kernel_size=1, stride=1, bias=False)
        
        # deep supervision for largest 4 stages, upsample logits
        if self.deep_supervision and self.num_stages >= 4:
            self.ds = []
            for stage in range(self.num_stages-4, self.num_stages-1):
                # conv1x1x1 reduce dim to num_class
                conv1x1x1 = nn.Conv3d(
                    in_channels=self.output_features[stage],
                    out_channels=self.num_classes,
                    kernel_size=1, stride=1, bias=False
                )

                # upsample to 128x128x128
                upsample_factor = 2 ** (self.num_stages - 1 - stage)
                upsample = Upsample(scale_factor=upsample_factor, mode='trilinear', align_corners=False)

                self.ds.append(nn.Sequential(conv1x1x1, upsample))

            self.ds = nn.ModuleList(self.ds)
        
        self.transpose_convs = nn.ModuleList(self.transpose_convs)
        self.stages = nn.ModuleList(self.stages)


    def forward(self, single_encoder_skips):
        # reverse skip feature
        single_encoder_skips = single_encoder_skips[::-1]

        # single_encoder_skips is reversed, so bottleneck is the first
        x = single_encoder_skips.pop(0)                 # bottleneck

        ds_outputs = []
        for stage in range(self.num_stages):
            # U Net forward
            x = self.transpose_convs[stage](x)
            x = torch.cat([single_encoder_skips[stage], x], dim=1)        # save GPU memory
            x = self.stages[stage](x)

            # deep supervision
            if self.deep_supervision and (self.num_stages - 4 <= stage and stage < self.num_stages - 1):
                ds_outputs.append(self.ds[stage - (self.num_stages - 4)](x))

        seg_output = self.segmentation_output(x)

        # seg level: up --> down
        # e.g. [128^3, 64^3, 32^3, 16^3] if upscale factor is 1
        if not self.deep_supervision:
            return [seg_output]
        else:
            return [seg_output] + ds_outputs[::-1]



if __name__ == "__main__":
    layer_args = {
        "kernel_size": 3, 
        "conv_bias": True, 
        "dropout_prob": None, 
        "norm": 'instance'
    }

    # single decoder forward test
    encoder = PlainUNetEncoder(4, 5, 2, layer_args, base_num_features=32, featmap_mul_downsample=2)
    decoder = SingleEncoderUNetDecoder(encoder, 3, layer_args, deep_supervision=True)
    
    x = torch.rand(1, 4, 128, 128, 128)
    # single_encoder_skips = encoder(x)

    # print("========================================================")
    # print("encoder output: ")
    # for feat in single_encoder_skips:
    #     print(feat.shape)

    outputs = decoder(encoder(x))
    # print(decoder)

    print("========================================================")
    print("seg & deep supervision output: ")
    for op in outputs:
        print(op.shape)

    print(outputs[0].max().item(), outputs[0].mean().item(), outputs[0].min().item())
