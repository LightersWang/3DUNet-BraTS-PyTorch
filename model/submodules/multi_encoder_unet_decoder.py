import torch
import torch.nn as nn

from model.submodules.blocks import StackedConvLayers, Upsample
from model.submodules.plain_unet_encoder import PlainUNetEncoder


class MultiEncoderUNetDecoder(nn.Module):
    """
    Add sigmoid in the end WILL cause numerical unstability.
    """

    def __init__(self, num_encoder, single_modal_encoder:PlainUNetEncoder,
                 num_classes, layer_args, deep_supervision=True):
        super(MultiEncoderUNetDecoder, self).__init__()
        
        # assume all encoders have same architecture, so one encoder is enough
        single_modal_output_features = single_modal_encoder.output_fearures
        single_modal_num_stages = single_modal_encoder.num_stages

        self.num_classes = num_classes
        self.num_encoder = num_encoder
        self.deep_supervision = deep_supervision

        self.num_stages = single_modal_num_stages - 1                                                               # exclude bottleneck
        self.num_blocks_per_stage = single_modal_encoder.num_blocks_per_stage                                       # same with encoder
        self.output_features = [f * self.num_encoder for f in single_modal_output_features[:-1][::-1]]              # concat single modal feature
        self.conv_input_features = [f * 2 for f in self.output_features]                                            # concat skip features & upsample feature
        self.transconv_input_features = [f * self.num_encoder for f in single_modal_output_features[1:][::-1]]      # dim reduction by transpose conv

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


    def forward(self, multi_encoder_skips):
        # reverse skip feature
        multi_encoder_skips = [skip[::-1] for skip in multi_encoder_skips]

        # multi_encoder_skips is reversed, so bottleneck is the first
        x = torch.cat([skip.pop(0) for skip in multi_encoder_skips], dim=1)           # bottleneck

        ds_outputs = []
        for stage in range(self.num_stages):
            # U Net forward
            x = self.transpose_convs[stage](x)
            skip_feat = torch.cat([skip[stage] for skip in multi_encoder_skips], dim=1)       # skip connections
            x = torch.cat([skip_feat, x], dim=1)                                            # save GPU memory
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
    # # transpose conv test
    # x = torch.rand(1, 320, 4, 4, 4)
    # transconv = nn.ConvTranspose3d(320, 256, kernel_size=2, stride=2, bias=False)
    # print(transconv(x).shape)

    # # 1x1x1 conv test
    # conv1x1x1 = nn.Conv3d(320, 32, kernel_size=1, stride=1)
    # print(conv1x1x1(x).shape)

    # # multi modal concat test
    # multi_modal_x = [torch.rand(1, 80, 4, 4, 4) for _ in range(4)]
    # print(torch.cat(multi_modal_x, dim=1).shape)

    # decoder architecure test
    layer_args = {
        "kernel_size": 3, 
        "conv_bias": True, 
        "dropout_prob": None, 
        "norm": 'instance'
    }
    # encoder = SingleModalityUNetEncoder(1, 5, 2, layer_args, base_num_features=8, featmap_mul_downsample=2)
    # decoder = MultiModalityUNetDecoder(4, encoder, 4, layer_args, deep_supervision=True)
    # print(decoder)

    # multi decoder forward test
    encoder_t1    = PlainUNetEncoder(1, 5, 2, layer_args, base_num_features=8, featmap_mul_downsample=2)
    encoder_t1ce  = PlainUNetEncoder(1, 5, 2, layer_args, base_num_features=8, featmap_mul_downsample=2)
    encoder_t2    = PlainUNetEncoder(1, 5, 2, layer_args, base_num_features=8, featmap_mul_downsample=2)
    encoder_flair = PlainUNetEncoder(1, 5, 2, layer_args, base_num_features=8, featmap_mul_downsample=2)
    decoder       = MultiEncoderUNetDecoder(4, encoder_t1, 3, layer_args, deep_supervision=True)

    x = torch.rand(1, 1, 128, 128, 128)
    multi_encoder_skips = []
    for encoder in [encoder_t1, encoder_t1ce, encoder_t2, encoder_flair]:
        skips = encoder(x)
        multi_encoder_skips.append(skips)

        # print("========================================================")
        # for feat in skips:
        #     print(feat.shape)

    outputs = decoder(multi_encoder_skips)
    print(decoder)

    print("========================================================")
    print("seg & deep supervision output: ")
    for op in outputs:
        print(op.shape)

    print(outputs[0].max().item(), outputs[0].mean().item(), outputs[0].min().item())


