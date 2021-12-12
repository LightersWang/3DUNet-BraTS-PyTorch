import sys
sys.path.append("/home/whr/Code/multi_modal_al/seg/")

import torch
import torch.nn as nn

from model.submodules.plain_unet_encoder import PlainUNetEncoder
from model.submodules.multi_encoder_unet_decoder import MultiEncoderUNetDecoder


class MultiEncoderPlainUNet(nn.Module):
    def __init__(self, num_encoder, input_channels, num_classes, num_downsample, num_blocks_per_stage, 
                 layer_args, deep_supervision=True, encoder_base_num_features=8, featmap_mul_downsample=2):
        super(MultiEncoderPlainUNet, self).__init__()
        
        self.num_encoder = num_encoder
        self.num_classes = num_classes
        self.num_downsample = num_downsample
        self.num_blocks_per_stage = num_blocks_per_stage
        self.deep_supervision = deep_supervision

        # encoders
        self.encoders = []
        for _ in range(self.num_encoder):
            encoder = PlainUNetEncoder(
                input_channels=input_channels,
                num_downsample=self.num_downsample,
                num_blocks_per_stage=self.num_blocks_per_stage,
                layer_args=layer_args,
                base_num_features=encoder_base_num_features,
                featmap_mul_downsample=featmap_mul_downsample
            )
            self.encoders.append(encoder)
        self.encoders = nn.ModuleList(self.encoders)

        # decoder
        self.decoder = MultiEncoderUNetDecoder(
            num_encoder=self.num_encoder,
            single_modal_encoder=self.encoders[0],
            num_classes=self.num_classes,
            layer_args=layer_args,
            deep_supervision=deep_supervision
        )


    def forward(self, x):
        # assume x is [B, C, H, W, D]
        multi_modal_skips = []
        for xx, encoder in zip(x.chunk(self.num_encoder, dim=1), self.encoders):
            multi_modal_skips.append(encoder(xx))

        return self.decoder(multi_modal_skips)


if __name__ == "__main__":
    class DeepSupervisionUNetEval(nn.Module):
        def __init__(self, net):
            super(DeepSupervisionUNetEval, self).__init__()
            self.net = net
        
        def forward(self, x):
            return self.net(x)[0]

    # normal test
    layer_args = {
        "kernel_size": 3, 
        "conv_bias": True, 
        "dropout_prob": None, 
        "norm": 'instance'
    }

    x_patch = torch.rand(2, 4, 128, 128, 128).cuda()
    unet = MultiEncoderPlainUNet(4, 1, 3, 5, 2, layer_args, deep_supervision=True)
    unet = unet.cuda()
    unet = nn.DataParallel(unet, device_ids=[0, 1])
    # outputs = unet(x_patch)
    # print("========================================================")
    # for op in outputs:
    #     print(op.shape)

    # slding window test
    from monai.inferers import sliding_window_inference

    with torch.no_grad():
        x_real = torch.rand(2, 4, 240, 240, 155).cuda()
        unet_eval = DeepSupervisionUNetEval(unet)
        x_infer = sliding_window_inference(
            inputs=x_real, roi_size=128, sw_batch_size=2, overlap=0.5,
            predictor=unet_eval
        )
        print(len(x_infer))
