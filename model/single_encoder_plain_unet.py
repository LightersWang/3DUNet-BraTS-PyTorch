import torch
import torch.nn as nn

from model.submodules.plain_unet_encoder import PlainUNetEncoder
from model.submodules.single_encoder_unet_decoder import SingleEncoderUNetDecoder


class SingleEncoderPlainUNet(nn.Module):
    def __init__(self, input_channels, num_classes, num_downsample, num_blocks_per_stage, 
                 deep_supervision=True, encoder_base_num_features=32, featmap_mul_downsample=2, **layer_args):
        super(SingleEncoderPlainUNet, self).__init__()
        
        self.num_classes = num_classes
        self.num_downsample = num_downsample
        self.num_blocks_per_stage = num_blocks_per_stage
        self.deep_supervision = deep_supervision

        # encoders
        self.encoder = PlainUNetEncoder(
            input_channels=input_channels,
            num_downsample=self.num_downsample,
            num_blocks_per_stage=self.num_blocks_per_stage,
            base_num_features=encoder_base_num_features,
            featmap_mul_downsample=featmap_mul_downsample,
            **layer_args
        )

        # decoder
        self.decoder = SingleEncoderUNetDecoder(
            single_modal_encoder=self.encoder,
            num_classes=self.num_classes,
            deep_supervision=deep_supervision,
            **layer_args
        )


    def forward(self, x):
        return self.decoder(self.encoder(x))


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
    unet = SingleEncoderPlainUNet(4, 3, 5, 2, deep_supervision=True, encoder_base_num_features=32, **layer_args).cuda()
    unet = nn.DataParallel(unet, device_ids=[0, 1])
    outputs = unet(x_patch)
    print("========================================================")
    for op in outputs:
        print(op.shape)

    # slding window test
    from monai.inferers import sliding_window_inference

    with torch.no_grad():
        x_real = torch.rand(1, 4, 240, 240, 155).cuda()
        unet_eval = DeepSupervisionUNetEval(unet)
        x_infer = sliding_window_inference(
            inputs=x_real, roi_size=128, sw_batch_size=x_real.shape[0], overlap=0.5,
            predictor=unet_eval
        )
        print(len(x_infer))
