import torch.nn as nn

from model.multi_encoder_plain_unet import MultiEncoderPlainUNet
from model.single_encoder_plain_unet import SingleEncoderPlainUNet


class DeepSupervisionUNetEval(nn.Module):
    def __init__(self, net):
        super(DeepSupervisionUNetEval, self).__init__()
        self.net = net
    
    def forward(self, x):
        return self.net(x)[0]


def get_multi_encoder_plain_unet(args):
    layer_args = {
        "kernel_size" : 3,
        "conv_bias"   : True,
        "dropout_prob": args.dropout_prob,
        "norm"        : args.norm
    }
    
    unet = MultiEncoderPlainUNet(
        num_encoder               = args.num_modality,
        input_channels            = args.input_channels,
        num_classes               = args.num_classes,
        num_downsample            = args.num_downsample,
        num_blocks_per_stage      = args.blocks_per_stage,
        layer_args                = layer_args,
        deep_supervision          = args.deep_supervision,
        encoder_base_num_features = 32 // args.num_modality  # 32 // 4
    )

    return unet


def get_single_encoder_plain_unet(args):
    layer_args = {
        "kernel_size" : 3,
        "conv_bias"   : True,
        "dropout_prob": args.dropout_prob,
        "norm"        : args.norm
    }
    
    unet = SingleEncoderPlainUNet(
        input_channels            = args.input_channels,
        num_classes               = args.num_classes,
        num_downsample            = args.num_downsample,
        num_blocks_per_stage      = args.blocks_per_stage,
        layer_args                = layer_args,
        deep_supervision          = args.deep_supervision,
        encoder_base_num_features = 32
    )

    return unet


def select_model(args):
    type2model = {
        'mencoder_plain_unet': get_multi_encoder_plain_unet(args),
        'sencoder_plain_unet': get_single_encoder_plain_unet(args),
        # 'mencoder_res_unet' : None,
        # 'mencoder_dense_unet' : None,
        # 'mencoder_att_unet' : None,
    }
    model = type2model[args.model_type]
    return model