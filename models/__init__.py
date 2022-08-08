from .blocks import PlainBlock, ResidualBlock
from .unet import UNet, MultiEncoderUNet


block_dict = {
    'plain': PlainBlock,
    'res': ResidualBlock
}


def get_unet(args):
    kwargs = {
        "input_channels"   : args.input_channels,
        "output_classes"   : args.num_classes,
        "channels_list"    : args.channels_list,
        "deep_supervision" : args.deep_supervision,
        "ds_layer"         : args.ds_layer,
        "kernel_size"      : args.kernel_size,
        "dropout_prob"     : args.dropout_prob,
        "norm_key"         : args.norm,
        "block"            : block_dict[args.block],
    }
    
    if args.unet_arch == 'unet':
        return UNet(**kwargs)
    elif args.unet_arch == 'multiencoder_unet':
        return MultiEncoderUNet(**kwargs)