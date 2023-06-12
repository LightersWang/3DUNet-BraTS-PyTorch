from .blocks import PlainBlock, ResidualBlock
from .unet import UNet, MultiEncoderUNet
from monai.networks.nets import UNETR


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
    elif args.unet_arch == 'unetr':
        return UNETR(
            spatial_dims=3,
            in_channels=args.input_channels,
            out_channels=args.num_classes,
            img_size=(args.patch_size, args.patch_size, args.patch_size),
            norm_name=args.norm,
            dropout_rate=args.dropout_prob,
        )
    else:
        raise NotImplementedError(args.unet_arch + " is not implemented.")