import torch
import torch.nn as nn
import torch.nn.functional as F


norm_dict = {
    'instance': nn.InstanceNorm3d,
    'batch': nn.BatchNorm3d
}


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class ConvDropoutNormLeakyReLU(nn.Module):

    def __init__(self, input_channels, output_channels, stride=1, kernel_size=3, conv_bias=True, 
                 dropout_prob=0.5, norm='instance'):
        super(ConvDropoutNormLeakyReLU, self).__init__()

        conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, 
                        padding=(kernel_size - 1) // 2, bias=conv_bias)

        if dropout_prob is None:
            do = Identity()
        else:
            do = nn.Dropout3d(p=dropout_prob, inplace=True)

        norm = norm_dict[norm](output_channels, eps=1e-5, affine=True)

        nonlin = nn.LeakyReLU(inplace=True)

        self.all = nn.Sequential(conv, do, norm, nonlin)

    def forward(self, x):
        return self.all(x)


class StackedConvLayers(nn.Module):
    def __init__(self, num_convs, input_channels, output_channels, first_stride, **layer_args):
        super(StackedConvLayers, self).__init__()

        self.convs = nn.Sequential(
            ConvDropoutNormLeakyReLU(input_channels, output_channels, stride=first_stride, **layer_args),
            *[ConvDropoutNormLeakyReLU(output_channels, output_channels, stride=1, **layer_args) 
                for _ in range(num_convs - 1)]
        )

    def forward(self, x):
        return self.convs(x)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=self.align_corners)


if __name__ == "__main__":
    # test conv layers
    conv_layers = StackedConvLayers(2, input_channels=3, output_channels=32, kernel_size=3, dropout_prob=None, first_stride=2, transpose=True)
    print(conv_layers)

    # # test F.interpolate
    # x = torch.rand(2, 32, 2, 2, 2)
    # x_upsample_1 = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
    # print(x_upsample_1.shape)
    # print(x_upsample_1[0, 0])

    # # test Upsample
    # upsample = Upsample(scale_factor=2, mode='trilinear', align_corners=False)
    # x_upsample_2 = upsample(x)
    # print(x_upsample_2.shape)
    # print(x_upsample_2[0, 0])
