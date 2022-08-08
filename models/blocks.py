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


class Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)


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


class PlainBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, kernel_size=3, 
                 norm_key='instance', dropout_prob=None):
        super(PlainBlock, self).__init__()

        conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, 
                        padding=(kernel_size - 1) // 2, bias=True)

        do = Identity() if dropout_prob is None else nn.Dropout3d(p=dropout_prob, inplace=True)

        norm = norm_dict[norm_key](output_channels, eps=1e-5, affine=True)

        nonlin = nn.LeakyReLU(inplace=True)

        self.all = nn.Sequential(conv, do, norm, nonlin)

    def forward(self, x):
        return self.all(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, kernel_size=3, 
                 norm_key='instance', dropout_prob=None):
        super(ResidualBlock, self).__init__()

        conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, 
                         padding=(kernel_size - 1) // 2, bias=True)

        norm = norm_dict[norm_key](output_channels, eps=1e-5, affine=True)

        do = Identity() if dropout_prob is None else nn.Dropout3d(p=dropout_prob, inplace=True)

        nonlin = nn.LeakyReLU(inplace=True)

        self.all = nn.Sequential(conv, norm, do, nonlin)

        # downsample residual
        if (input_channels != output_channels) or (stride != 1):
            self.downsample_skip = nn.Sequential(
                nn.Conv3d(input_channels, output_channels, 1, stride, bias=True),
                norm_dict[norm_key](output_channels, eps=1e-5, affine=True), 
            )
        else:
            self.downsample_skip = lambda x: x

    def forward(self, x):
        residual = x

        out = self.all(x)

        residual = self.downsample_skip(x)

        return residual + out


# unit test
if __name__ == "__main__":
    # conv block
    conv_layers = PlainBlock(input_channels=3, output_channels=32, kernel_size=3, dropout_prob=None, stride=2)
    print(conv_layers)

    # residual block
    res_block = ResidualBlock(32, 64, kernel_size=3, stride=2, norm_key='instance', dropout_prob=None)
    print(res_block)
    x = torch.randn(1, 32, 64, 64, 64)
    print(x.shape)
    print(res_block(x).shape)
