'''
#  filename: r2plus1d.py
#  R(2+1)d model
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
# import resnet
from models import resnet


class TAM(nn.Module):
    def __init__(self,
                 in_channels,
                 n_segment=6,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(TAM, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        print('TAM with kernel_size {}.'.format(kernel_size))

        self.G = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2), nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False), nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        '''

        :param x: input tensor, in shape of [N, C, T, H, W]
        :return:
        '''
        n, c, t, h, w = x.size()
        t = self.n_segment
        out = F.adaptive_avg_pool2d(x.view(n * c, t, h, w), (1, 1))
        out = out.view(-1, t)
        conv_kernel = self.G(out.view(-1, t)).view(n * c, 1, -1, 1)
        local_activation = self.L(out.view(n, c, t)).view(n, c, t, 1, 1)
        new_x = x * local_activation
        out = F.conv2d(new_x.view(1, n * c, t, h * w),
                       conv_kernel,
                       bias=None,
                       stride=(self.stride, 1),
                       padding=(self.padding, 0),
                       groups=n * c)
        out = out.view(n, c, t, h, w)
        # out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)

        return out


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, padding=1):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.padding = padding
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride, padding),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.tam = TAM(planes)
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, 1, padding),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.tam(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def r2plus1d_18_layer(planes):

    layers = []
    layers.append(BasicBlock(inplanes=planes, planes=planes,
                             conv_builder=Conv2Plus1D))
    layers.append(BasicBlock(inplanes=planes, planes=planes,
                             conv_builder=Conv2Plus1D))

    return nn.Sequential(*layers)




