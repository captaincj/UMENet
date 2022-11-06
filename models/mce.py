'''
#  filename: mce.py
#  Likun Qin, 2021
'''

import torch
import torch.nn as nn


class MCEModule(nn.Module):
    """ Motion exciation module

    :param reduction=16
    :param n_segment=8/16
    """

    def __init__(self, channel, reduction=8):
        super(MCEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel // self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False

        self.conv2 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel // self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel // self.reduction,
            bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        # self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)
        # for i in self.bn3.parameters():
        #     i.requires_grad = False

        # self.identity = nn.Identity()

    def forward(self, x, ex, time_step):
        '''

        :param x: feature of current frame, [batch_size, self.channel, 16, 16]
        :param ex:  feature of the previous frame, [batch_size, self.channel, 16, 16]
        :param time_step: the position in a sequence, int
        :return:
        '''
        if time_step == 0:
            return x

        n, c, h, w = x.size()
        bottleneck = self.conv1(x)  # n, c//r, h, w
        bottleneck = self.bn1(bottleneck)  # n, c//r, h, w

        # apply transformation conv to t feature
        conv_bottleneck = self.conv2(bottleneck)  # n, c//r, h, w

        ex_bottleneck = self.conv1(ex)
        ex_bottleneck = self.bn1(ex_bottleneck)  # n, c//r, h, w

        # # t-1 feature
        # reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
        # t_fea, __ = reshape_bottleneck.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w
        #
        # # reshape fea: n, t, c//r, h, w
        # reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:])
        # __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w

        # motion fea = t_fea - t-1_fea
        # pad the last timestamp
        diff_fea = conv_bottleneck - ex_bottleneck  # n, c//r, h, w
        # pad = (0,0,0,0,0,0,1,0)
        # pad at the start
        # diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
        # diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  # nt, c//r, h, w
        y = self.avg_pool(diff_fea)  # n, c//r, 1, 1
        y = self.conv3(y)  # n, c, 1, 1
        y = self.bn3(y)  # n, c, 1, 1
        y = self.sigmoid(y)  # n, c, 1, 1
        y = y - 0.5
        output = x + x * y.expand_as(x)
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
