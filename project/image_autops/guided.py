"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2020-2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 09月 09日 星期三 23:56:45 CST
# ***
# ************************************************************************************/
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import pdb


def diff_x(input, r: int):
    assert input.dim() == 4

    left = input[:, :, r : 2 * r + 1]
    middle = input[:, :, 2 * r + 1 :] - input[:, :, : -2 * r - 1]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1 : -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r: int):
    assert input.dim() == 4

    left = input[:, :, :, r : 2 * r + 1]
    middle = input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1 : -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        ## N
        # N = self.boxfilter(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))
        N = self.boxfilter(torch.ones_like(lr_x)[:, 0:1, :, :])

        ## mean_x
        mean_x = self.boxfilter(lr_x) / N
        ## mean_y
        mean_y = self.boxfilter(lr_y) / N
        ## cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        ## var_x
        var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + self.eps)
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)

        return (mean_A * hr_x + mean_b).clamp(0.0, 1.0)


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        # N = self.boxfilter(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0))
        N = self.boxfilter(torch.ones_like(x)[:, 0:1, :, :])

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()

        self.box_filter = nn.Conv2d(3, 3, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=3)
        self.conv_a = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, bias=False),
        )
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 3, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr) / N
        ## mean_y
        mean_y = self.box_filter(y_lr) / N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        ## var_x
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)

        return mean_A * x_hr + mean_b


def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        n_out, n_in, h, w = m.weight.data.size()
        # Last Layer
        if n_out < n_in:
            init.xavier_uniform_(m.weight.data)
            return

        # Except Last Layer
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0

    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


def build_lr_net(norm=AdaptiveNorm, layer=5):
    layers = [
        nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1, layer):
        layers += [
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=2 ** l, dilation=2 ** l, bias=False),
            norm(24),
            nn.LeakyReLU(0.2, inplace=True),
        ]

    layers += [
        nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(24, 3, kernel_size=1, stride=1, padding=0, dilation=1),
    ]

    net = nn.Sequential(*layers)

    net.apply(weights_init_identity)

    return net


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super(DeepGuidedFilter, self).__init__()
        self.lr = build_lr_net()
        self.gf = FastGuidedFilter(radius, eps)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)

    # def init_lr(self, path):
    #     self.lr.load_state_dict(torch.load(path))


class DeepGuidedFilterAdvanced(DeepGuidedFilter):
    def __init__(self, radius=1, eps=1e-4):
        super(DeepGuidedFilterAdvanced, self).__init__(radius, eps)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, 15, 1, bias=False), AdaptiveNorm(15), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(15, 3, 1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x_hr):
        # x_lr
        x_lr = F.interpolate(x_hr, (128, 128), mode="bilinear", align_corners=True)
        return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr))