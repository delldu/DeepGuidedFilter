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
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import todos
# import ggml_engine
import pdb


def diff_rows(input, r: int):
    assert input.dim() == 4
    # input.size() -- [1, 1, 50, 64]
    left = input[:, :, r : 2 * r + 1] # size() -- [1, 1, 2, 64]
    middle = input[:, :, 2 * r + 1 :] - input[:, :, : -2 * r - 1] # size() -- [1, 1, 47, 64]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1 : -r - 1] # size() -- [1, 1, 1, 64]

    output = torch.cat([left, middle, right], dim=2) # [1, 1, 50, 64]

    return output


def diff_cols(input, r: int):
    assert input.dim() == 4

    left = input[:, :, :, r : 2 * r + 1]
    middle = input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1 : -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = r

    def forward(self, x):
        assert x.dim() == 4
        # tensor [x] size: [1, 1, 50, 64], min: 1.0, max: 1.0, mean: 1.0
        return diff_cols(diff_rows(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-5):
        super().__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, lr_x, lr_y, hr_x):
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        ## N
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
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=False)

        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=False)

        output = mean_A * hr_x + mean_b

        return output.clamp(0.0, 1.0)

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))
        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


def build_lr_net(layer=5):
    layers = [
        nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        AdaptiveNorm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1, layer):
        layers += [
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=2 ** l, dilation=2 ** l, bias=False),
            AdaptiveNorm(24),
            nn.LeakyReLU(0.2, inplace=True),
        ]

    layers += [
        nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        AdaptiveNorm(24),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(24, 3, kernel_size=1, stride=1, padding=0, dilation=1),
    ]

    net = nn.Sequential(*layers)

    return net


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-5):
        super().__init__()
        self.MAX_H = 4096
        self.MAX_W = 4096
        self.MAX_TIMES = 1
        # GPU: 7.8G, 70ms

        self.lr = build_lr_net()
        self.gf = FastGuidedFilter(radius, eps)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, 15, 1, bias=False), 
            AdaptiveNorm(15), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(15, 3, 1)
        )

        self.load_weights(model_path="models/image_autops.pth")

    def load_weights(self, model_path="models/image_autops.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

    def forward(self, x_hr):
        # x_lr
        B, C, H, W = x_hr.size()

        x_lr = F.interpolate(x_hr, (H//8, W//8), mode="bilinear", align_corners=False)

        return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr))

if __name__ == "__main__":
    model = DeepGuidedFilter()
