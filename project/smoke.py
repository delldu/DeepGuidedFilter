# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import pdb
import os
import time
import random
import torch

import image_autops

from tqdm import tqdm

if __name__ == "__main__":
    model, device = image_autops.get_autops_model()

    N = 100
    B, C, H, W = 1, 3, model.max_h, model.max_w

    mean_time = 0
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        h = random.randint(-16, 16)
        w = random.randint(-16, 16)
        x = torch.randn(B, C, H + h, W + w)
        # print("x: ", x.size())

        start_time = time.time()
        with torch.no_grad():
            y = model(x.to(device))
        torch.cuda.synchronize()
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as p:
        for ii in range(100):
            image = torch.randn(B, C, H, W)
            with torch.no_grad():
                y = model(image.to(device))
            torch.cuda.synchronize()
        p.step()

    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    os.system("nvidia-smi | grep python")