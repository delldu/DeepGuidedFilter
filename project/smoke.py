import os
import time

import pdb
import random
import torch

import image_autops

from tqdm import tqdm


if __name__ == "__main__":
    model, device = image_autops.get_model()

    N = 100
    B, C, H, W = 1, 3, 2048, 2048

    start_time = time.time()
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        h = random.randint(0, 32)
        w = random.randint(0, 32)
        x = torch.randn(B, C, H + h, W + w)

        try:
            with torch.no_grad():
                y = model(x.to(device))
            torch.cuda.synchronize()
        except:
            print("x: ", x.size())

    print("Average spend time: ", (time.time() - start_time) / N, "seconds")
    os.system("nvidia-smi | grep python")
