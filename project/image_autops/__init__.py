"""Image/Video Autops Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import torch

import todos
from . import guided

import pdb

def get_autops_model():
    """Create model."""
    base = guided.DeepGuidedFilter()
    model = todos.model.ResizePadModel(base)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);
    todos.data.mkdir("output")
    if not os.path.exists("output/image_autops.torch"):
        model.save("output/image_autops.torch")

    print(f"Running model on {device} ...")

    return model, device


def image_autops_predict(input_files, output_dir, horizon=None):
    from tqdm import tqdm

    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_autops_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        B, C, H, W = input_tensor.size()

        if horizon is not None:
            if not ((horizon and W >= H) or (not horizon and H >= W)): # match ?
                continue

        orig_tensor = input_tensor.clone().detach()
        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()

