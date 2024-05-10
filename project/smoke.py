# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2024(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import os
import torch
import image_autops
import argparse
import pdb

def test_input_shape():
    import time
    import random
    from tqdm import tqdm

    print("Test input shape ...")

    model, device = image_autops.get_autops_model()

    N = 100
    B, C, H, W = 1, 3, model.MAX_H, model.MAX_W

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
        if 'cpu' not in str(device):
            torch.cuda.synchronize()
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")


def run_bench_mark():
    print("Run benchmark ...")

    model, device = image_autops.get_autops_model()
    N = 100
    B, C, H, W = 1, 3, model.MAX_H, model.MAX_W

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as p:
        for ii in range(N):
            image = torch.randn(B, C, H, W)
            with torch.no_grad():
                y = model(image.to(device))
            if 'cpu' not in str(device):
                torch.cuda.synchronize()
        p.step()

    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    os.system("nvidia-smi | grep python")


def export_onnx_model():
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md

    import onnx
    import onnxruntime
    from onnxsim import simplify
    import onnxoptimizer
    
    print("Export onnx model ...")

    # 1. Run torch model
    model, device = image_autops.get_autops_model()

    B, C, H, W = 1, 3, model.MAX_H, model.MAX_W
    dummy_input = torch.randn(B, C, H, W).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    torch_outputs = [dummy_output.cpu()]

    # 2. Export onnx model
    input_names = [ "input" ]
    output_names = [ "output" ]
    dynamic_axes = { 
        'input' : {2: 'height', 3: 'width'}, 
        'output' : {2: 'height', 3: 'width'} 
    } 
    onnx_filename = "output/image_autops.onnx"

    torch.onnx.export(model, dummy_input, onnx_filename, 
        verbose=False, 
        input_names=input_names, 
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # 3. Check onnx model file
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx_model = onnxoptimizer.optimize(onnx_model)    
    onnx.save(onnx_model, onnx_filename)
    # print(onnx.helper.printable_graph(onnx_model.graph))

    # 4. Run onnx model
    if 'cuda' in device.type:
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CUDAExecutionProvider'])
    else:        
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnx_inputs = {input_names[0]: to_numpy(dummy_input) }
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # 5.Compare output results
    assert len(torch_outputs) == len(onnx_outputs)
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnx_output), rtol=0.01, atol=0.01)

    print("!!!!!! Torch and ONNX Runtime output matched !!!!!!")

def compile_model():
    print("Compile model ...")
    model, device = image_autops.get_autops_model()
    B, C, H, W = 1, 3, model.MAX_H, model.MAX_W
    dummy_input = torch.randn(B, C, H, W).to(device)
    # with torch.no_grad():
    #     dummy_output = model(dummy_input)
    # torch_outputs = [dummy_output.cpu()]

    example_inputs=(dummy_input,)
    batch_dim = torch.export.Dim("batch", min=1, max=32)
    so_path = torch._export.aot_compile(
        model,
        example_inputs,
        # Specify the first dimension of the input x as dynamic
        # dynamic_shapes={"x": {0: batch_dim}},
        # Specify the generated shared library path
        options={"aot_inductor.output_path": os.path.join(os.getcwd(), "output/image_autops.so")},
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smoke Test')
    parser.add_argument('-s', '--shape_test', action="store_true", help="test shape")
    parser.add_argument('-b', '--bench_mark', action="store_true", help="test benchmark")
    parser.add_argument('-e', '--export_onnx', action="store_true", help="export onnx model")
    parser.add_argument('-c', '--compile', action="store_true", help="Compile model to autops.so")
    args = parser.parse_args()

    if args.shape_test:
        test_input_shape()
    if args.bench_mark:
        run_bench_mark()
    if args.export_onnx:
        export_onnx_model()
    if args.compile:
        compile_model()
    
    if not (args.shape_test or args.bench_mark or args.export_onnx or args.compile):
        parser.print_help()
