import os
from tqdm import tqdm
import todos
import pdb


def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    tvm_model = todos.tvmod.load("output/image_autops.so", "cuda")

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        B, C, H, W = input_tensor.shape
        input_tensor = todos.data.resize_tensor(input_tensor, 512, 512)

        predict_tensor = todos.tvmod.forward(tvm_model, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        predict_tensor = todos.data.resize_tensor(predict_tensor, H, W)

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)


predict("images/*.png", "output/so")
