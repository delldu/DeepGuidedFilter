#include <iostream>
#include <vector>

#include <torch/torch.h>
// #include <torch/csrc/inductor/aoti_model_container_runner_cuda.h>
#include <torch/csrc/inductor/aoti_model_container_runner.h>

int main() {
    c10::InferenceMode mode;

    // torch::inductor::AOTIModelContainerRunnerCuda runner("model.so");
    torch::inductor::AOTIModelContainerRunnerCpu runner((char *)"image_autops.so");

    // std::vector<torch::Tensor> inputs = {torch::randn({8, 10}, at::kCUDA)};
    std::vector<torch::Tensor> inputs = {torch::randn({1, 3, 1024, 1024}, at::kCPU)};

    std::vector<torch::Tensor> outputs = runner.run(inputs);
    std::cout << "Result from the first inference:"<< std::endl;
    // std::cout << outputs[0] << std::endl;

    // The second inference uses a different batch size and it works because we
    // specified that dimension as dynamic when compiling model.so.
    std::cout << "Result from the second inference:"<< std::endl;
    // std::cout << runner.run({torch::randn({3, 1024, 1024}, at::kCUDA)})[0] << std::endl;
    // std::cout << runner.run({torch::randn({3, 1024, 1024}, at::kCPU)})[0] << std::endl;

    runner.run({torch::randn({3, 1024, 1024}, at::kCPU)})[0];

    return 0;
}

