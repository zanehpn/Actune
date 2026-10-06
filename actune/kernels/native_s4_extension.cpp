#include <torch/extension.h>

torch::Tensor native_s4s4_linear_cuda(
    torch::Tensor packed_activation,
    torch::Tensor activation_scale,
    torch::Tensor packed_weight,
    torch::Tensor weight_scale,
    torch::Tensor bias,
    torch::Tensor output_template,
    int64_t weight_bits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "linear",
      &native_s4s4_linear_cuda,
      "Packed S4xS4 Tensor-Core Linear (CUDA)");
}
