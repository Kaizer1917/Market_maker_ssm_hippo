#include "mamba_tvm.h"
#include <tvm/runtime/registry.h>

MambaTVM::MambaTVM(int batch_size, int num_channels, int seq_len)
    : batch_size_(batch_size),
      num_channels_(num_channels),
      seq_len_(seq_len) {
    load_tvm_model();
}

void MambaTVM::load_tvm_model() {
    // Load compiled TVM model
    mod_ = tvm::runtime::Module::LoadFromFile("ssm_hippo_lib.so");
    ssm_forward_ = mod_.GetFunction("ssm_forward");
}

torch::Tensor MambaTVM::forward(const torch::Tensor& x) {
    // Convert PyTorch tensor to DLTensor
    DLTensor* x_dl;
    TVMArrayFromTensor(x, &x_dl);
    
    // Prepare output tensor
    std::vector<int64_t> out_shape = {batch_size_, num_channels_, seq_len_};
    auto options = torch::TensorOptions()
        .dtype(x.dtype())
        .device(x.device());
    auto output = torch::empty(out_shape, options);
    DLTensor* out_dl;
    TVMArrayFromTensor(output, &out_dl);
    
    // Call TVM function
    ssm_forward_(x_dl, out_dl);
    
    return output;
} 