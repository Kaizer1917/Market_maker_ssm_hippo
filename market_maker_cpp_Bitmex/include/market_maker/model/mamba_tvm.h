#pragma once

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <torch/torch.h>

class MambaTVM {
public:
    MambaTVM(int batch_size, int num_channels, int seq_len);
    
    torch::Tensor forward(const torch::Tensor& x);
    
private:
    tvm::runtime::Module mod_;
    tvm::runtime::PackedFunc ssm_forward_;
    int batch_size_;
    int num_channels_;
    int seq_len_;
    
    void load_tvm_model();
}; 