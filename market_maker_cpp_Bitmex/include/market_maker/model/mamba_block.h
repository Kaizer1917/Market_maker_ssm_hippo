#pragma once

#include <torch/torch.h>
#include "model_args.h"

class MambaBlock : public torch::nn::Module {
public:
    explicit MambaBlock(const ModelArgs& args);
    
    torch::Tensor forward(torch::Tensor x, double training_progress = 0.0);
    
private:
    ModelArgs args_;
    torch::nn::Linear in_proj_{nullptr};
    torch::nn::Conv1d conv1d_{nullptr};
    torch::nn::Linear x_proj_{nullptr};
    torch::nn::Linear dt_proj_{nullptr};
    torch::nn::Parameter A_log_{nullptr};
    torch::nn::Parameter D_{nullptr};
    torch::nn::Linear out_proj_{nullptr};

    torch::Tensor selective_scan(
        const torch::Tensor& u,
        const torch::Tensor& delta,
        const torch::Tensor& A,
        const torch::Tensor& B,
        const torch::Tensor& C,
        const torch::Tensor& D);

    torch::Tensor ssm(const torch::Tensor& x, const torch::Tensor& A, const torch::Tensor& B);
}; 