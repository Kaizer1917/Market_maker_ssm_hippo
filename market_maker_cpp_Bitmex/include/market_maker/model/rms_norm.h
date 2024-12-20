#pragma once

#include <torch/torch.h>

class RMSNorm : public torch::nn::Module {
public:
    RMSNorm(int d_model, double eps = 1e-5) 
        : eps_(eps) {
        weight_ = register_parameter("weight", torch::ones(d_model));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto variance = x.pow(2).mean(-1, true);
        x = x * torch::rsqrt(variance + eps_);
        return x * weight_;
    }

private:
    double eps_;
    torch::Tensor weight_{nullptr};
}; 