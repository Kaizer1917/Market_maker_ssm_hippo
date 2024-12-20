#pragma once

#include <torch/torch.h>

struct SSMLayerConfig {
    int d_model;
    int d_state;
    torch::Tensor A;
    torch::Tensor B;
};

class SSMLayer : public torch::nn::Module {
public:
    explicit SSMLayer(const SSMLayerConfig& config);
    torch::Tensor forward(torch::Tensor x, double training_progress = 1.0);

private:
    SSMLayerConfig config_;
    torch::Tensor A_;
    torch::Tensor B_;
    torch::nn::Linear in_proj_{nullptr};
    torch::nn::Linear out_proj_{nullptr};
}; 