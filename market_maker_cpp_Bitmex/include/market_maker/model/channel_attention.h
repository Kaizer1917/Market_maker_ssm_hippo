#pragma once

#include <torch/torch.h>

class ChannelAttention : public torch::nn::Module {
public:
    ChannelAttention(int num_channels, int reduction_ratio = 8) {
        avg_pool_ = register_module("avg_pool", 
            torch::nn::AdaptiveAvgPool1d(torch::nn::AdaptiveAvgPool1dOptions(1)));
        
        max_pool_ = register_module("max_pool",
            torch::nn::AdaptiveMaxPool1d(torch::nn::AdaptiveMaxPool1dOptions(1)));
            
        fc1_ = register_module("fc1",
            torch::nn::Linear(num_channels, num_channels / reduction_ratio));
            
        fc2_ = register_module("fc2",
            torch::nn::Linear(num_channels / reduction_ratio, num_channels));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto avg_out = avg_pool_(x).squeeze(-1);
        auto max_out = max_pool_(x).squeeze(-1);
        
        avg_out = fc2_->forward(torch::relu(fc1_->forward(avg_out)));
        max_out = fc2_->forward(torch::relu(fc1_->forward(max_out)));
        
        auto out = torch::sigmoid(avg_out + max_out);
        return out.unsqueeze(-1);
    }

private:
    torch::nn::AdaptiveAvgPool1d avg_pool_{nullptr};
    torch::nn::AdaptiveMaxPool1d max_pool_{nullptr};
    torch::nn::Linear fc1_{nullptr};
    torch::nn::Linear fc2_{nullptr};
}; 