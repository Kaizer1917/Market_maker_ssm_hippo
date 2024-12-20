#pragma once

#include <torch/torch.h>

namespace model_utils {

inline torch::nn::Sequential create_mlp(
    int input_dim,
    int hidden_dim,
    int output_dim,
    int num_layers,
    double dropout_rate = 0.1
) {
    torch::nn::Sequential mlp;
    
    // Input layer
    mlp->push_back(torch::nn::Linear(input_dim, hidden_dim));
    mlp->push_back(torch::nn::LayerNorm(hidden_dim));
    mlp->push_back(torch::nn::GELU());
    mlp->push_back(torch::nn::Dropout(dropout_rate));

    // Hidden layers
    for (int i = 0; i < num_layers - 2; i++) {
        mlp->push_back(torch::nn::Linear(hidden_dim, hidden_dim));
        mlp->push_back(torch::nn::LayerNorm(hidden_dim));
        mlp->push_back(torch::nn::GELU());
        mlp->push_back(torch::nn::Dropout(dropout_rate));
    }

    // Output layer
    mlp->push_back(torch::nn::Linear(hidden_dim, output_dim));

    return mlp;
}

inline torch::optim::OptimizerOptions create_optimizer_options(
    const std::string& optimizer_type,
    double learning_rate,
    double weight_decay = 0.01
) {
    if (optimizer_type == "adam") {
        return torch::optim::AdamOptions(learning_rate)
            .weight_decay(weight_decay);
    } else if (optimizer_type == "adamw") {
        return torch::optim::AdamWOptions(learning_rate)
            .weight_decay(weight_decay);
    } else {
        throw std::runtime_error("Unsupported optimizer type: " + optimizer_type);
    }
}

} // namespace model_utils 