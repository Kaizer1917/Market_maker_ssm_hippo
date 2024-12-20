#pragma once

#include <torch/torch.h>

class AdaptiveRegularization : public torch::nn::Module {
public:
    AdaptiveRegularization(
        std::shared_ptr<torch::nn::Module> model,
        double dropout_rate = 0.1,
        double l1_factor = 1e-5,
        double l2_factor = 1e-4
    ) : model_(model),
        dropout_(register_module("dropout", torch::nn::Dropout(dropout_rate))),
        l1_factor_(l1_factor),
        l2_factor_(l2_factor) {}

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, double training_progress) {
        // Initialize regularization terms
        auto l1_reg = torch::tensor(0.0, x.options());
        auto l2_reg = torch::tensor(0.0, x.options());

        // Apply adaptive regularization to model parameters
        for (const auto& pair : model_->named_parameters()) {
            const auto& name = pair.key();
            const auto& param = pair.value();

            if (name.find("weight") != std::string::npos) {
                // Increase L1 regularization over time for sparsity
                l1_reg += l1_factor_ * training_progress * torch::norm(param, 1);
                // Decrease L2 regularization over time
                l2_reg += l2_factor_ * (1 - training_progress) * torch::norm(param, 2);
            }
        }

        // Apply dropout with adaptive rate
        x = dropout_->forward(x * (1 - 0.5 * training_progress));

        return {x, l1_reg + l2_reg};
    }

private:
    std::shared_ptr<torch::nn::Module> model_;
    torch::nn::Dropout dropout_{nullptr};
    double l1_factor_;
    double l2_factor_;
}; 