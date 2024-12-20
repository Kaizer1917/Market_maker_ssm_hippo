#pragma once

#include <torch/torch.h>

class AdaptiveTemporalCoherenceLoss : public torch::nn::Module {
public:
    AdaptiveTemporalCoherenceLoss(double alpha = 0.3, double beta = 0.2)
        : alpha_(alpha), beta_(beta) {}

    torch::Tensor forward(
        const torch::Tensor& pred,
        const torch::Tensor& target,
        double training_progress = 1.0
    ) {
        // MSE Loss
        auto mse_loss = torch::mse_loss(pred, target);

        // Temporal coherence loss
        auto pred_diff = pred.diff(1, -1);
        auto target_diff = target.diff(1, -1);
        auto temporal_loss = torch::mse_loss(pred_diff, target_diff);

        // Scale factors based on training progress
        auto mse_scale = 1.0 - alpha_ * training_progress;
        auto temporal_scale = beta_ * training_progress;

        return mse_scale * mse_loss + temporal_scale * temporal_loss;
    }

private:
    double alpha_;
    double beta_;
}; 