#pragma once

#include <torch/torch.h>
#include "model_args.h"
#include "losses.h"
#include "regularization.h"
#include "ssm_hippo.h"

class ModelTrainer {
public:
    ModelTrainer(
        const ModelArgs& args,
        std::shared_ptr<SSMHippo> model,
        std::shared_ptr<torch::optim::Optimizer> optimizer
    );

    void train(
        const torch::Tensor& train_data,
        const torch::Tensor& val_data,
        const std::string& checkpoint_path = "model_checkpoint.pt"
    );

    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    
private:
    ModelArgs args_;
    std::shared_ptr<SSMHippo> model_;
    std::shared_ptr<torch::optim::Optimizer> optimizer_;
    std::unique_ptr<AdaptiveTemporalCoherenceLoss> criterion_;
    std::unique_ptr<AdaptiveRegularization> regularizer_;
    
    int current_epoch_{0};
    float best_val_loss_{std::numeric_limits<float>::max()};
    int patience_counter_{0};
    
    void train_epoch(const torch::Tensor& train_data);
    float validate(const torch::Tensor& val_data);
    void update_learning_rate();
}; 