#include "trainer.h"
#include <iostream>
#include <fstream>
#include <json/json.h>

ModelTrainer::ModelTrainer(
    const ModelArgs& args,
    std::shared_ptr<SSMHippo> model,
    std::shared_ptr<torch::optim::Optimizer> optimizer
) : args_(args),
    model_(model),
    optimizer_(optimizer),
    criterion_(std::make_unique<AdaptiveTemporalCoherenceLoss>()),
    regularizer_(std::make_unique<AdaptiveRegularization>(model)) {
}

void ModelTrainer::train(
    const torch::Tensor& train_data,
    const torch::Tensor& val_data,
    const std::string& checkpoint_path
) {
    const int patience = 10;
    
    for (int epoch = 0; epoch < args_.num_epochs; ++epoch) {
        current_epoch_ = epoch;
        float training_progress = static_cast<float>(epoch) / args_.num_epochs;
        
        // Training phase
        model_->train();
        train_epoch(train_data);
        
        // Validation phase
        model_->eval();
        float val_loss = validate(val_data);
        
        // Early stopping check
        if (val_loss < best_val_loss_) {
            best_val_loss_ = val_loss;
            patience_counter_ = 0;
            save_checkpoint(checkpoint_path);
        } else {
            patience_counter_++;
            if (patience_counter_ >= patience) {
                std::cout << "Early stopping triggered\n";
                break;
            }
        }
        
        // Update learning rate
        update_learning_rate();
        
        if (args_.verbose) {
            std::cout << "Epoch " << epoch + 1 << "/" << args_.num_epochs 
                     << ", Val Loss: " << val_loss << "\n";
        }
    }
}

void ModelTrainer::train_epoch(const torch::Tensor& train_data) {
    float training_progress = static_cast<float>(current_epoch_) / args_.num_epochs;
    
    for (int i = 0; i < train_data.size(0); i += args_.batch_size) {
        auto batch = train_data.slice(0, i, std::min(i + args_.batch_size, 
                                                    static_cast<int>(train_data.size(0))));
        
        optimizer_->zero_grad();
        
        auto x = batch.slice(1, 0, -args_.forecast_len);
        auto y = batch.slice(1, -args_.forecast_len);
        
        // Forward pass with regularization
        auto [output, reg_loss] = regularizer_->forward(
            model_->forward(x, training_progress),
            training_progress
        );
        
        // Calculate loss
        auto loss = criterion_->forward(output, y, training_progress) + reg_loss;
        
        // Backward pass
        loss.backward();
        
        // Gradient clipping
        torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
        
        optimizer_->step();
    }
}

float ModelTrainer::validate(const torch::Tensor& val_data) {
    torch::NoGradGuard no_grad;
    float total_loss = 0.0;
    int num_batches = 0;
    
    for (int i = 0; i < val_data.size(0); i += args_.batch_size) {
        auto batch = val_data.slice(0, i, std::min(i + args_.batch_size, 
                                                  static_cast<int>(val_data.size(0))));
        
        auto x = batch.slice(1, 0, -args_.forecast_len);
        auto y = batch.slice(1, -args_.forecast_len);
        
        auto output = model_->forward(x, 1.0);
        total_loss += criterion_->forward(output, y, 1.0).item<float>();
        num_batches++;
    }
    
    return total_loss / num_batches;
}

void ModelTrainer::save_checkpoint(const std::string& path) {
    torch::save(model_, path + ".pt");
    
    Json::Value config;
    config["current_epoch"] = current_epoch_;
    config["best_val_loss"] = best_val_loss_;
    config["patience_counter"] = patience_counter_;
    
    std::ofstream config_file(path + "_config.json");
    config_file << config;
}

void ModelTrainer::load_checkpoint(const std::string& path) {
    torch::load(model_, path + ".pt");
    
    std::ifstream config_file(path + "_config.json");
    Json::Value config;
    config_file >> config;
    
    current_epoch_ = config["current_epoch"].asInt();
    best_val_loss_ = config["best_val_loss"].asFloat();
    patience_counter_ = config["patience_counter"].asInt();
}

void ModelTrainer::update_learning_rate() {
    float progress = static_cast<float>(current_epoch_) / args_.num_epochs;
    
    // Implement learning rate schedule
    if (progress < 0.1) {
        // Warmup phase
        float lr = args_.learning_rate * (progress / 0.1);
        for (auto& group : optimizer_->param_groups()) {
            group.options().set_lr(lr);
        }
    } else {
        // Cosine annealing
        float lr = args_.learning_rate * 0.5 * (1 + cos(M_PI * (progress - 0.1) / 0.9));
        for (auto& group : optimizer_->param_groups()) {
            group.options().set_lr(lr);
        }
    }
} 