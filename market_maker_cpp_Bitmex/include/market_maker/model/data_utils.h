#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include "model_args.h"

class DataPreprocessor {
public:
    DataPreprocessor(const ModelArgs& args);
    
    std::pair<torch::Tensor, torch::Tensor> prepare_data(
        const torch::Tensor& data,
        bool is_training = true
    );
    
    torch::Tensor normalize_data(const torch::Tensor& data);
    torch::Tensor denormalize_data(const torch::Tensor& data);
    
    void update_statistics(const torch::Tensor& data);
    void save_statistics(const std::string& path);
    void load_statistics(const std::string& path);

private:
    ModelArgs args_;
    torch::Tensor mean_;
    torch::Tensor std_;
    bool statistics_computed_{false};
    
    torch::Tensor create_patches(const torch::Tensor& data);
    torch::Tensor apply_channel_mixup(const torch::Tensor& data);
};

class DataLoader {
public:
    DataLoader(
        const torch::Tensor& data,
        const ModelArgs& args,
        bool shuffle = true
    );
    
    std::pair<torch::Tensor, torch::Tensor> next_batch();
    void reset();
    bool has_next() const;
    
private:
    torch::Tensor data_;
    ModelArgs args_;
    bool shuffle_;
    int current_index_{0};
    std::vector<size_t> indices_;
    
    void shuffle_indices();
}; 