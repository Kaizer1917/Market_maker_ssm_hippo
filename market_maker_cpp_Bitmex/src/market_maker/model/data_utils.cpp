#include "data_utils.h"
#include <random>
#include <fstream>
#include <json/json.h>

DataPreprocessor::DataPreprocessor(const ModelArgs& args) : args_(args) {}

std::pair<torch::Tensor, torch::Tensor> DataPreprocessor::prepare_data(
    const torch::Tensor& data,
    bool is_training
) {
    // Update statistics if training
    if (is_training) {
        update_statistics(data);
    }
    
    // Normalize data
    auto normalized_data = normalize_data(data);
    
    // Create patches
    auto patched_data = create_patches(normalized_data);
    
    // Apply channel mixup during training
    if (is_training) {
        patched_data = apply_channel_mixup(patched_data);
    }
    
    // Split into input and target
    auto x = patched_data.slice(1, 0, -args_.forecast_len);
    auto y = patched_data.slice(1, -args_.forecast_len);
    
    return {x, y};
}

torch::Tensor DataPreprocessor::normalize_data(const torch::Tensor& data) {
    if (!statistics_computed_) {
        throw std::runtime_error("Statistics not computed. Call update_statistics first.");
    }
    return (data - mean_) / std_;
}

torch::Tensor DataPreprocessor::denormalize_data(const torch::Tensor& data) {
    if (!statistics_computed_) {
        throw std::runtime_error("Statistics not computed. Call update_statistics first.");
    }
    return data * std_ + mean_;
}

void DataPreprocessor::update_statistics(const torch::Tensor& data) {
    mean_ = data.mean({0, 1});
    std_ = data.std({0, 1});
    std_ = torch::where(std_ == 0, torch::ones_like(std_), std_);
    statistics_computed_ = true;
}

torch::Tensor DataPreprocessor::create_patches(const torch::Tensor& data) {
    std::vector<torch::Tensor> patches;
    
    for (int i = 0; i < data.size(1) - args_.patch_len + 1; i += args_.stride) {
        auto patch = data.slice(1, i, i + args_.patch_len);
        patches.push_back(patch);
    }
    
    return torch::stack(patches, 1);
}

torch::Tensor DataPreprocessor::apply_channel_mixup(const torch::Tensor& data) {
    if (!args_.sigma > 0) {
        return data;
    }
    
    auto batch_size = data.size(0);
    auto num_channels = data.size(2);
    
    // Generate random permutation
    auto perm = torch::randperm(num_channels);
    
    // Generate mixing coefficients
    auto lambda = torch::normal(0, args_.sigma, {num_channels}, data.options());
    
    // Apply mixup
    auto mixed_data = data + lambda.unsqueeze(1) * data.index({torch::indexing::Slice(), 
                                                              torch::indexing::Slice(), 
                                                              perm});
    
    return mixed_data;
}

void DataPreprocessor::save_statistics(const std::string& path) {
    if (!statistics_computed_) {
        throw std::runtime_error("No statistics to save.");
    }
    
    torch::save(mean_, path + "_mean.pt");
    torch::save(std_, path + "_std.pt");
}

void DataPreprocessor::load_statistics(const std::string& path) {
    torch::load(mean_, path + "_mean.pt");
    torch::load(std_, path + "_std.pt");
    statistics_computed_ = true;
}

// DataLoader implementation
DataLoader::DataLoader(
    const torch::Tensor& data,
    const ModelArgs& args,
    bool shuffle
) : data_(data),
    args_(args),
    shuffle_(shuffle) {
    
    indices_.resize(data.size(0));
    std::iota(indices_.begin(), indices_.end(), 0);
    
    if (shuffle_) {
        shuffle_indices();
    }
}

std::pair<torch::Tensor, torch::Tensor> DataLoader::next_batch() {
    if (!has_next()) {
        throw std::runtime_error("No more batches available.");
    }
    
    int batch_end = std::min(current_index_ + args_.batch_size, 
                            static_cast<int>(indices_.size()));
    
    std::vector<torch::Tensor> batch_tensors;
    for (int i = current_index_; i < batch_end; ++i) {
        batch_tensors.push_back(data_[indices_[i]]);
    }
    
    auto batch = torch::stack(batch_tensors);
    current_index_ = batch_end;
    
    // Split into input and target
    auto x = batch.slice(1, 0, -args_.forecast_len);
    auto y = batch.slice(1, -args_.forecast_len);
    
    return {x, y};
}

void DataLoader::reset() {
    current_index_ = 0;
    if (shuffle_) {
        shuffle_indices();
    }
}

bool DataLoader::has_next() const {
    return current_index_ < static_cast<int>(indices_.size());
}

void DataLoader::shuffle_indices() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices_.begin(), indices_.end(), gen);
} 