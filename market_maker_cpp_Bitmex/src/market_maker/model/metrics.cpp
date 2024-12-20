#include "metrics.h"
#include <cmath>
#include <fstream>
#include <json/json.h>

MetricsCalculator::Metrics MetricsCalculator::calculate_metrics(
    const torch::Tensor& predictions,
    const torch::Tensor& targets,
    bool reduce
) {
    Metrics metrics;
    
    // Calculate MSE
    auto mse = torch::mse_loss(predictions, targets, reduce ? torch::Reduction::Mean 
                                                          : torch::Reduction::None);
    metrics.mse = reduce ? mse.item<float>() : mse.mean().item<float>();
    
    // Calculate RMSE
    metrics.rmse = std::sqrt(metrics.mse);
    
    // Calculate MAE
    auto mae = torch::abs(predictions - targets).mean();
    metrics.mae = mae.item<float>();
    
    // Calculate R2 score
    metrics.r2 = calculate_r2_score(predictions, targets);
    
    // Calculate MAPE
    auto mape = torch::abs((predictions - targets) / (targets + 1e-8)) * 100.0;
    metrics.mape = mape.mean().item<float>();
    
    // Calculate directional accuracy
    metrics.directional_accuracy = calculate_directional_accuracy(predictions, targets);
    
    return metrics;
}

std::unordered_map<std::string, float> MetricsCalculator::calculate_channel_metrics(
    const torch::Tensor& predictions,
    const torch::Tensor& targets
) {
    std::unordered_map<std::string, float> channel_metrics;
    
    auto num_channels = predictions.size(-1);
    
    for (int i = 0; i < num_channels; ++i) {
        auto pred_channel = predictions.select(-1, i);
        auto target_channel = targets.select(-1, i);
        
        auto metrics = calculate_metrics(pred_channel, target_channel);
        
        channel_metrics["channel_" + std::to_string(i) + "_mse"] = metrics.mse;
        channel_metrics["channel_" + std::to_string(i) + "_r2"] = metrics.r2;
        channel_metrics["channel_" + std::to_string(i) + "_da"] = metrics.directional_accuracy;
    }
    
    return channel_metrics;
}

torch::Tensor MetricsCalculator::calculate_rolling_metrics(
    const torch::Tensor& predictions,
    const torch::Tensor& targets,
    int window_size
) {
    auto seq_len = predictions.size(1);
    std::vector<float> rolling_metrics;
    
    for (int i = window_size; i <= seq_len; ++i) {
        auto pred_window = predictions.slice(1, i - window_size, i);
        auto target_window = targets.slice(1, i - window_size, i);
        
        auto metrics = calculate_metrics(pred_window, target_window);
        rolling_metrics.push_back(metrics.mse);
    }
    
    return torch::tensor(rolling_metrics);
}

float MetricsCalculator::calculate_r2_score(
    const torch::Tensor& predictions,
    const torch::Tensor& targets
) {
    auto target_mean = targets.mean();
    auto total_sum_squares = torch::sum(torch::pow(targets - target_mean, 2));
    auto residual_sum_squares = torch::sum(torch::pow(targets - predictions, 2));
    
    return (1 - residual_sum_squares / total_sum_squares).item<float>();
}

float MetricsCalculator::calculate_directional_accuracy(
    const torch::Tensor& predictions,
    const torch::Tensor& targets
) {
    auto pred_diff = predictions.diff();
    auto target_diff = targets.diff();
    
    auto correct_directions = torch::sign(pred_diff) == torch::sign(target_diff);
    return correct_directions.to(torch::kFloat32).mean().item<float>();
}

// MetricsTracker implementation
MetricsTracker::MetricsTracker() {
    reset();
}

void MetricsTracker::update(const MetricsCalculator::Metrics& metrics) {
    total_metrics_.mse += metrics.mse;
    total_metrics_.rmse += metrics.rmse;
    total_metrics_.mae += metrics.mae;
    total_metrics_.r2 += metrics.r2;
    total_metrics_.mape += metrics.mape;
    total_metrics_.directional_accuracy += metrics.directional_accuracy;
    count_++;
}

void MetricsTracker::update_channel_metrics(
    const std::unordered_map<std::string, float>& channel_metrics
) {
    for (const auto& [key, value] : channel_metrics) {
        total_channel_metrics_[key] += value;
    }
}

MetricsCalculator::Metrics MetricsTracker::get_average_metrics() const {
    if (count_ == 0) return MetricsCalculator::Metrics();
    
    MetricsCalculator::Metrics avg = total_metrics_;
    avg.mse /= count_;
    avg.rmse /= count_;
    avg.mae /= count_;
    avg.r2 /= count_;
    avg.mape /= count_;
    avg.directional_accuracy /= count_;
    
    return avg;
}

std::unordered_map<std::string, float> MetricsTracker::get_average_channel_metrics() const {
    std::unordered_map<std::string, float> avg_metrics;
    
    if (count_ == 0) return avg_metrics;
    
    for (const auto& [key, value] : total_channel_metrics_) {
        avg_metrics[key] = value / count_;
    }
    
    return avg_metrics;
}

void MetricsTracker::reset() {
    count_ = 0;
    total_metrics_ = MetricsCalculator::Metrics();
    total_channel_metrics_.clear();
}

void MetricsTracker::save_to_file(const std::string& filepath) const {
    Json::Value root;
    
    // Save average metrics
    auto avg_metrics = get_average_metrics();
    root["metrics"]["mse"] = avg_metrics.mse;
    root["metrics"]["rmse"] = avg_metrics.rmse;
    root["metrics"]["mae"] = avg_metrics.mae;
    root["metrics"]["r2"] = avg_metrics.r2;
    root["metrics"]["mape"] = avg_metrics.mape;
    root["metrics"]["directional_accuracy"] = avg_metrics.directional_accuracy;
    
    // Save channel metrics
    auto avg_channel_metrics = get_average_channel_metrics();
    for (const auto& [key, value] : avg_channel_metrics) {
        root["channel_metrics"][key] = value;
    }
    
    // Write to file
    std::ofstream file(filepath);
    Json::StyledWriter writer;
    file << writer.write(root);
} 