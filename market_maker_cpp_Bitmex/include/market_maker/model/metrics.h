#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <unordered_map>

class MetricsCalculator {
public:
    struct Metrics {
        float mse{0.0f};
        float rmse{0.0f};
        float mae{0.0f};
        float r2{0.0f};
        float mape{0.0f};
        float directional_accuracy{0.0f};
    };

    static Metrics calculate_metrics(
        const torch::Tensor& predictions,
        const torch::Tensor& targets,
        bool reduce = true
    );
    
    static std::unordered_map<std::string, float> calculate_channel_metrics(
        const torch::Tensor& predictions,
        const torch::Tensor& targets
    );
    
    static torch::Tensor calculate_rolling_metrics(
        const torch::Tensor& predictions,
        const torch::Tensor& targets,
        int window_size
    );

private:
    static float calculate_r2_score(
        const torch::Tensor& predictions,
        const torch::Tensor& targets
    );
    
    static float calculate_directional_accuracy(
        const torch::Tensor& predictions,
        const torch::Tensor& targets
    );
};

class MetricsTracker {
public:
    MetricsTracker();
    
    void update(const MetricsCalculator::Metrics& metrics);
    void update_channel_metrics(
        const std::unordered_map<std::string, float>& channel_metrics
    );
    
    MetricsCalculator::Metrics get_average_metrics() const;
    std::unordered_map<std::string, float> get_average_channel_metrics() const;
    
    void reset();
    void save_to_file(const std::string& filepath) const;
    
private:
    int count_{0};
    MetricsCalculator::Metrics total_metrics_;
    std::unordered_map<std::string, float> total_channel_metrics_;
}; 