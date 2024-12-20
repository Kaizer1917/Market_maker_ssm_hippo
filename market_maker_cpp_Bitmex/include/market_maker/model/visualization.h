#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include "metrics.h"

class ModelVisualizer {
public:
    struct VisualizationConfig {
        std::string output_dir{"./visualizations"};
        bool save_plots{true};
        bool show_plots{false};
        std::string style{"seaborn"};
        int fig_width{12};
        int fig_height{8};
        float dpi{100.0f};
    };

    explicit ModelVisualizer(const VisualizationConfig& config = VisualizationConfig());

    void plot_predictions(
        const torch::Tensor& predictions,
        const torch::Tensor& targets,
        const std::string& title = "Model Predictions",
        int channel_idx = 0
    );

    void plot_loss_curve(
        const std::vector<float>& train_losses,
        const std::vector<float>& val_losses,
        const std::string& title = "Training Loss"
    );

    void plot_metrics_evolution(
        const std::vector<MetricsCalculator::Metrics>& metrics_history,
        const std::string& title = "Metrics Evolution"
    );

    void plot_attention_weights(
        const torch::Tensor& attention_weights,
        const std::string& title = "Attention Weights"
    );

    void save_metrics_summary(
        const MetricsCalculator::Metrics& metrics,
        const std::string& filename = "metrics_summary.json"
    );

private:
    VisualizationConfig config_;
    void initialize_matplotlib();
    void create_output_directory();
    std::string get_timestamp();
}; 