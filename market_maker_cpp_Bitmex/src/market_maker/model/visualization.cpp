#include "visualization.h"
#include <matplotlibcpp.h>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <json/json.h>

namespace plt = matplotlibcpp;

ModelVisualizer::ModelVisualizer(const VisualizationConfig& config) 
    : config_(config) {
    initialize_matplotlib();
    create_output_directory();
}

void ModelVisualizer::plot_predictions(
    const torch::Tensor& predictions,
    const torch::Tensor& targets,
    const std::string& title,
    int channel_idx
) {
    plt::figure(figsize(config_.fig_width, config_.fig_height));
    
    // Convert tensors to vectors
    std::vector<float> pred_vec(predictions.select(-1, channel_idx).data_ptr<float>(),
                               predictions.select(-1, channel_idx).data_ptr<float>() + 
                               predictions.size(1));
    
    std::vector<float> target_vec(targets.select(-1, channel_idx).data_ptr<float>(),
                                 targets.select(-1, channel_idx).data_ptr<float>() + 
                                 targets.size(1));
    
    // Create time axis
    std::vector<float> time(pred_vec.size());
    std::iota(time.begin(), time.end(), 0);
    
    // Plot predictions and targets
    plt::plot(time, pred_vec, "b-", {{"label", "Predictions"}});
    plt::plot(time, target_vec, "r--", {{"label", "Targets"}});
    
    plt::title(title);
    plt::xlabel("Time Step");
    plt::ylabel("Value");
    plt::legend();
    plt::grid(true);
    
    if (config_.save_plots) {
        std::string filename = config_.output_dir + "/predictions_" + 
                             get_timestamp() + ".png";
        plt::savefig(filename, config_.dpi);
    }
    
    if (config_.show_plots) {
        plt::show();
    }
    plt::close();
}

void ModelVisualizer::plot_loss_curve(
    const std::vector<float>& train_losses,
    const std::vector<float>& val_losses,
    const std::string& title
) {
    plt::figure(figsize(config_.fig_width, config_.fig_height));
    
    std::vector<float> epochs(train_losses.size());
    std::iota(epochs.begin(), epochs.end(), 1);
    
    plt::plot(epochs, train_losses, "b-", {{"label", "Training Loss"}});
    plt::plot(epochs, val_losses, "r-", {{"label", "Validation Loss"}});
    
    plt::title(title);
    plt::xlabel("Epoch");
    plt::ylabel("Loss");
    plt::legend();
    plt::grid(true);
    plt::yscale("log");
    
    if (config_.save_plots) {
        std::string filename = config_.output_dir + "/loss_curve_" + 
                             get_timestamp() + ".png";
        plt::savefig(filename, config_.dpi);
    }
    
    if (config_.show_plots) {
        plt::show();
    }
    plt::close();
}

void ModelVisualizer::plot_metrics_evolution(
    const std::vector<MetricsCalculator::Metrics>& metrics_history,
    const std::string& title
) {
    plt::figure(figsize(config_.fig_width, config_.fig_height));
    
    std::vector<float> epochs(metrics_history.size());
    std::iota(epochs.begin(), epochs.end(), 1);
    
    std::vector<float> mse_values, r2_values, da_values;
    for (const auto& metrics : metrics_history) {
        mse_values.push_back(metrics.mse);
        r2_values.push_back(metrics.r2);
        da_values.push_back(metrics.directional_accuracy);
    }
    
    plt::subplot(3, 1, 1);
    plt::plot(epochs, mse_values, "b-", {{"label", "MSE"}});
    plt::title(title);
    plt::legend();
    plt::grid(true);
    
    plt::subplot(3, 1, 2);
    plt::plot(epochs, r2_values, "g-", {{"label", "R²"}});
    plt::legend();
    plt::grid(true);
    
    plt::subplot(3, 1, 3);
    plt::plot(epochs, da_values, "r-", {{"label", "Directional Accuracy"}});
    plt::xlabel("Epoch");
    plt::legend();
    plt::grid(true);
    
    if (config_.save_plots) {
        std::string filename = config_.output_dir + "/metrics_evolution_" + 
                             get_timestamp() + ".png";
        plt::savefig(filename, config_.dpi);
    }
    
    if (config_.show_plots) {
        plt::show();
    }
    plt::close();
}

void ModelVisualizer::plot_attention_weights(
    const torch::Tensor& attention_weights,
    const std::string& title
) {
    plt::figure(figsize(config_.fig_width, config_.fig_height));
    
    // Convert attention weights to vector of vectors for heatmap
    auto weights = attention_weights.cpu().detach();
    std::vector<std::vector<float>> heatmap_data;
    
    for (int i = 0; i < weights.size(0); ++i) {
        std::vector<float> row(weights[i].data_ptr<float>(),
                             weights[i].data_ptr<float>() + weights.size(1));
        heatmap_data.push_back(row);
    }
    
    plt::imshow(heatmap_data, {{"cmap", "viridis"}, {"aspect", "auto"}});
    plt::colorbar();
    plt::title(title);
    plt::xlabel("Key");
    plt::ylabel("Query");
    
    if (config_.save_plots) {
        std::string filename = config_.output_dir + "/attention_weights_" + 
                             get_timestamp() + ".png";
        plt::savefig(filename, config_.dpi);
    }
    
    if (config_.show_plots) {
        plt::show();
    }
    plt::close();
}

void ModelVisualizer::save_metrics_summary(
    const MetricsCalculator::Metrics& metrics,
    const std::string& filename
) {
    Json::Value root;
    root["mse"] = metrics.mse;
    root["rmse"] = metrics.rmse;
    root["mae"] = metrics.mae;
    root["r2"] = metrics.r2;
    root["mape"] = metrics.mape;
    root["directional_accuracy"] = metrics.directional_accuracy;
    
    std::string filepath = config_.output_dir + "/" + filename;
    std::ofstream file(filepath);
    Json::StyledWriter writer;
    file << writer.write(root);
}

void ModelVisualizer::initialize_matplotlib() {
    plt::backend("Agg");
    plt::style(config_.style);
}

void ModelVisualizer::create_output_directory() {
    std::filesystem::create_directories(config_.output_dir);
}

std::string ModelVisualizer::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    return ss.str();
} 