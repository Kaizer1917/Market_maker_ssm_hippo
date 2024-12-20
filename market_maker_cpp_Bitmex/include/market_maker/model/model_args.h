#pragma once

#include <string>
#include <memory>
#include <torch/torch.h>

struct ModelArgs {
    // Model architecture
    int d_model{128};
    int n_layer{4};
    int seq_len{96};
    int d_state{16};
    int expand{2};
    std::string dt_rank{"auto"};
    int d_conv{4};
    int pad_multiple{8};
    bool conv_bias{true};
    bool bias{false};
    
    // Data parameters
    int num_channels{24};
    int patch_len{16};
    int stride{8};
    int forecast_len{96};
    
    // Training parameters
    float sigma{0.5};
    int reduction_ratio{8};
    bool verbose{false};
    float learning_rate{0.001};
    int num_epochs{100};
    int batch_size{32};
    std::string optimizer_type{"adamw"};
    float weight_decay{0.01};
    
    // Derived parameters
    int d_inner;
    int dt_rank_val;
    int d_state_min;
    int d_state_max;
    float patch_overlap{0.5};
    float expand_factor{1.5};
    int max_expansion{3};
    int num_patches;

    ModelArgs() {
        initialize_derived_params();
    }

    void initialize_derived_params() {
        d_inner = expand * d_model;
        dt_rank_val = (dt_rank == "auto") ? ceil(d_model / 16.0) : std::stoi(dt_rank);
        d_state_min = d_state;
        d_state_max = d_state * 2;
        stride = std::max(1, static_cast<int>(patch_len * (1 - patch_overlap)));
        num_patches = (seq_len - patch_len) / stride + 1;
        
        if (forecast_len % pad_multiple != 0) {
            forecast_len += (pad_multiple - forecast_len % pad_multiple);
        }
    }
}; 