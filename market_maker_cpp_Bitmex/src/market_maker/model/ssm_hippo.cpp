#include "ssm_hippo.h"
#include "mamba_block.h"
#include "channel_attention.h"

SSMHippo::SSMHippo(const ModelArgs& args) : args_(args) {
    // Input projection with skip connection
    input_proj_ = register_module("input_proj", 
        torch::nn::Sequential(
            torch::nn::Linear(args.num_channels, args.d_model),
            torch::nn::LayerNorm(args.d_model),
            torch::nn::GELU()
        ));
    
    input_skip_ = register_module("input_skip",
        torch::nn::Linear(args.num_channels, args.d_model));

    // Build patch embedding layers
    for (int i = 0; i < args.n_layer; ++i) {
        int in_dim = args.d_model * (1 + (i-1)/2);
        int out_dim = args.d_model * (1 + i/2);
        
        auto patch_layer = torch::nn::Sequential(
            torch::nn::Linear(in_dim, out_dim),
            torch::nn::LayerNorm(out_dim),
            torch::nn::GELU()
        );
        
        patch_embed_.push_back(patch_layer);
        register_module("patch_embed_" + std::to_string(i), patch_layer);
    }

    // Build SSM blocks
    for (int i = 0; i < args.n_layer; ++i) {
        ModelArgs block_args = args;
        block_args.d_model *= (1 + i/2);
        block_args.d_state *= (1 + i/2);
        
        auto block = MambaBlock(block_args);
        ssm_blocks_.push_back(block);
        register_module("ssm_block_" + std::to_string(i), block);
    }

    // Final normalization and projection
    int final_dim = args.d_model * (1 + (args.n_layer-1)/2);
    norm_f_ = register_module("norm_f", torch::nn::LayerNorm(final_dim));
    
    output_proj_ = register_module("output_proj",
        torch::nn::Linear(final_dim * args.num_patches, 
                         args.num_channels * args.forecast_len));
}

torch::Tensor SSMHippo::forward(torch::Tensor x, double training_progress) {
    // Input projection with skip connection
    auto x_proj = x.transpose(1, 2);
    auto proj = input_proj_->forward(x_proj);
    proj = proj + input_skip_->forward(x_proj);
    x = proj.transpose(1, 2);

    // Create patches
    std::vector<torch::Tensor> patches;
    for (int i = 0; i < args_.seq_len - args_.patch_len + 1; i += args_.stride) {
        auto patch = x.slice(2, i, i + args_.patch_len).mean(2);
        patches.push_back(patch);
    }
    
    x = torch::stack(patches, 1);

    // Forward through layers
    for (int i = 0; i < args_.n_layer; ++i) {
        auto x_res = x;
        x = patch_embed_[i]->forward(x);
        if (i > 0) {
            x = x + x_res;
        }
        x = x + ssm_blocks_[i]->forward(x, training_progress);
    }

    x = norm_f_->forward(x);
    x = x.reshape({x.size(0), -1});
    x = output_proj_->forward(x);
    
    return x.reshape({x.size(0), args_.num_channels, -1});
} 