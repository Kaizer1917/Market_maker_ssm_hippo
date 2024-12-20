#include "Rollercoaster_girls.h"
#include <execution>
#include <future>

MarketPredictor::MarketPredictor(Config config) : config_(config) {
    // Initialize SSM-HIPPO model
    SSMHippoConfig model_config{
        .d_model = config.model_args.d_model,
        .n_layer = config.model_args.n_layer,
        .seq_len = config.model_args.seq_len,
        .d_state = config.model_args.d_state,
        .num_channels = config.model_args.num_channels,
        .forecast_len = config.model_args.forecast_len,
        .patch_len = config.model_args.patch_len,
        .stride = config.model_args.stride,
        .sigma = config.model_args.sigma,
        .reduction_ratio = config.model_args.reduction_ratio
    };
    
    model_ = std::make_unique<SSMHippo>(model_config);
    loss_fn_ = std::make_unique<AdaptiveTemporalCoherenceLoss>();
    
    // Move model to device
    device_ = torch::Device(config.inference.use_cuda ? torch::kCUDA : torch::kCPU);
    model_->to(device_);
}

torch::Tensor MarketPredictor::predict(const stable_vector<float>& features) {
    torch::NoGradGuard no_grad;
    model_->eval();
    
    auto input_tensor = preprocess_features(features);
    return model_->forward(input_tensor);
}

torch::Tensor MarketPredictor::preprocess_features(const stable_vector<float>& features) {
    // Convert to tensor and reshape
    auto tensor = torch::from_blob(
        const_cast<float*>(features.data()),
        {1, config_.model_args.num_channels, -1},
        torch::kFloat32
    ).to(device_);
    
    // Apply normalization if needed
    if (config_.preprocessing.normalize) {
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8);
    }
    
    return tensor;
}
