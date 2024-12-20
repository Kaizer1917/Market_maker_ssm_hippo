#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <json/json.h>

namespace market_maker {
namespace config {

struct GlobalConfig {
    // Environment settings
    bool use_gpu{true};
    int gpu_device_id{0};
    int num_threads{4};
    std::string log_level{"INFO"};
    
    // Model paths
    std::string model_dir{"models"};
    std::string checkpoint_dir{"checkpoints"};
    std::string data_dir{"data"};
    
    // Training settings
    bool enable_wandb{false};
    std::string wandb_project{"market_maker"};
    std::string experiment_name{"default"};
    
    // TVM settings
    bool use_tvm_optimization{true};
    std::string tvm_target{"llvm"};
    int tvm_opt_level{3};
    
    // Memory settings
    size_t max_memory_mb{8192};
    bool enable_memory_pool{true};
    
    static GlobalConfig& getInstance() {
        static GlobalConfig instance;
        return instance;
    }
    
    void load_from_file(const std::string& config_path);
    void save_to_file(const std::string& config_path) const;
    
private:
    GlobalConfig() = default;
};

// Environment variables
const char* const ENV_GPU_DEVICE = "MARKET_MAKER_GPU_DEVICE";
const char* const ENV_NUM_THREADS = "MARKET_MAKER_NUM_THREADS";
const char* const ENV_LOG_LEVEL = "MARKET_MAKER_LOG_LEVEL";

// Configuration paths
const char* const DEFAULT_CONFIG_PATH = "config/market_maker.json";
const char* const MODEL_CONFIG_PATH = "config/model.json";

} // namespace config
} // namespace market_maker 