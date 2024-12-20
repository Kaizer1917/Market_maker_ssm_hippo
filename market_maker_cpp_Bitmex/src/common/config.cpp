#include "market_maker/common/config.h"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <filesystem>

namespace market_maker {
namespace config {

void GlobalConfig::load_from_file(const std::string& config_path) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_path);
    }

    Json::Value root;
    Json::Reader reader;
    if (!reader.parse(config_file, root)) {
        throw std::runtime_error("Failed to parse config file: " + 
                               reader.getFormattedErrorMessages());
    }

    // Environment settings
    if (root.isMember("environment")) {
        const auto& env = root["environment"];
        use_gpu = env.get("use_gpu", use_gpu).asBool();
        gpu_device_id = env.get("gpu_device_id", gpu_device_id).asInt();
        num_threads = env.get("num_threads", num_threads).asInt();
        log_level = env.get("log_level", log_level).asString();
    }

    // Check environment variables
    if (const char* env_gpu = std::getenv(ENV_GPU_DEVICE)) {
        gpu_device_id = std::stoi(env_gpu);
    }
    if (const char* env_threads = std::getenv(ENV_NUM_THREADS)) {
        num_threads = std::stoi(env_threads);
    }
    if (const char* env_log = std::getenv(ENV_LOG_LEVEL)) {
        log_level = env_log;
    }

    // Model paths
    if (root.isMember("paths")) {
        const auto& paths = root["paths"];
        model_dir = paths.get("model_dir", model_dir).asString();
        checkpoint_dir = paths.get("checkpoint_dir", checkpoint_dir).asString();
        data_dir = paths.get("data_dir", data_dir).asString();

        // Create directories if they don't exist
        std::filesystem::create_directories(model_dir);
        std::filesystem::create_directories(checkpoint_dir);
        std::filesystem::create_directories(data_dir);
    }

    // Training settings
    if (root.isMember("training")) {
        const auto& training = root["training"];
        enable_wandb = training.get("enable_wandb", enable_wandb).asBool();
        wandb_project = training.get("wandb_project", wandb_project).asString();
        experiment_name = training.get("experiment_name", experiment_name).asString();
    }

    // TVM settings
    if (root.isMember("tvm")) {
        const auto& tvm = root["tvm"];
        use_tvm_optimization = tvm.get("use_optimization", use_tvm_optimization).asBool();
        tvm_target = tvm.get("target", tvm_target).asString();
        tvm_opt_level = tvm.get("opt_level", tvm_opt_level).asInt();
    }

    // Memory settings
    if (root.isMember("memory")) {
        const auto& memory = root["memory"];
        max_memory_mb = memory.get("max_memory_mb", max_memory_mb).asUInt64();
        enable_memory_pool = memory.get("enable_memory_pool", enable_memory_pool).asBool();
    }

    // Validate configuration
    validate_config();
}

void GlobalConfig::save_to_file(const std::string& config_path) const {
    Json::Value root;

    // Environment settings
    root["environment"]["use_gpu"] = use_gpu;
    root["environment"]["gpu_device_id"] = gpu_device_id;
    root["environment"]["num_threads"] = num_threads;
    root["environment"]["log_level"] = log_level;

    // Model paths
    root["paths"]["model_dir"] = model_dir;
    root["paths"]["checkpoint_dir"] = checkpoint_dir;
    root["paths"]["data_dir"] = data_dir;

    // Training settings
    root["training"]["enable_wandb"] = enable_wandb;
    root["training"]["wandb_project"] = wandb_project;
    root["training"]["experiment_name"] = experiment_name;

    // TVM settings
    root["tvm"]["use_optimization"] = use_tvm_optimization;
    root["tvm"]["target"] = tvm_target;
    root["tvm"]["opt_level"] = tvm_opt_level;

    // Memory settings
    root["memory"]["max_memory_mb"] = Json::Value::UInt64(max_memory_mb);
    root["memory"]["enable_memory_pool"] = enable_memory_pool;

    // Write to file
    std::ofstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("Could not open config file for writing: " + config_path);
    }

    Json::StyledWriter writer;
    config_file << writer.write(root);
}

void GlobalConfig::validate_config() {
    // Validate GPU settings
    if (use_gpu && gpu_device_id < 0) {
        throw std::runtime_error("Invalid GPU device ID: " + std::to_string(gpu_device_id));
    }

    // Validate thread count
    if (num_threads <= 0) {
        throw std::runtime_error("Invalid thread count: " + std::to_string(num_threads));
    }

    // Validate memory settings
    if (max_memory_mb == 0) {
        throw std::runtime_error("Invalid maximum memory setting");
    }

    // Validate TVM settings
    if (tvm_opt_level < 0 || tvm_opt_level > 3) {
        throw std::runtime_error("Invalid TVM optimization level: " + 
                               std::to_string(tvm_opt_level));
    }

    // Validate paths
    if (model_dir.empty() || checkpoint_dir.empty() || data_dir.empty()) {
        throw std::runtime_error("Required directory paths cannot be empty");
    }
}

} // namespace config
} // namespace market_maker 