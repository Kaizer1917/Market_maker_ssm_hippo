#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <variant>
#include <chrono>

namespace market_maker {
namespace types {

// Time-related types
using TimePoint = std::chrono::system_clock::time_point;
using Duration = std::chrono::milliseconds;

// Data types
using FloatTensor = torch::Tensor;
using LongTensor = torch::Tensor;
using BatchTensor = std::vector<torch::Tensor>;

// Model parameter types
struct ModelState {
    std::unordered_map<std::string, torch::Tensor> params;
    std::unordered_map<std::string, torch::Tensor> buffers;
};

// Training types
struct TrainingState {
    int epoch;
    int iteration;
    float learning_rate;
    float best_loss;
    TimePoint last_update;
};

// Market data types
struct MarketData {
    TimePoint timestamp;
    float price;
    float volume;
    std::vector<float> features;
};

using MarketDataBatch = std::vector<MarketData>;

// Error handling
enum class ErrorCode {
    SUCCESS = 0,
    INVALID_INPUT,
    MODEL_ERROR,
    CUDA_ERROR,
    TVM_ERROR,
    IO_ERROR,
    UNKNOWN_ERROR
};

class ModelException : public std::exception {
public:
    ModelException(ErrorCode code, const std::string& message)
        : code_(code), message_(message) {}
    
    ErrorCode code() const { return code_; }
    const char* what() const noexcept override { return message_.c_str(); }

private:
    ErrorCode code_;
    std::string message_;
};

// Configuration types
using ConfigValue = std::variant<
    bool,
    int,
    float,
    std::string,
    std::vector<int>,
    std::vector<float>,
    std::vector<std::string>
>;

struct ConfigNode {
    std::unordered_map<std::string, ConfigValue> values;
    std::unordered_map<std::string, ConfigNode> children;
};

// Callback types
using TrainingCallback = std::function<void(const TrainingState&)>;
using PredictionCallback = std::function<void(const torch::Tensor&)>;

// Memory management
template<typename T>
using DevicePtr = std::unique_ptr<T, std::function<void(T*)>>;

template<typename T>
struct DeviceArray {
    DevicePtr<T> data;
    size_t size;
    
    DeviceArray(size_t n) : size(n) {
        data = DevicePtr<T>(
            new T[n],
            [](T* ptr) { delete[] ptr; }
        );
    }
};

} // namespace types
} // namespace market_maker 