#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <chrono>
#include <string_view>
#include "stable_vector.h"
#include <torch/torch.h>
#include "model/losses.h"
#include "model/ssm_layer.h"

// Performance monitoring
class PerformanceProfiler {
public:
    struct Metrics {
        std::chrono::high_resolution_clock::time_point start_time;
        double total_time_ms{0};
        size_t calls{0};
        size_t peak_memory{0};
        std::string name;
    };

    static PerformanceProfiler& instance() {
        static PerformanceProfiler instance;
        return instance;
    }

    void start_operation(std::string_view name);
    void end_operation(std::string_view name);
    void report() const;

private:
    std::unordered_map<std::string, Metrics> metrics_;
    std::mutex metrics_mutex_;
};

// Memory pool for efficient allocation
template<typename T>
class MemoryPool {
public:
    static constexpr size_t POOL_SIZE = 1024 * 1024;  // 1MB chunks
    
    T* allocate(size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_blocks_.empty() || free_blocks_.back().size < n) {
            allocate_new_block(n);
        }
        return get_from_pool(n);
    }

    void deallocate(T* ptr, size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        free_blocks_.push_back({ptr, n});
        consolidate_blocks();
    }

private:
    struct Block {
        T* ptr;
        size_t size;
    };
    
    std::vector<Block> free_blocks_;
    std::mutex mutex_;
    
    void allocate_new_block(size_t n);
    T* get_from_pool(size_t n);
    void consolidate_blocks();
};

// Enhanced error handling
class MarketPredictorError : public std::runtime_error {
public:
    enum class ErrorCode {
        CUDA_ERROR,
        MEMORY_ERROR,
        MODEL_ERROR,
        DATA_ERROR,
        TRAINING_ERROR
    };

    MarketPredictorError(ErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    ErrorCode code() const { return code_; }

private:
    ErrorCode code_;
};

// Main class with enhanced functionality
class MarketPredictor {
public:
    // Configuration struct
    struct Config {
        ModelArgs model_args;
        struct Training {
            int batch_size{32};
            float learning_rate{1e-3f};
            float lr_min{1e-6f};
            int patience{5};
            float early_stopping_delta{1e-4f};
            int early_stopping_patience{10};
            bool use_mixed_precision{true};
            bool use_gradient_clipping{true};
            float max_grad_norm{1.0f};
        } training;
        
        struct Inference {
            bool use_tensorrt{false};
            bool use_dynamic_batching{true};
            int inference_batch_size{64};
        } inference;
    };

    explicit MarketPredictor(Config config);
    ~MarketPredictor();

    // ... previous declarations ...

private:
    // Memory management
    MemoryPool<float> memory_pool_;
    
    // Performance monitoring
    std::unique_ptr<torch::autograd::ProfilerState> profiler_state_;
    
    // Enhanced model components
    struct ModelState {
        torch::jit::script::Module traced_model;
        torch::Device device;
        bool is_quantized{false};
        bool using_tensorrt{false};
    };
    
    std::unique_ptr<ModelState> model_state_;
    
    // Methods for advanced functionality
    void setup_mixed_precision_training();
    void optimize_model_for_inference();
    void handle_cuda_error(const char* file, int line);
    void profile_execution(const char* operation_name);
    
    // Thread pool for parallel processing
    class ThreadPool {
    public:
        explicit ThreadPool(size_t num_threads);
        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args);
    private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        std::mutex queue_mutex_;
        std::condition_variable condition_;
        bool stop_{false};
    };
    
    std::unique_ptr<ThreadPool> thread_pool_;
    
    // Add new members
    std::unique_ptr<AdaptiveTemporalCoherenceLoss> loss_fn_;
    double training_progress_{1.0};
    
    // Add helper method
    torch::Tensor preprocess_features(const stable_vector<float>& features);
}; 