#pragma once

#include <tvm/runtime/ndarray.h>
#include <torch/torch.h>
#include <memory>

class TVMMemoryManager {
public:
    static TVMMemoryManager& getInstance() {
        static TVMMemoryManager instance;
        return instance;
    }

    tvm::runtime::NDArray allocate_tvm_array(
        const std::vector<int64_t>& shape,
        tvm::DataType dtype,
        tvm::Device dev
    );

    torch::Tensor tvm_to_torch(const tvm::runtime::NDArray& arr);
    tvm::runtime::NDArray torch_to_tvm(const torch::Tensor& tensor);

    void clear_cache();

private:
    TVMMemoryManager() = default;
    ~TVMMemoryManager() = default;
    TVMMemoryManager(const TVMMemoryManager&) = delete;
    TVMMemoryManager& operator=(const TVMMemoryManager&) = delete;

    struct MemoryPool {
        std::vector<std::unique_ptr<tvm::runtime::NDArray>> free_arrays;
        std::mutex mutex;
    };

    std::unordered_map<std::string, MemoryPool> memory_pools_;
    std::mutex global_mutex_;

    std::string get_pool_key(
        const std::vector<int64_t>& shape,
        tvm::DataType dtype,
        tvm::Device dev
    );
}; 