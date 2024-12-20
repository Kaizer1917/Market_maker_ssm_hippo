#include "mamba_tvm_memory.h"

tvm::runtime::NDArray TVMMemoryManager::allocate_tvm_array(
    const std::vector<int64_t>& shape,
    tvm::DataType dtype,
    tvm::Device dev
) {
    std::string pool_key = get_pool_key(shape, dtype, dev);
    
    std::lock_guard<std::mutex> lock(global_mutex_);
    auto& pool = memory_pools_[pool_key];
    
    std::lock_guard<std::mutex> pool_lock(pool.mutex);
    if (!pool.free_arrays.empty()) {
        auto arr = std::move(pool.free_arrays.back());
        pool.free_arrays.pop_back();
        return *arr;
    }
    
    return tvm::runtime::NDArray::Empty(shape, dtype, dev);
}

torch::Tensor TVMMemoryManager::tvm_to_torch(const tvm::runtime::NDArray& arr) {
    auto shape = arr.Shape();
    auto dtype = arr.DataType();
    
    auto options = torch::TensorOptions()
        .dtype(convert_tvm_dtype_to_torch(dtype))
        .device(convert_tvm_device_to_torch(arr.device()));
    
    auto tensor = torch::empty(shape, options);
    arr.CopyToBytes(tensor.data_ptr(), tensor.numel() * tensor.element_size());
    
    return tensor;
}

tvm::runtime::NDArray TVMMemoryManager::torch_to_tvm(const torch::Tensor& tensor) {
    auto shape = tensor.sizes().vec();
    auto dtype = convert_torch_dtype_to_tvm(tensor.dtype());
    auto device = convert_torch_device_to_tvm(tensor.device());
    
    auto arr = allocate_tvm_array(shape, dtype, device);
    arr.CopyFromBytes(tensor.data_ptr(), tensor.numel() * tensor.element_size());
    
    return arr;
}

void TVMMemoryManager::clear_cache() {
    std::lock_guard<std::mutex> lock(global_mutex_);
    memory_pools_.clear();
}

std::string TVMMemoryManager::get_pool_key(
    const std::vector<int64_t>& shape,
    tvm::DataType dtype,
    tvm::Device dev
) {
    std::stringstream ss;
    for (auto dim : shape) ss << dim << "_";
    ss << dtype.code() << "_" << dtype.bits() << "_";
    ss << dev.device_type << "_" << dev.device_id;
    return ss.str();
} 