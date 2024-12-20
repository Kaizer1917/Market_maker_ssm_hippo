#pragma once

#include <tvm/runtime/module.h>
#include <tvm/target/target.h>
#include <string>

namespace tvm_utils {

class TargetDevice {
public:
    static tvm::Target get_optimal_target();
    static bool has_gpu();
    static bool has_cuda();
    static bool has_rocm();
    static std::string get_device_name();
};

class OptimizationUtils {
public:
    static tvm::runtime::Module optimize_for_hardware(
        tvm::runtime::Module mod,
        const tvm::Target& target
    );
    
    static tvm::runtime::Module apply_memory_optimizations(
        tvm::runtime::Module mod,
        const tvm::Target& target
    );
    
    static tvm::runtime::Module vectorize_operations(
        tvm::runtime::Module mod,
        const tvm::Target& target
    );
};

} // namespace tvm_utils 