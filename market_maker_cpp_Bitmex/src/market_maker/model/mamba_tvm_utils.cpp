#include "mamba_tvm_utils.h"
#include <tvm/runtime/registry.h>
#include <tvm/tir/transform.h>

namespace tvm_utils {

tvm::Target TargetDevice::get_optimal_target() {
    if (has_cuda()) {
        return tvm::Target("cuda");
    } else if (has_rocm()) {
        return tvm::Target("rocm");
    } else {
        return tvm::Target("llvm -mcpu=native");
    }
}

bool TargetDevice::has_gpu() {
    return has_cuda() || has_rocm();
}

bool TargetDevice::has_cuda() {
    return tvm::runtime::Registry::Get("device_api.cuda") != nullptr;
}

bool TargetDevice::has_rocm() {
    return tvm::runtime::Registry::Get("device_api.rocm") != nullptr;
}

std::string TargetDevice::get_device_name() {
    if (has_cuda()) return "CUDA";
    if (has_rocm()) return "ROCm";
    return "CPU";
}

tvm::runtime::Module OptimizationUtils::optimize_for_hardware(
    tvm::runtime::Module mod,
    const tvm::Target& target
) {
    using namespace tvm::tir::transform;

    auto pass_ctx = tvm::transform::PassContext::Create();
    pass_ctx->config.Set("tir.disable_vectorize", false);
    
    with(pass_ctx, [&]() {
        if (target->kind->name == "cuda") {
            // CUDA-specific optimizations
            mod = InjectPTXIntrinsics()(mod);
            mod = LowerWarpMemory()(mod);
            mod = InjectSoftwarePipeline()(mod);
        } else if (target->kind->name == "llvm") {
            // CPU-specific optimizations
            mod = VectorizeLoop()(mod);
            mod = LoopPartition()(mod);
            mod = UnrollLoop()(mod);
        }
    });

    return mod;
}

tvm::runtime::Module OptimizationUtils::apply_memory_optimizations(
    tvm::runtime::Module mod,
    const tvm::Target& target
) {
    using namespace tvm::tir::transform;

    auto pass_ctx = tvm::transform::PassContext::Create();
    
    with(pass_ctx, [&]() {
        // Common memory optimizations
        mod = StorageRewrite()(mod);
        mod = InferFragment()(mod);
        
        if (target->kind->name == "cuda") {
            // GPU memory optimizations
            mod = LowerThreadAllocation()(mod);
            mod = LowerCrossThreadReduction()(mod);
        }
    });

    return mod;
}

tvm::runtime::Module OptimizationUtils::vectorize_operations(
    tvm::runtime::Module mod,
    const tvm::Target& target
) {
    using namespace tvm::tir::transform;

    auto pass_ctx = tvm::transform::PassContext::Create();
    
    with(pass_ctx, [&]() {
        // Vectorization passes
        mod = VectorizeLoop()(mod);
        mod = InjectVirtualThread()(mod);
        
        if (target->kind->name == "llvm") {
            // CPU-specific vectorization
            mod = LoopPartition()(mod);
            mod = VerifyMemory()(mod);
        }
    });

    return mod;
}

} // namespace tvm_utils 