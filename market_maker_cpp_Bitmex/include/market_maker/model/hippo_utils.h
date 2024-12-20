#pragma once

#include <torch/torch.h>
#include <string>
#include <tuple>
#include <complex>

namespace hippo {

std::tuple<torch::Tensor, torch::Tensor> transition(
    const std::string& measure,
    int N,
    torch::Device device = torch::kCPU
);

torch::Tensor rank_correction(
    const std::string& measure,
    int N,
    int rank = 1,
    torch::ScalarType dtype = torch::kFloat32
);

torch::Tensor initial_C(
    const std::string& measure,
    int N,
    torch::ScalarType dtype = torch::kFloat32
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> nplr(
    const std::string& measure,
    int N,
    int rank = 1,
    torch::ScalarType dtype = torch::kFloat32,
    bool diagonalize_precision = true,
    double B_clip = 2.0
);

torch::Tensor optimize_hippo_transition(
    const std::string& measure,
    int N,
    double training_progress,
    torch::Device device
);

} // namespace hippo 