#include "hippo_utils.h"
#include <cmath>
#include <vector>

namespace hippo {

std::tuple<torch::Tensor, torch::Tensor> transition(
    const std::string& measure,
    int N,
    torch::Device device
) {
    torch::Tensor A, B;

    if (measure == "legs") {
        // Legendre (scaled) implementation
        auto q = torch::arange(N, torch::kFloat32);
        auto r = 2 * q + 1;
        
        // Create matrices
        auto M = torch::zeros({N, N});
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i >= j) {
                    M[i][j] = -r[j];
                }
            }
        }
        M = M - torch::diag(q);

        // Transform matrices
        auto T = torch::sqrt(torch::diag(2 * q + 1));
        A = T.mm(M).mm(torch::inverse(T));
        B = torch::diag(T);
        B = B.unsqueeze(-1);

    } else if (measure == "legt") {
        // Legendre (translated) implementation
        auto Q = torch::arange(N, torch::kFloat32);
        auto R = torch::sqrt(2 * Q + 1);
        
        A = torch::zeros({N, N});
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = R[i] * (i < j ? -1 : 1) * R[j];
            }
        }
        A = -A * 0.5;
        B = R.unsqueeze(-1) * 0.5;

    } else {
        throw std::runtime_error("Unsupported measure: " + measure);
    }

    return {A.to(device), B.to(device)};
}

torch::Tensor optimize_hippo_transition(
    const std::string& measure,
    int N,
    double training_progress,
    torch::Device device
) {
    auto [A, B] = transition(measure, N, device);
    
    // Dynamic scaling based on training progress
    auto progress_factor = torch::sigmoid(torch::tensor(training_progress * 10 - 5));
    
    if (measure == "legs") {
        // Optimize Legendre-based transitions
        auto scale = torch::exp(-progress_factor * torch::arange(N, device) / N);
        A = A * scale.unsqueeze(0);
        
    } else if (measure == "legt") {
        // Optimize translated Legendre
        auto freq_scale = 1.0 + progress_factor * (torch::arange(N, device) / N);
        A = A * freq_scale.unsqueeze(0);
    }
    
    // Add stability regularization
    A = A - torch::eye(N, device) * 0.1 * (1 - progress_factor);
    
    return A;
}

torch::Tensor rank_correction(
    const std::string& measure,
    int N,
    int rank,
    torch::ScalarType dtype
) {
    torch::Tensor P;

    if (measure == "legs") {
        if (rank < 1) throw std::runtime_error("Rank must be >= 1 for legs measure");
        P = torch::sqrt(0.5 + torch::arange(N, dtype)).unsqueeze(0);
        
    } else if (measure == "legt") {
        if (rank < 2) throw std::runtime_error("Rank must be >= 2 for legt measure");
        auto base = torch::sqrt(1 + 2 * torch::arange(N, dtype));
        auto P0 = base.clone();
        auto P1 = base.clone();
        P0.index_put_({torch::arange(0, N, 2)}, 0);
        P1.index_put_({torch::arange(1, N, 2)}, 0);
        P = torch::stack({P0, P1}) * std::sqrt(0.5);
        
    } else {
        P = torch::zeros({1, N}, dtype);
    }

    if (rank > P.size(0)) {
        P = torch::cat({P, torch::zeros({rank - P.size(0), N}, dtype)});
    }
    
    return P;
}

} // namespace hippo 