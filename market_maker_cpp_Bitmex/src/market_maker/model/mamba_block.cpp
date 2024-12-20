#include "mamba_block.h"
#include "hippo_utils.h"

MambaBlock::MambaBlock(const ModelArgs& args) : args_(args) {
    // Input projection
    in_proj_ = register_module("in_proj",
        torch::nn::Linear(args.d_model, args.d_inner * 2, args.bias));

    // Convolution layer
    conv1d_ = register_module("conv1d",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(
            args.d_inner, args.d_inner, args.d_conv)
            .groups(args.d_inner)
            .bias(args.conv_bias)
            .padding(args.d_conv - 1)));

    // Projections
    x_proj_ = register_module("x_proj",
        torch::nn::Linear(args.d_inner, args.dt_rank_val + args.d_state * 2, false));
    
    dt_proj_ = register_module("dt_proj",
        torch::nn::Linear(args.dt_rank_val, args.d_inner, true));

    // Initialize A_log and D parameters
    auto A = torch::arange(1, args.d_state + 1).repeat({args.d_inner, 1});
    A_log_ = register_parameter("A_log", torch::log(A));
    D_ = register_parameter("D", torch::ones(args.d_inner));

    // Output projection
    out_proj_ = register_module("out_proj",
        torch::nn::Linear(args.d_inner, args.d_model, args.bias));
}

torch::Tensor MambaBlock::forward(torch::Tensor x, double training_progress) {
    auto [b, l, d] = x.sizes();

    // Split into x and residual
    auto x_and_res = in_proj_->forward(x);
    auto chunks = x_and_res.chunk(2, -1);
    auto x_proj = chunks[0];
    auto res = chunks[1];

    // Apply convolution
    x_proj = x_proj.transpose(1, 2);
    x_proj = conv1d_->forward(x_proj);
    x_proj = x_proj.index({"...", torch::indexing::Slice(torch::none), torch::indexing::Slice(torch::none, l)});
    x_proj = x_proj.transpose(1, 2);

    // Apply SiLU activation
    x_proj = torch::silu(x_proj);

    // Get optimized HiPPO matrices
    auto [A, B] = get_hippo_matrices("legs", args_.d_state, x.device());
    A = optimize_hippo_transition("legs", args_.d_state, training_progress, x.device());

    // Apply SSM
    auto y = ssm(x_proj, A, B);

    // Multiply with activated residual
    y = y * torch::silu(res);

    // Final projection
    return out_proj_->forward(y);
}

torch::Tensor MambaBlock::ssm(const torch::Tensor& x, const torch::Tensor& A, const torch::Tensor& B) {
    auto x_dbl = x_proj_->forward(x);
    
    // Split into delta, B, and C
    auto delta = x_dbl.narrow(-1, 0, args_.dt_rank_val);
    auto B_proj = x_dbl.narrow(-1, args_.dt_rank_val, args_.d_state);
    auto C = x_dbl.narrow(-1, args_.dt_rank_val + args_.d_state, args_.d_state);

    // Process delta through projection and softplus
    delta = torch::softplus(dt_proj_->forward(delta));

    // Get discretized parameters
    auto A_disc = -torch::exp(A_log_.to(x.device()));

    return selective_scan(x, delta, A_disc, B_proj, C, D_);
}

torch::Tensor MambaBlock::selective_scan(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& D) {
    
    auto L = delta.size(1);
    auto N = A.size(-1);

    // Initialize state
    auto x = torch::zeros({u.size(0), u.size(-1), N}, u.device());
    std::vector<torch::Tensor> ys;

    // Scan implementation
    for (int i = 0; i < L; i++) {
        // Update state using discretized state space equation
        x = x + delta.index({"...", i, :"}).unsqueeze(-1) * (
            torch::matmul(x, A.transpose(-1, -2)) + 
            u.index({"...", i, :"}).unsqueeze(-1) * B
        );

        // Generate output
        auto C_i = C.index({"...", i, :"}).view({C.size(0), 1, -1});
        auto y = (x * C_i).sum(-1);
        ys.push_back(y);
    }

    auto y = torch::stack(ys, -2);
    y = y + u * D.unsqueeze(-2);

    return y;
} 