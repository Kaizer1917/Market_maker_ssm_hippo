#include <gtest/gtest.h>
#include <market_maker/model/channel_attention.h>
#include <torch/torch.h>

class ChannelAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        args.d_model = 64;
        args.num_heads = 4;
        args.num_channels = 8;
        attention = std::make_unique<ChannelAttention>(args);
    }
    
    ModelArgs args;
    std::unique_ptr<ChannelAttention> attention;
};

TEST_F(ChannelAttentionTest, OutputShape) {
    torch::Tensor input = torch::randn({32, 8, 64}); // batch_size, num_channels, d_model
    auto output = attention->forward(input);
    
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(ChannelAttentionTest, AttentionWeights) {
    torch::Tensor input = torch::randn({1, 8, 64});
    auto [output, weights] = attention->forward_with_attention(input);
    
    // Check attention weights sum to 1
    auto sum_weights = weights.sum({-1});
    EXPECT_TRUE(torch::allclose(sum_weights, torch::ones_like(sum_weights), 1e-5));
}

TEST_F(ChannelAttentionTest, GradientFlow) {
    torch::Tensor input = torch::randn({16, 8, 64}, torch::requires_grad());
    auto output = attention->forward(input);
    auto loss = output.mean();
    loss.backward();
    
    EXPECT_TRUE(input.grad().defined());
} 