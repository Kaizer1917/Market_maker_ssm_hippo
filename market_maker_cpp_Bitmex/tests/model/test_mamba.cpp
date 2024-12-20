#include <gtest/gtest.h>
#include <market_maker/model/mamba_block.h>

class MambaBlockTest : public ::testing::Test {
protected:
    void SetUp() override {
        args.d_model = 64;
        args.d_state = 16;
        args.expansion_factor = 2;
        mamba = std::make_unique<MambaBlock>(args);
    }
    
    ModelArgs args;
    std::unique_ptr<MambaBlock> mamba;
};

TEST_F(MambaBlockTest, OutputShape) {
    torch::Tensor input = torch::randn({32, 128, 64}); // batch, seq_len, d_model
    auto output = mamba->forward(input);
    
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(MambaBlockTest, SelectiveUpdate) {
    torch::Tensor input = torch::randn({1, 128, 64});
    auto [output, gates] = mamba->forward_with_gates(input);
    
    // Check gate values are between 0 and 1
    EXPECT_TRUE(gates.ge(0).all().item<bool>());
    EXPECT_TRUE(gates.le(1).all().item<bool>());
}

TEST_F(MambaBlockTest, MemoryEfficiency) {
    torch::Tensor input = torch::randn({128, 1024, 64});
    
    size_t mem_before = torch::cuda::memory_allocated();
    auto output = mamba->forward(input);
    size_t mem_after = torch::cuda::memory_allocated();
    
    // Check memory growth is reasonable
    EXPECT_LT(mem_after - mem_before, 1024 * 1024 * 100); // Less than 100MB growth
} 