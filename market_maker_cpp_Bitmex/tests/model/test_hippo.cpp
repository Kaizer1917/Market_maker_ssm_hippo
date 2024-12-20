#include <gtest/gtest.h>
#include <market_maker/model/ssm_hippo.h>

class SSMHippoTest : public ::testing::Test {
protected:
    void SetUp() override {
        args.d_model = 64;
        args.d_state = 64;
        args.seq_len = 128;
        args.num_channels = 4;
        model = std::make_unique<SSMHippo>(args);
    }
    
    ModelArgs args;
    std::unique_ptr<SSMHippo> model;
};

TEST_F(SSMHippoTest, ForwardPass) {
    torch::Tensor input = torch::randn({32, 4, 128}); // batch, channels, seq_len
    auto output = model->forward(input);
    
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(SSMHippoTest, StateEvolution) {
    torch::Tensor input = torch::randn({1, 4, 128});
    auto [output, states] = model->forward_with_states(input);
    
    EXPECT_EQ(states.size(2), args.d_state);
    EXPECT_TRUE(states.abs().mean().item<float>() < 100.0); // Check state stability
}

TEST_F(SSMHippoTest, Serialization) {
    torch::Tensor input = torch::randn({1, 4, 128});
    auto output1 = model->forward(input);
    
    model->save("test_model.pt");
    auto loaded_model = std::make_unique<SSMHippo>(args);
    loaded_model->load("test_model.pt");
    
    auto output2 = loaded_model->forward(input);
    EXPECT_TRUE(torch::allclose(output1, output2, 1e-5));
} 