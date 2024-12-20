#include <gtest/gtest.h>
#include <market_maker/model/data_utils.h>

class DataPreprocessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        args.seq_len = 128;
        args.patch_len = 16;
        args.num_channels = 4;
        preprocessor = std::make_unique<DataPreprocessor>(args);
    }
    
    ModelArgs args;
    std::unique_ptr<DataPreprocessor> preprocessor;
};

TEST_F(DataPreprocessorTest, Normalization) {
    torch::Tensor data = torch::randn({100, 4, 128});
    auto normalized = preprocessor->normalize_data(data);
    
    // Check mean and std
    auto mean = normalized.mean();
    auto std = normalized.std();
    EXPECT_NEAR(mean.item<float>(), 0.0, 1e-5);
    EXPECT_NEAR(std.item<float>(), 1.0, 1e-5);
}

TEST_F(DataPreprocessorTest, PatchCreation) {
    torch::Tensor data = torch::randn({1, 4, 128});
    auto [x, y] = preprocessor->prepare_data(data);
    
    int expected_patches = (args.seq_len - args.patch_len) / args.stride + 1;
    EXPECT_EQ(x.size(1), expected_patches);
}

TEST_F(DataPreprocessorTest, StatisticsPersistence) {
    torch::Tensor data = torch::randn({100, 4, 128});
    preprocessor->update_statistics(data);
    preprocessor->save_statistics("test_stats.json");
    
    auto new_preprocessor = std::make_unique<DataPreprocessor>(args);
    new_preprocessor->load_statistics("test_stats.json");
    
    auto norm1 = preprocessor->normalize_data(data);
    auto norm2 = new_preprocessor->normalize_data(data);
    
    EXPECT_TRUE(torch::allclose(norm1, norm2, 1e-5));
} 