#include <gtest/gtest.h>
#include <market_maker/model/metrics.h>

class MetricsCalculatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        predictions = torch::randn({32, 128, 4});
        targets = predictions + torch::randn({32, 128, 4}) * 0.1;
    }
    
    torch::Tensor predictions;
    torch::Tensor targets;
};

TEST_F(MetricsCalculatorTest, BasicMetrics) {
    auto metrics = MetricsCalculator::calculate_metrics(predictions, targets);
    
    EXPECT_GT(metrics.mse, 0.0);
    EXPECT_GT(metrics.rmse, 0.0);
    EXPECT_GT(metrics.mae, 0.0);
    EXPECT_LT(metrics.mape, 100.0);
}

TEST_F(MetricsCalculatorTest, ChannelMetrics) {
    auto channel_metrics = MetricsCalculator::calculate_channel_metrics(
        predictions, targets);
    
    EXPECT_EQ(channel_metrics.size(), 4);
    for (const auto& [channel, value] : channel_metrics) {
        EXPECT_GT(value, 0.0);
    }
}

TEST_F(MetricsCalculatorTest, RollingMetrics) {
    auto rolling_metrics = MetricsCalculator::calculate_rolling_metrics(
        predictions, targets, 32);
    
    EXPECT_EQ(rolling_metrics.size(0), predictions.size(0) - 31);
} 