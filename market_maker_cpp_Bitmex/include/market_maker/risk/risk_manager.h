#pragma once

#include <atomic>
#include <chrono>
#include "market_data.h"
#include "order_manager.h"

class RiskManager {
public:
    struct RiskLimits {
        double max_position_value{1000000.0};
        double max_daily_loss{50000.0};
        double max_order_value{100000.0};
        double max_position_concentration{0.2};
        int max_message_rate_per_second{100};
        double max_adverse_selection{0.01};
        double var_limit{100000.0};
        double stress_test_multiplier{3.0};
    };
    
    explicit RiskManager(RiskLimits limits) 
        : limits_(limits)
        , start_time_(std::chrono::system_clock::now()) {}
    
    bool check_order_risk(const Order& order, const MarketDepth& depth);
    bool check_position_risk(const std::string& symbol, double position, double price);
    void update_metrics(const Order& order, const MarketDepth& depth);
    void calculate_var(const stable_vector<double>& returns, double confidence = 0.99);
    
    // Real-time monitoring
    struct RiskMetrics {
        std::atomic<double> current_var{0.0};
        std::atomic<double> daily_pnl{0.0};
        std::atomic<double> max_drawdown{0.0};
        std::atomic<int> message_count{0};
        std::atomic<double> adverse_selection_cost{0.0};
        std::chrono::system_clock::time_point last_reset;
    };
    
    const RiskMetrics& get_metrics() const { return metrics_; }
    void reset_daily_metrics();

    struct CircuitBreaker {
        double loss_threshold;
        int max_consecutive_losses;
        double max_drawdown;
        std::chrono::seconds cooldown_period;
        bool is_triggered{false};
        std::chrono::steady_clock::time_point trigger_time;
    };

    void check_circuit_breakers() {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        if (metrics_.daily_pnl < -circuit_breaker_.loss_threshold ||
            metrics_.max_drawdown > circuit_breaker_.max_drawdown) {
            
            trigger_circuit_breaker("Risk limits exceeded");
        }
    }

private:
    RiskLimits limits_;
    RiskMetrics metrics_;
    std::chrono::system_clock::time_point start_time_;
    
    // Thread-safe metric updates
    std::mutex metrics_mutex_;
    stable_vector<double> price_history_;
    stable_vector<double> pnl_history_;
    
    // Risk calculation helpers
    double calculate_adverse_selection(const Order& order, const MarketDepth& depth);
    double calculate_position_concentration(const std::string& symbol);
    bool run_stress_test(double var, double position_value);
    
    CircuitBreaker circuit_breaker_;
    
    void trigger_circuit_breaker(const std::string& reason) {
        circuit_breaker_.is_triggered = true;
        circuit_breaker_.trigger_time = std::chrono::steady_clock::now();
        // Notify strategy manager to stop trading
    }
}; 