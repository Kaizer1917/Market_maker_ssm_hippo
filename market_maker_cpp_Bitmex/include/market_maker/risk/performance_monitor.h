#pragma once

#include <chrono>
#include "risk_manager.h"

class PerformanceMonitor {
public:
    struct PerformanceMetrics {
        double sharpe_ratio{0.0};
        double information_ratio{0.0};
        double max_drawdown{0.0};
        double win_rate{0.0};
        double profit_factor{0.0};
        double avg_adverse_selection{0.0};
        double avg_spread_capture{0.0};
        int trades_per_second{0};
    };
    
    void update_trade_metrics(const Order& order, const MarketDepth& depth);
    void calculate_performance_metrics();
    const PerformanceMetrics& get_metrics() const { return metrics_; }
    
private:
    PerformanceMetrics metrics_;
    stable_vector<double> trade_pnls_;
    stable_vector<double> trade_times_;
    
    // Helper methods
    double calculate_sharpe_ratio();
    double calculate_information_ratio(const stable_vector<double>& benchmark_returns);
    void update_trade_statistics(double pnl);
}; 