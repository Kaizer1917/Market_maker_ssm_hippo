#pragma once

#include "market_maker_strategy.h"
#include "risk_manager.h"
#include "performance_monitor.h"
#include <filesystem>

class BacktestEngine {
public:
    struct BacktestConfig {
        std::string data_path;
        std::string output_path;
        std::chrono::system_clock::time_point start_time;
        std::chrono::system_clock::time_point end_time;
        double initial_capital{1000000.0};
        bool include_transaction_costs{true};
        double transaction_cost_bps{0.5};
        bool include_slippage{true};
        double slippage_bps{1.0};
        size_t warm_up_bars{100};
    };
    
    explicit BacktestEngine(
        std::shared_ptr<MarketMakingStrategy> strategy,
        std::shared_ptr<RiskManager> risk_manager,
        BacktestConfig config)
        : strategy_(strategy)
        , risk_manager_(risk_manager)
        , config_(config)
        , performance_monitor_() {}
    
    struct BacktestResults {
        PerformanceMonitor::PerformanceMetrics metrics;
        stable_vector<double> equity_curve;
        stable_vector<double> drawdown_curve;
        stable_vector<std::pair<double, double>> position_history;
        stable_vector<Order> trade_history;
        
        // Risk metrics
        double max_leverage_used;
        double avg_position_size;
        double max_position_size;
        double avg_holding_time;
        double turnover_ratio;
        
        // Market impact analysis
        double avg_market_impact;
        double total_transaction_costs;
        double total_slippage;
        
        void save_to_csv(const std::string& path) const;
    };
    
    BacktestResults run();
    void analyze_results();
    
private:
    std::shared_ptr<MarketMakingStrategy> strategy_;
    std::shared_ptr<RiskManager> risk_manager_;
    BacktestConfig config_;
    PerformanceMonitor performance_monitor_;
    
    // Internal state
    double current_capital_;
    stable_vector<MarketDepth> market_data_;
    
    // Helper methods
    void load_market_data();
    double calculate_transaction_costs(const Order& order);
    double calculate_slippage(const Order& order, const MarketDepth& depth);
    void update_position(const Order& order);
    void record_metrics(const Order& order, const MarketDepth& depth);
}; 