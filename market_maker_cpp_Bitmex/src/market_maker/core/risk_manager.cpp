#include "risk_manager.h"
#include <algorithm>
#include <numeric>
#include <cmath>

bool RiskManager::check_order_risk(const Order& order, const MarketDepth& depth) {
    // Check order value
    double order_value = order.price * order.quantity;
    if (order_value > limits_.max_order_value) {
        return false;
    }
    
    // Check message rate
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - metrics_.last_reset
    ).count();
    
    if (elapsed > 0 && 
        metrics_.message_count.load() / elapsed > limits_.max_message_rate_per_second) {
        return false;
    }
    
    // Check adverse selection
    double adverse_selection = calculate_adverse_selection(order, depth);
    if (adverse_selection > limits_.max_adverse_selection) {
        return false;
    }
    
    // Update message count
    metrics_.message_count.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void RiskManager::calculate_var(
    const stable_vector<double>& returns,
    double confidence) {
    
    std::vector<double> sorted_returns(returns.begin(), returns.end());
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    size_t var_index = static_cast<size_t>(
        (1.0 - confidence) * sorted_returns.size()
    );
    
    double var = -sorted_returns[var_index];
    metrics_.current_var.store(var, std::memory_order_release);
    
    // Run stress test
    if (!run_stress_test(var, metrics_.daily_pnl.load())) {
        // Trigger risk alert
        // Implementation omitted for brevity
    }
}

void RiskManager::update_metrics(const Order& order, const MarketDepth& depth) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Update P&L
    double trade_pnl = order.side == OrderSide::BUY ?
        -(order.price * order.filled_quantity) :
        order.price * order.filled_quantity;
    
    double current_pnl = metrics_.daily_pnl.load();
    metrics_.daily_pnl.store(current_pnl + trade_pnl, std::memory_order_release);
    
    // Update max drawdown
    if (trade_pnl < 0) {
        double drawdown = std::abs(trade_pnl);
        double current_max_drawdown = metrics_.max_drawdown.load();
        if (drawdown > current_max_drawdown) {
            metrics_.max_drawdown.store(drawdown, std::memory_order_release);
        }
    }
    
    // Update price history for VaR calculation
    if (price_history_.size() > 1000) {
        price_history_.erase(price_history_.begin());
    }
    price_history_.push_back(depth.get_mid_price());
    
    // Calculate returns and update VaR
    if (price_history_.size() > 1) {
        stable_vector<double> returns;
        returns.reserve(price_history_.size() - 1);
        
        for (size_t i = 1; i < price_history_.size(); ++i) {
            returns.push_back(
                std::log(price_history_[i] / price_history_[i-1])
            );
        }
        
        calculate_var(returns);
    }
}

double RiskManager::calculate_adverse_selection(
    const Order& order,
    const MarketDepth& depth) {
    
    double mid_price = depth.get_mid_price();
    double execution_price = order.price;
    
    if (order.side == OrderSide::BUY) {
        return (execution_price - mid_price) / mid_price;
    } else {
        return (mid_price - execution_price) / mid_price;
    }
}

bool RiskManager::run_stress_test(double var, double position_value) {
    double stressed_var = var * limits_.stress_test_multiplier;
    return std::abs(position_value) * stressed_var <= limits_.var_limit;
} 