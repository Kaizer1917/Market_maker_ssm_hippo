#include "backtest_engine.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <execution>

BacktestEngine::BacktestResults BacktestEngine::run() {
    BacktestResults results;
    current_capital_ = config_.initial_capital;
    
    // Load market data
    load_market_data();
    
    // Initialize result containers
    results.equity_curve.reserve(market_data_.size());
    results.drawdown_curve.reserve(market_data_.size());
    results.position_history.reserve(market_data_.size());
    
    double high_water_mark = current_capital_;
    double max_drawdown = 0.0;
    
    // Warm-up period
    for (size_t i = 0; i < config_.warm_up_bars && i < market_data_.size(); ++i) {
        strategy_->on_market_data(market_data_[i]);
    }
    
    // Main backtest loop
    #pragma omp parallel for if(market_data_.size() > 10000)
    for (size_t i = config_.warm_up_bars; i < market_data_.size(); ++i) {
        const auto& depth = market_data_[i];
        
        // Strategy execution
        strategy_->on_market_data(depth);
        
        // Process orders
        auto orders = strategy_->get_active_orders();
        for (const auto& order : orders) {
            if (risk_manager_->check_order_risk(order, depth)) {
                // Apply transaction costs and slippage
                double transaction_cost = 0.0;
                double slippage = 0.0;
                
                if (config_.include_transaction_costs) {
                    transaction_cost = calculate_transaction_costs(order);
                }
                
                if (config_.include_slippage) {
                    slippage = calculate_slippage(order, depth);
                }
                
                // Update position and capital
                update_position(order);
                current_capital_ -= (transaction_cost + slippage);
                
                // Record trade
                results.trade_history.push_back(order);
                
                // Update metrics
                results.total_transaction_costs += transaction_cost;
                results.total_slippage += slippage;
            }
        }
        
        // Update equity curve and drawdown
        results.equity_curve.push_back(current_capital_);
        
        if (current_capital_ > high_water_mark) {
            high_water_mark = current_capital_;
        }
        
        double drawdown = (high_water_mark - current_capital_) / high_water_mark;
        results.drawdown_curve.push_back(drawdown);
        max_drawdown = std::max(max_drawdown, drawdown);
        
        // Record position
        results.position_history.push_back({
            depth.get_mid_price(),
            strategy_->get_current_position()
        });
        
        // Record metrics
        record_metrics(order, depth);
    }
    
    // Calculate final metrics
    analyze_results();
    results.metrics = performance_monitor_.get_metrics();
    results.max_leverage_used = calculate_max_leverage();
    
    return results;
}

void BacktestEngine::analyze_results() {
    // Calculate advanced metrics
    auto calculate_returns = [](const stable_vector<double>& prices) {
        stable_vector<double> returns;
        returns.reserve(prices.size() - 1);
        
        for (size_t i = 1; i < prices.size(); ++i) {
            returns.push_back(std::log(prices[i] / prices[i-1]));
        }
        return returns;
    };
    
    // Calculate strategy returns
    auto strategy_returns = calculate_returns(results.equity_curve);
    
    // Calculate benchmark returns (assuming market_data_ contains benchmark prices)
    stable_vector<double> benchmark_prices;
    std::transform(market_data_.begin(), market_data_.end(),
                  std::back_inserter(benchmark_prices),
                  [](const MarketDepth& depth) { return depth.get_mid_price(); });
    
    auto benchmark_returns = calculate_returns(benchmark_prices);
    
    // Update performance metrics
    performance_monitor_.calculate_performance_metrics(
        strategy_returns,
        benchmark_returns
    );
}

double BacktestEngine::calculate_transaction_costs(const Order& order) {
    return order.price * order.quantity * (config_.transaction_cost_bps / 10000.0);
}

double BacktestEngine::calculate_slippage(
    const Order& order,
    const MarketDepth& depth) {
    
    double base_slippage = order.price * (config_.slippage_bps / 10000.0);
    
    // Add market impact based on order size relative to available liquidity
    double market_impact = 0.0;
    if (order.side == OrderSide::BUY) {
        double available_liquidity = std::accumulate(
            depth.asks.begin(),
            depth.asks.end(),
            0.0,
            [](double sum, const MarketDepth::Level& level) {
                return sum + level.quantity;
            }
        );
        market_impact = base_slippage * (order.quantity / available_liquidity);
    } else {
        double available_liquidity = std::accumulate(
            depth.bids.begin(),
            depth.bids.end(),
            0.0,
            [](double sum, const MarketDepth::Level& level) {
                return sum + level.quantity;
            }
        );
        market_impact = base_slippage * (order.quantity / available_liquidity);
    }
    
    return base_slippage + market_impact;
} 