#include "stoikov_strategy.h"
#include <execution>
#include <cmath>
#include <random>

void StoikovStrategy::on_market_data(const MarketDepth& depth) {
    // Update price history and volatility estimate
    {
        std::lock_guard<std::mutex> lock(market_data_mutex_);
        double mid_price = depth.get_mid_price();
        volatility_estimator_.update(mid_price);
        
        if (price_history_.size() >= config_.volatility_window) {
            price_history_.erase(price_history_.begin());
        }
        price_history_.push_back(mid_price);
    }
    market_data_cv_.notify_one();

    // Calculate time remaining (similar to T in Bitmex/main.py)
    double time_remaining = config_.time_horizon - 
                          (std::time(nullptr) - start_time_) / (24.0 * 3600.0);
    if (time_remaining <= 0) return;

    // Get current position and volatility
    double inventory = order_manager_->get_position();
    double volatility = volatility_estimator_.get_volatility();
    double mid_price = depth.get_mid_price();

    // Calculate reserve price (similar to Bitmex/main.py implementation)
    double reserve_price = mid_price - 
                         inventory * config_.risk_aversion * 
                         std::pow(volatility, 2) * time_remaining;

    // Calculate optimal spread using Stoikov formula
    double reserve_spread = (2.0 / config_.risk_aversion) * 
                          std::log(1.0 + config_.risk_aversion / config_.market_impact);

    // Calculate optimal quotes
    double ask_price = reserve_price + reserve_spread / 2.0;
    double bid_price = reserve_price - reserve_spread / 2.0;

    // Calculate order intensities (similar to Bitmex/main.py)
    double delta_ask = ask_price - mid_price;
    double delta_bid = mid_price - bid_price;
    
    double base_intensity = mid_price / (200.0 * config_.time_horizon);
    double ask_intensity = base_intensity * std::exp(-config_.market_impact * delta_ask);
    double bid_intensity = base_intensity * std::exp(-config_.market_impact * delta_bid);

    // Calculate position-dependent order sizes
    double inventory_skew = (inventory - config_.inventory_target) / config_.position_limit;
    double base_size = config_.order_size;
    
    double bid_size = base_size * std::exp(-config_.risk_aversion * inventory_skew);
    double ask_size = base_size * std::exp(config_.risk_aversion * inventory_skew);

    // Place orders through BitMEX
    if (bid_size > 0.0 && bid_intensity > config_.min_intensity) {
        Order bid_order{
            .side = OrderSide::BUY,
            .price = bid_price,
            .quantity = bid_size
        };
        
        if (bitmex_connector_->place_order(bid_order)) {
            order_manager_->update_order(bid_order);
        }
    }
    
    if (ask_size > 0.0 && ask_intensity > config_.min_intensity) {
        Order ask_order{
            .side = OrderSide::SELL,
            .price = ask_price,
            .quantity = ask_size
        };
        
        if (bitmex_connector_->place_order(ask_order)) {
            order_manager_->update_order(ask_order);
        }
    }
}

// Add Brownian motion simulation for price prediction
std::vector<double> StoikovStrategy::simulate_price_path(
    double current_price,
    double volatility,
    int n_steps) {
    
    std::vector<double> price_path(n_steps + 1);
    price_path[0] = current_price;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal(0.0, std::sqrt(config_.time_horizon / n_steps));

    for (int i = 1; i <= n_steps; ++i) {
        double drift = config_.drift * config_.time_horizon / n_steps;
        double diffusion = volatility * normal(gen);
        price_path[i] = price_path[i-1] * std::exp(drift + diffusion);
    }

    return price_path;
} 