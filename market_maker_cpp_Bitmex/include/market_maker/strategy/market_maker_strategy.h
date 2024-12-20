#pragma once

#include "market_data.h"
#include "order_manager.h"
#include "stable_vector.h"
#include "bitmex_connector.h"
#include <memory>
#include <mutex>
#include <atomic>

class MarketMakingStrategy {
public:
    struct Config {
        double spread_multiplier = 1.0;      
        double position_limit = 1000.0;      
        double order_size = 100.0;          
        double risk_factor = 0.1;           
        int max_orders_per_side = 3;        
        
        virtual ~Config() = default;
    };

    MarketMakingStrategy(
        std::shared_ptr<MarketPredictor> predictor,
        std::shared_ptr<OrderManager> order_manager,
        std::shared_ptr<BitMEXConnector> bitmex_connector,
        const Config& config)
        : predictor_(predictor)
        , order_manager_(order_manager)
        , bitmex_connector_(bitmex_connector)
        , config_(config)
        , is_running_(false)
        , active_orders_(256)  // Pre-allocate space for active orders
        , market_data_history_(1024)  // Pre-allocate space for market data history
    {}
    
    virtual ~MarketMakingStrategy() {
        stop();
    }
    
    virtual bool initialize() {
        std::lock_guard<std::mutex> lock(strategy_mutex_);
        is_running_ = true;
        active_orders_.reserve(256);  // Reserve space for typical usage
        market_data_history_.reserve(1024);  // Reserve space for market data history
        return true;
    }
    
    virtual void stop() {
        std::lock_guard<std::mutex> lock(strategy_mutex_);
        is_running_ = false;
        active_orders_.clear();
        market_data_history_.clear();
    }

    virtual void on_market_data(const MarketDepth& depth) = 0;
    
    virtual void handle_error(const std::string& error_msg) {
        std::lock_guard<std::mutex> lock(strategy_mutex_);
        error_history_.push_back(error_msg);
        if (error_history_.size() > 1000) {  // Keep last 1000 errors
            error_history_.erase(error_history_.begin());
        }
    }

protected:
    std::shared_ptr<MarketPredictor> predictor_;
    std::shared_ptr<OrderManager> order_manager_;
    std::shared_ptr<BitMEXConnector> bitmex_connector_;
    Config config_;
    
    // Strategy state
    std::atomic<bool> is_running_;
    std::mutex strategy_mutex_;
    
    // Use stable_vector for active orders to maintain pointer stability
    stable_vector<Order> active_orders_;
    
    // Use stable_vector for market data history
    stable_vector<MarketDepth> market_data_history_;
    
    // Use stable_vector for error history to avoid reallocation
    stable_vector<std::string> error_history_;
    
    bool is_running() const { return is_running_; }
    
    virtual bool validate_market_data(const MarketDepth& depth) const {
        return depth.is_valid();
    }
    
    // Helper method to maintain market data history
    void update_market_history(const MarketDepth& depth) {
        std::lock_guard<std::mutex> lock(strategy_mutex_);
        market_data_history_.push_back(depth);
        
        // Keep a fixed window of history
        static constexpr size_t MAX_HISTORY = 1000;
        if (market_data_history_.size() > MAX_HISTORY) {
            // Remove oldest entries
            market_data_history_.erase(market_data_history_.begin());
        }
    }
    
    // Helper method to update active orders
    void update_active_orders(const Order& order) {
        std::lock_guard<std::mutex> lock(strategy_mutex_);
        // Find and remove cancelled/filled orders
        auto it = std::remove_if(active_orders_.begin(), active_orders_.end(),
            [](const Order& o) { return o.is_complete(); });
        active_orders_.erase(it, active_orders_.end());
        
        // Add new order
        active_orders_.push_back(order);
    }
};