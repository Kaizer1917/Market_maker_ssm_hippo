#pragma once

#include "market_data.h"
#include "order_manager.h"
#include "stable_vector.h"
#include <queue>
#include <random>
#include <system_error>

class ExchangeError : public std::runtime_error {
public:
    enum class ErrorCode {
        CONNECTIVITY_LOST,
        RATE_LIMIT_EXCEEDED,
        INSUFFICIENT_LIQUIDITY,
        INVALID_ORDER,
        SYSTEM_ERROR
    };
    
    ExchangeError(ErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}
    
    ErrorCode code() const { return code_; }
private:
    ErrorCode code_;
};

class OrderBookSimulator {
public:
    struct SimConfig {
        double base_tick_size{0.01};
        double base_lot_size{1.0};
        double price_volatility{0.0001};
        double volume_volatility{0.1};
        double cancel_rate{0.3};
        double modify_rate{0.2};
        size_t max_book_levels{20};
        bool simulate_latency{true};
        std::chrono::microseconds mean_latency{100};
    };
    
    explicit OrderBookSimulator(SimConfig config)
        : config_(config)
        , rng_(std::random_device{}())
        , latency_dist_(
              config.mean_latency.count(),
              config.mean_latency.count() * 0.2
          ) {}
    
    struct SimulatedOrder {
        Order order;
        std::chrono::nanoseconds arrival_time;
        std::chrono::nanoseconds process_time;
        bool is_marketable;
    };
    
    MarketDepth simulate_step(
        const MarketDepth& base_depth,
        const stable_vector<Order>& new_orders
    ) {
        try {
            if (!is_exchange_healthy()) {
                throw ExchangeError(
                    ExchangeError::ErrorCode::CONNECTIVITY_LOST,
                    "Exchange connection lost"
                );
            }
            // ... existing simulation logic ...
        } catch (const std::exception& e) {
            handle_simulation_error(e);
            return current_depth_; // Return last known state
        }
    }
    
    void add_order(const Order& order);
    void cancel_order(int64_t order_id);
    void modify_order(const Order& order);
    
    // Getters for simulation state
    const MarketDepth& get_current_depth() const { return current_depth_; }
    const stable_vector<SimulatedOrder>& get_processed_orders() const {
        return processed_orders_;
    }
    
private:
    SimConfig config_;
    MarketDepth current_depth_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> latency_dist_;
    
    // Order processing queues
    std::priority_queue<
        SimulatedOrder,
        std::vector<SimulatedOrder>,
        std::greater<>
    > order_queue_;
    
    stable_vector<SimulatedOrder> processed_orders_;
    
    // Internal state
    struct PriceLevel {
        double price;
        double total_volume;
        std::unordered_map<int64_t, Order> orders;
    };
    
    std::map<double, PriceLevel, std::greater<>> bid_levels_;
    std::map<double, PriceLevel> ask_levels_;
    
    // Helper methods
    void process_queue(std::chrono::nanoseconds current_time);
    void match_orders();
    void update_book_state();
    void simulate_market_impact(const Order& order);
    std::chrono::nanoseconds simulate_latency();
    
    // Add new members
    std::atomic<bool> is_healthy_{true};
    std::chrono::steady_clock::time_point last_heartbeat_;
    
    bool is_exchange_healthy() const {
        auto now = std::chrono::steady_clock::now();
        return is_healthy_.load() && 
               (now - last_heartbeat_) < std::chrono::seconds(5);
    }
    
    void handle_simulation_error(const std::exception& e);
}; 