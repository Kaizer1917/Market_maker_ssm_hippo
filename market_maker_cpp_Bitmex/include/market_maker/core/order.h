#pragma once

#include <atomic>
#include <memory>
#include "stable_vector.h"
#include "market_data.h"

enum class OrderSide { BUY, SELL };
enum class OrderStatus { NEW, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED };

struct Order {
    int64_t order_id;
    OrderSide side;
    double price;
    double quantity;
    double filled_quantity{0.0};
    OrderStatus status{OrderStatus::NEW};
    int64_t creation_time;
    int64_t last_update_time;
    
    bool is_active() const {
        return status == OrderStatus::NEW || status == OrderStatus::PARTIALLY_FILLED;
    }
};

class OrderManager {
public:
    struct Config {
        double max_position{100.0};
        double max_order_size{10.0};
        double max_notional{10000.0};
        int max_active_orders{50};
        double min_spread{0.0001};
    };
    
    explicit OrderManager(Config config) : config_(config) {}
    
    // Thread-safe order operations
    std::optional<Order> place_order(OrderSide side, double price, double quantity);
    bool cancel_order(int64_t order_id);
    void update_order(const Order& order);
    
    // Position management
    double get_position() const { return position_.load(std::memory_order_acquire); }
    double get_notional_exposure() const { return notional_exposure_.load(std::memory_order_acquire); }
    
    // Risk checks
    bool check_risk_limits(OrderSide side, double quantity, double price) const;
    
private:
    Config config_;
    std::atomic<int64_t> next_order_id_{1};
    std::atomic<double> position_{0.0};
    std::atomic<double> notional_exposure_{0.0};
    
    mutable std::shared_mutex orders_mutex_;
    stable_vector<Order> active_orders_;
    
    int64_t generate_order_id() {
        return next_order_id_.fetch_add(1, std::memory_order_relaxed);
    }
};
