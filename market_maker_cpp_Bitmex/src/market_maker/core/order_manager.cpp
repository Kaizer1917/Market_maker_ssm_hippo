#include "order_manager.h"
#include <algorithm>
#include <chrono>

std::optional<Order> OrderManager::place_order(
    OrderSide side, 
    double price, 
    double quantity) {
    
    // Risk checks
    if (!check_risk_limits(side, quantity, price)) {
        return std::nullopt;
    }
    
    // Create new order
    Order order{
        .order_id = generate_order_id(),
        .side = side,
        .price = price,
        .quantity = quantity,
        .creation_time = std::chrono::system_clock::now().time_since_epoch().count(),
        .last_update_time = std::chrono::system_clock::now().time_since_epoch().count()
    };
    
    // Update position tracking
    {
        std::unique_lock<std::shared_mutex> lock(orders_mutex_);
        
        // Check max active orders
        if (std::count_if(active_orders_.begin(), active_orders_.end(),
                         [](const Order& o) { return o.is_active(); }) 
            >= config_.max_active_orders) {
            return std::nullopt;
        }
        
        active_orders_.push_back(order);
    }
    
    return order;
}

bool OrderManager::check_risk_limits(
    OrderSide side, 
    double quantity, 
    double price) const {
    
    if (quantity > config_.max_order_size) {
        return false;
    }
    
    double current_position = position_.load(std::memory_order_acquire);
    double position_delta = side == OrderSide::BUY ? quantity : -quantity;
    double new_position = current_position + position_delta;
    
    if (std::abs(new_position) > config_.max_position) {
        return false;
    }
    
    double current_notional = notional_exposure_.load(std::memory_order_acquire);
    double notional_delta = price * quantity;
    
    if (current_notional + notional_delta > config_.max_notional) {
        return false;
    }
    
    return true;
}

void OrderManager::update_order(const Order& order) {
    std::unique_lock<std::shared_mutex> lock(orders_mutex_);
    
    auto it = std::find_if(active_orders_.begin(), active_orders_.end(),
                          [&](const Order& o) { return o.order_id == order.order_id; });
    
    if (it != active_orders_.end()) {
        double fill_delta = order.filled_quantity - it->filled_quantity;
        
        if (fill_delta > 0) {
            // Update position
            double position_delta = order.side == OrderSide::BUY ? fill_delta : -fill_delta;
            position_.fetch_add(position_delta, std::memory_order_release);
            
            // Update notional exposure
            double notional_delta = order.price * fill_delta;
            notional_exposure_.fetch_add(notional_delta, std::memory_order_release);
        }
        
        *it = order;
    }
}
