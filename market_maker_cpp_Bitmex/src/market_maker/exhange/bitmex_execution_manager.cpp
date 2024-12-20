#include "bitmex_execution_manager.h"
#include <thread>

bool BitMEXExecutionManager::submit_order(Order& order) {
    // Perform risk checks
    if (!check_risk_limits(order)) {
        return false;
    }
    
    // Try to submit the order
    return retry_order_submission(order);
}

bool BitMEXExecutionManager::retry_order_submission(Order& order, int attempts) {
    while (attempts < config_.max_retry_attempts) {
        try {
            if (connector_->place_order(order)) {
                // Update order tracking
                std::unique_lock<std::shared_mutex> lock(orders_mutex_);
                active_orders_[order.order_id] = order;
                return true;
            }
        }
        catch (const std::exception& e) {
            // Log error and retry
            attempts++;
            if (attempts < config_.max_retry_attempts) {
                std::this_thread::sleep_for(config_.retry_delay);
            }
            continue;
        }
    }
    return false;
}

bool BitMEXExecutionManager::check_risk_limits(const Order& order) {
    return validate_order_size(order) &&
           validate_position_value(order) &&
           validate_leverage(order);
}

bool BitMEXExecutionManager::validate_order_size(const Order& order) {
    double order_value = order.price * order.quantity;
    return order_value <= config_.max_order_value;
}

bool BitMEXExecutionManager::validate_position_value(const Order& order) {
    // Get current position from BitMEX
    double current_position = connector_->get_current_position();
    double new_position = current_position;
    
    if (order.side == OrderSide::BUY) {
        new_position += order.quantity;
    } else {
        new_position -= order.quantity;
    }
    
    double position_value = std::abs(new_position * order.price);
    return position_value <= config_.max_position_value;
}

bool BitMEXExecutionManager::validate_leverage(const Order& order) {
    double leverage = get_current_leverage();
    return leverage <= config_.max_leverage;
} 