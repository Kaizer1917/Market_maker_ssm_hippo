#pragma once

#include "bitmex_connector.h"
#include "order_manager.h"
#include <unordered_map>
#include <shared_mutex>

class BitMEXExecutionManager {
public:
    struct ExecutionConfig {
        int max_retry_attempts = 3;
        std::chrono::milliseconds retry_delay{500};
        double max_position_value = 100000.0;  // Maximum position value in USD
        double max_order_value = 10000.0;      // Maximum single order value in USD
        double max_leverage = 5.0;             // Maximum allowed leverage
    };

    explicit BitMEXExecutionManager(
        std::shared_ptr<BitMEXConnector> connector,
        ExecutionConfig config = ExecutionConfig{})
        : connector_(connector)
        , config_(config) {}

    // Order execution methods
    bool submit_order(Order& order);
    bool cancel_order(int64_t order_id);
    bool amend_order(const Order& order);
    
    // Order tracking
    std::optional<Order> get_order_status(int64_t order_id);
    std::vector<Order> get_active_orders();
    
    // Risk management
    bool check_risk_limits(const Order& order);
    double get_current_leverage();

private:
    std::shared_ptr<BitMEXConnector> connector_;
    ExecutionConfig config_;
    
    // Order tracking
    mutable std::shared_mutex orders_mutex_;
    std::unordered_map<int64_t, Order> active_orders_;
    
    // Execution helpers
    bool retry_order_submission(Order& order, int attempts = 0);
    void update_order_status(const Order& order);
    
    // Risk checks
    bool validate_order_size(const Order& order);
    bool validate_position_value(const Order& order);
    bool validate_leverage(const Order& order);
}; 