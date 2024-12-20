#include <market_maker/core/order_manager.h>
#include <market_maker/core/risk_manager.h>
#include <market_maker/strategy/rollercoaster_girls.h>
#include <market_maker/model/model_args.h>
#include <iostream>

int main() {
    try {
        // Initialize configuration
        ModelArgs model_args;
        model_args.d_model = 128;
        model_args.n_layer = 4;
        model_args.seq_len = 96;
        model_args.num_channels = 24;
        
        MarketPredictor::Config predictor_config;
        predictor_config.model_args = model_args;
        predictor_config.inference.use_cuda = true;
        
        // Initialize market predictor
        MarketPredictor predictor(predictor_config);
        
        // Initialize risk manager
        RiskManager::Limits risk_limits;
        risk_limits.max_position = 1000000;
        risk_limits.max_order_value = 100000;
        risk_limits.max_message_rate_per_second = 50;
        auto risk_manager = std::make_shared<RiskManager>(risk_limits);
        
        // Initialize order manager
        OrderManager::Config order_config;
        order_config.max_active_orders = 100;
        order_config.max_position = 1000000;
        auto order_manager = std::make_shared<OrderManager>(order_config);
        
        // Initialize strategy
        auto strategy = std::make_shared<MarketMakingStrategy>(
            predictor,
            order_manager,
            risk_manager
        );
        
        // Market data simulation
        MarketDepth depth;
        depth.bids = {{100.0, 1.0}, {99.0, 2.0}};
        depth.asks = {{101.0, 1.0}, {102.0, 2.0}};
        
        // Strategy execution
        strategy->on_market_data(depth);
        
        // Print active orders
        auto active_orders = order_manager->get_active_orders();
        std::cout << "Active orders: " << active_orders.size() << std::endl;
        for (const auto& order : active_orders) {
            std::cout << "Order ID: " << order.order_id
                     << ", Side: " << (order.side == OrderSide::BUY ? "BUY" : "SELL")
                     << ", Price: " << order.price
                     << ", Quantity: " << order.quantity
                     << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 