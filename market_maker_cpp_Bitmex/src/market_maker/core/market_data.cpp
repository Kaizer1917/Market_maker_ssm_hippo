#include "market_data.h"
#include <algorithm>

void MarketDepth::update_ask(size_t level, double price, double qty) {
    if (level >= MAX_LEVELS) return;
    
    asks[level].price = price;
    asks[level].quantity = qty;
    asks[level].update_time.store(
        std::chrono::system_clock::now().time_since_epoch().count(),
        std::memory_order_release
    );
    
    last_update.store(
        std::chrono::system_clock::now().time_since_epoch().count(),
        std::memory_order_release
    );
}

void MarketDepth::update_bid(size_t level, double price, double qty) {
    if (level >= MAX_LEVELS) return;
    
    bids[level].price = price;
    bids[level].quantity = qty;
    bids[level].update_time.store(
        std::chrono::system_clock::now().time_since_epoch().count(),
        std::memory_order_release
    );
    
    last_update.store(
        std::chrono::system_clock::now().time_since_epoch().count(),
        std::memory_order_release
    );
}

double MarketDepth::get_mid_price() const {
    if (asks[0].price <= 0 || bids[0].price <= 0) return 0.0;
    return (asks[0].price + bids[0].price) * 0.5;
}

double MarketDepth::get_spread() const {
    if (asks[0].price <= 0 || bids[0].price <= 0) return 0.0;
    return asks[0].price - bids[0].price;
}
