#include "market_microstructure.h"
#include <numeric>
#include <algorithm>

void MarketMicrostructure::update(const MarketDepth& depth, const Order& order) {
    // Create and store order book snapshot
    OrderBookSnapshot snapshot;
    snapshot.timestamp = std::chrono::system_clock::now().time_since_epoch();
    
    // Convert market depth to snapshot format
    for (size_t i = 0; i < MarketDepth::MAX_LEVELS; ++i) {
        if (depth.bids[i].price > 0) {
            snapshot.bids.push_back({
                depth.bids[i].price,
                depth.bids[i].quantity,
                1,  // Assuming one order per level for simplicity
                std::chrono::nanoseconds(depth.bids[i].update_time)
            });
        }
        
        if (depth.asks[i].price > 0) {
            snapshot.asks.push_back({
                depth.asks[i].price,
                depth.asks[i].quantity,
                1,
                std::chrono::nanoseconds(depth.asks[i].update_time)
            });
        }
    }
    
    // Store snapshot
    book_history_.push_back(snapshot);
    if (book_history_.size() > HISTORY_SIZE) {
        book_history_.pop_front();
    }
    
    // Store order timestamp
    order_timestamps_[order.order_id] = snapshot.timestamp;
}

double MarketMicrostructure::calculate_vpin(const stable_vector<Order>& trades) {
    constexpr size_t BUCKET_SIZE = 50;  // Number of trades per bucket
    
    if (trades.size() < BUCKET_SIZE) {
        return 0.0;
    }
    
    stable_vector<double> vpin_values;
    size_t num_buckets = trades.size() / BUCKET_SIZE;
    
    for (size_t i = 0; i < num_buckets; ++i) {
        double buy_volume = 0.0;
        double sell_volume = 0.0;
        
        for (size_t j = 0; j < BUCKET_SIZE; ++j) {
            const auto& trade = trades[i * BUCKET_SIZE + j];
            if (trade.side == OrderSide::BUY) {
                buy_volume += trade.quantity;
            } else {
                sell_volume += trade.quantity;
            }
        }
        
        double total_volume = buy_volume + sell_volume;
        if (total_volume > 0) {
            vpin_values.push_back(std::abs(buy_volume - sell_volume) / total_volume);
        }
    }
    
    return std::accumulate(vpin_values.begin(), vpin_values.end(), 0.0) /
           vpin_values.size();
}

double MarketMicrostructure::estimate_kyle_lambda(
    const stable_vector<OrderBookSnapshot>& snapshots,
    const stable_vector<Order>& trades) {
    
    if (snapshots.empty() || trades.empty()) {
        return 0.0;
    }
    
    // Calculate price changes and signed volume
    stable_vector<double> price_changes;
    stable_vector<double> signed_volumes;
    
    for (size_t i = 1; i < snapshots.size(); ++i) {
        double price_change = snapshots[i].get_weighted_midprice() -
                            snapshots[i-1].get_weighted_midprice();
        
        double signed_volume = 0.0;
        for (const auto& trade : trades) {
            if (order_timestamps_[trade.order_id] >= snapshots[i-1].timestamp &&
                order_timestamps_[trade.order_id] < snapshots[i].timestamp) {
                signed_volume += trade.side == OrderSide::BUY ?
                    trade.quantity : -trade.quantity;
            }
        }
        
        if (signed_volume != 0.0) {
            price_changes.push_back(price_change);
            signed_volumes.push_back(signed_volume);
        }
    }
    
    // Estimate Kyle's lambda using linear regression
    if (price_changes.empty()) {
        return 0.0;
    }
    
    double sum_xy = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < price_changes.size(); ++i) {
        sum_xy += price_changes[i] * signed_volumes[i];
        sum_xx += signed_volumes[i] * signed_volumes[i];
    }
    
    return sum_xx > 0.0 ? sum_xy / sum_xx : 0.0;
} 