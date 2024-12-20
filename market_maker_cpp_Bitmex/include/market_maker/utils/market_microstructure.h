#pragma once

#include "market_data.h"
#include <deque>
#include <unordered_map>

class MarketMicrostructure {
public:
    struct MicrostructureMetrics {
        double effective_spread;
        double realized_spread;
        double price_impact;
        double order_book_imbalance;
        double flow_toxicity;
        double vpin;  // Volume-synchronized Probability of Informed Trading
        double kyle_lambda;  // Price impact coefficient
        double hasbrouck_info_share;
    };
    
    struct FlowMetrics {
        double buy_volume;
        double sell_volume;
        double buy_initiated_trades;
        double sell_initiated_trades;
        double order_to_trade_ratio;
        double cancel_to_trade_ratio;
        double avg_trade_size;
        double avg_life_time;
    };
    
    class OrderBookSnapshot {
    public:
        struct BookLevel {
            double price;
            double volume;
            size_t order_count;
            std::chrono::nanoseconds update_time;
        };
        
        std::vector<BookLevel> bids;
        std::vector<BookLevel> asks;
        std::chrono::nanoseconds timestamp;
        
        double get_weighted_midprice(size_t levels = 5) const;
        double calculate_imbalance(size_t levels = 5) const;
    };
    
    void update(const MarketDepth& depth, const Order& order);
    MicrostructureMetrics calculate_metrics(
        const stable_vector<OrderBookSnapshot>& snapshots,
        const stable_vector<Order>& trades
    );
    
private:
    static constexpr size_t HISTORY_SIZE = 1000;
    std::deque<OrderBookSnapshot> book_history_;
    std::unordered_map<int64_t, std::chrono::nanoseconds> order_timestamps_;
    
    // Helper methods
    double calculate_vpin(const stable_vector<Order>& trades);
    double estimate_kyle_lambda(
        const stable_vector<OrderBookSnapshot>& snapshots,
        const stable_vector<Order>& trades
    );
    double calculate_flow_toxicity(const stable_vector<Order>& trades);
}; 