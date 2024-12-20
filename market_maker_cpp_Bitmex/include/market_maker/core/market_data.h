#pragma once

#include <atomic>
#include "stable_vector.h"

struct MarketDepth {
    struct Level {
        double price;
        double quantity;
        std::atomic<int64_t> update_time;
    };
    
    static constexpr size_t MAX_LEVELS = 20;
    std::array<Level, MAX_LEVELS> asks;
    std::array<Level, MAX_LEVELS> bids;
    std::atomic<int64_t> last_update{0};
    
    void update_ask(size_t level, double price, double qty);
    void update_bid(size_t level, double price, double qty);
    double get_mid_price() const;
    double get_spread() const;
};

class MarketDataBuffer {
public:
    explicit MarketDataBuffer(size_t capacity = 1024)
        : capacity_(capacity) {
        depth_buffer_.reserve(capacity);
    }
    
    void push_depth(const MarketDepth& depth) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (depth_buffer_.size() >= capacity_) {
            depth_buffer_.erase(depth_buffer_.begin());
        }
        depth_buffer_.push_back(depth);
    }
    
    stable_vector<MarketDepth> get_recent_depth(size_t n) const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t start = depth_buffer_.size() > n ? depth_buffer_.size() - n : 0;
        return stable_vector<MarketDepth>(
            depth_buffer_.begin() + start,
            depth_buffer_.end()
        );
    }

private:
    const size_t capacity_;
    mutable std::mutex mutex_;
    stable_vector<MarketDepth> depth_buffer_;
};
