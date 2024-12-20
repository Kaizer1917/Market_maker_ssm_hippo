#pragma once

#include "stoikov_strategy.h"
#include "thread_pool.h"
#include <unordered_map>
#include <chrono>

class StrategyManager {
public:
    struct StrategyState {
        bool is_active{true};
        int error_count{0};
        std::chrono::steady_clock::time_point last_error;
        std::chrono::steady_clock::time_point recovery_time;
    };

    explicit StrategyManager(size_t num_threads = std::thread::hardware_concurrency())
        : thread_pool_(num_threads) {}
    
    void add_strategy(
        const std::string& symbol,
        std::shared_ptr<MarketMakingStrategy> strategy) {
        std::lock_guard<std::mutex> lock(strategies_mutex_);
        strategies_[symbol] = strategy;
    }
    
    void on_market_data(const std::string& symbol, const MarketDepth& depth) {
        auto strategy = get_strategy(symbol);
        if (strategy && is_strategy_healthy(symbol)) {
            thread_pool_.enqueue([this, strategy, symbol, depth] {
                try {
                    strategy->on_market_data(depth);
                } catch (const ExchangeError& e) {
                    handle_exchange_error(symbol, e);
                } catch (const std::exception& e) {
                    handle_strategy_error(symbol, e);
                }
            });
        }
    }
    
    void stop_all() {
        std::lock_guard<std::mutex> lock(strategies_mutex_);
        strategies_.clear();
    }
    
private:
    ThreadPool thread_pool_;
    std::mutex strategies_mutex_;
    std::unordered_map<std::string, std::shared_ptr<MarketMakingStrategy>> strategies_;
    std::unordered_map<std::string, StrategyState> strategy_states_;
    
    std::shared_ptr<MarketMakingStrategy> get_strategy(const std::string& symbol) {
        std::lock_guard<std::mutex> lock(strategies_mutex_);
        auto it = strategies_.find(symbol);
        return it != strategies_.end() ? it->second : nullptr;
    }
    
    bool is_strategy_healthy(const std::string& symbol) {
        std::lock_guard<std::mutex> lock(strategies_mutex_);
        auto& state = strategy_states_[symbol];
        
        auto now = std::chrono::steady_clock::now();
        if (state.error_count > MAX_ERRORS) {
            if (now - state.last_error > ERROR_RESET_PERIOD) {
                state.error_count = 0;
            } else {
                return false;
            }
        }
        return state.is_active;
    }
    
    void handle_exchange_error(const std::string& symbol, const ExchangeError& e) {
        std::lock_guard<std::mutex> lock(strategies_mutex_);
        auto& state = strategy_states_[symbol];
        state.error_count++;
        state.last_error = std::chrono::steady_clock::now();
        
        if (e.code() == ExchangeError::ErrorCode::CONNECTIVITY_LOST) {
            state.is_active = false;
            state.recovery_time = state.last_error + RECOVERY_DELAY;
        }
        
        // Log error and notify monitoring system
    }

    static constexpr int MAX_ERRORS = 3;
    static constexpr auto ERROR_RESET_PERIOD = std::chrono::minutes(5);
    static constexpr auto RECOVERY_DELAY = std::chrono::minutes(1);
}; 