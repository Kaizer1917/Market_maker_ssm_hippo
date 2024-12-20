#pragma once

#include "market_maker_strategy.h"
#include <cmath>
#include <ctime>

class StoikovStrategy : public MarketMakingStrategy {
public:
    struct StoikovConfig : public Config {
        double risk_aversion{0.1};        // gamma in Bitmex/main.py
        double market_impact{1.5};        // k in Bitmex/main.py
        double volatility_window{100};
        double inventory_target{0.0};
        double time_horizon{1.0};         // T in days
        double drift{0.1};                // Price drift term
        double min_intensity{0.01};       // Minimum order intensity threshold
        double position_limit{10.0};      // Maximum position size
    };
    
    explicit StoikovStrategy(
        std::shared_ptr<MarketPredictor> predictor,
        std::shared_ptr<OrderManager> order_manager,
        StoikovConfig config)
        : MarketMakingStrategy(predictor, order_manager, config)
        , config_(config)
        , volatility_estimator_(config.volatility_window)
        , start_time_(std::time(nullptr)) {}
    
private:
    StoikovConfig config_;
    std::time_t start_time_;
    
    class VolatilityEstimator {
    public:
        explicit VolatilityEstimator(size_t window_size)
            : window_size_(window_size) {
            returns_.reserve(window_size);
        }
        
        void update(double price) {
            if (last_price_ > 0) {
                double ret = std::log(price / last_price_);
                if (returns_.size() >= window_size_) {
                    returns_.erase(returns_.begin());
                }
                returns_.push_back(ret);
            }
            last_price_ = price;
        }
        
        double get_volatility() const {
            if (returns_.size() < 2) return 0.0;
            
            double mean = std::accumulate(returns_.begin(), returns_.end(), 0.0) 
                         / returns_.size();
            
            double sq_sum = std::inner_product(
                returns_.begin(), returns_.end(),
                returns_.begin(), 0.0,
                std::plus<>(),
                [mean](double x, double y) { 
                    return (x - mean) * (y - mean); 
                }
            );
            
            return std::sqrt(sq_sum / (returns_.size() - 1)) * std::sqrt(252.0);
        }
        
    private:
        const size_t window_size_;
        double last_price_{0.0};
        std::vector<double> returns_;
    };
    
    VolatilityEstimator volatility_estimator_;
    
    // Thread-safe price/volatility updates
    std::mutex market_data_mutex_;
    std::condition_variable market_data_cv_;
    stable_vector<double> price_history_;
    
    // Stoikov-specific calculations
    double calculate_optimal_spread(double volatility, double inventory);
    std::pair<double, double> calculate_stoikov_quotes(
        double mid_price, 
        double volatility,
        double inventory
    );
    
    // New method for price simulation
    std::vector<double> simulate_price_path(
        double current_price,
        double volatility,
        int n_steps);
}; 