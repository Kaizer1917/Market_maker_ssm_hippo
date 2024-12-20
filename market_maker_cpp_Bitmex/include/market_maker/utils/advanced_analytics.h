#pragma once

#include "market_microstructure.h"
#include "order_book_simulator.h"
#include <Eigen/Dense>

class AdvancedAnalytics {
public:
    struct OrderBookMetrics {
        // Price levels analysis
        double spread_distribution_mean;
        double spread_distribution_std;
        double price_level_density;
        double volume_concentration;
        
        // Order flow metrics
        double order_flow_imbalance;
        double tick_size_clustering;
        double price_impact_decay;
        double resiliency_factor;
        
        // Temporal metrics
        double update_frequency;
        double quote_lifetime;
        double cancel_rate_distribution;
        
        // Liquidity metrics
        double depth_weighted_spread;
        double market_depth_resilience;
        double liquidity_replenishment_rate;
    };
    
    struct TradeFlowAnalysis {
        // Trade size analysis
        Eigen::VectorXd trade_size_distribution;
        double avg_trade_size;
        double trade_size_skewness;
        
        // Trade timing
        std::vector<double> intraday_pattern;
        double trade_clustering_factor;
        double time_between_trades;
        
        // Price impact
        Eigen::MatrixXd price_impact_matrix;
        std::vector<double> impact_decay_curve;
        double permanent_impact_factor;
    };
    
    struct OrderLifetimeAnalysis {
        // Duration metrics
        std::vector<double> lifetime_distribution;
        double median_lifetime;
        double lifetime_variance;
        
        // Cancellation analysis
        double cancel_rate_by_distance;
        double modify_rate_by_distance;
        std::vector<double> queue_position_impact;
        
        // Execution probability
        Eigen::MatrixXd execution_probability_matrix;
        std::vector<double> fill_rate_by_size;
    };
    
    AdvancedAnalytics(size_t window_size = 1000)
        : window_size_(window_size) {}
    
    OrderBookMetrics analyze_order_book(
        const stable_vector<MarketMicrostructure::OrderBookSnapshot>& snapshots);
    
    TradeFlowAnalysis analyze_trade_flow(
        const stable_vector<Order>& trades,
        const stable_vector<MarketMicrostructure::OrderBookSnapshot>& snapshots);
    
    OrderLifetimeAnalysis analyze_order_lifetime(
        const stable_vector<Order>& orders,
        const stable_vector<MarketMicrostructure::OrderBookSnapshot>& snapshots);

private:
    const size_t window_size_;
    
    // Helper methods for order book analysis
    Eigen::MatrixXd calculate_price_impact_matrix(
        const stable_vector<Order>& trades,
        const stable_vector<MarketMicrostructure::OrderBookSnapshot>& snapshots);
        
    std::vector<double> compute_liquidity_curve(
        const MarketMicrostructure::OrderBookSnapshot& snapshot);
        
    double calculate_resiliency(
        const stable_vector<MarketMicrostructure::OrderBookSnapshot>& snapshots);
}; 