#include "advanced_analytics.h"
#include "order_book_simulator.h"
#include <cmath>
#include <algorithm>
#include <execution>

AdvancedAnalytics::OrderBookMetrics AdvancedAnalytics::analyze_order_book(
    const stable_vector<MarketMicrostructure::OrderBookSnapshot>& snapshots) {
    
    OrderBookMetrics metrics;
    
    // Calculate spread distribution
    std::vector<double> spreads;
    spreads.reserve(snapshots.size());
    
    for (const auto& snapshot : snapshots) {
        if (!snapshot.asks.empty() && !snapshot.bids.empty()) {
            spreads.push_back(snapshot.asks[0].price - snapshot.bids[0].price);
        }
    }
    
    // Calculate mean and std of spreads
    metrics.spread_distribution_mean = std::accumulate(
        spreads.begin(), spreads.end(), 0.0) / spreads.size();
    
    double sq_sum = std::transform_reduce(
        std::execution::par_unseq,
        spreads.begin(), spreads.end(),
        0.0,
        std::plus<>(),
        [mean = metrics.spread_distribution_mean](double x) {
            return (x - mean) * (x - mean);
        }
    );
    metrics.spread_distribution_std = std::sqrt(sq_sum / spreads.size());
    
    // Calculate price level density
    auto calculate_density = [](const auto& levels) {
        double total_volume = 0.0;
        double weighted_price_sum = 0.0;
        
        for (const auto& level : levels) {
            total_volume += level.volume;
            weighted_price_sum += level.price * level.volume;
        }
        
        return total_volume > 0 ? weighted_price_sum / total_volume : 0.0;
    };
    
    double avg_bid_density = 0.0;
    double avg_ask_density = 0.0;
    
    for (const auto& snapshot : snapshots) {
        avg_bid_density += calculate_density(snapshot.bids);
        avg_ask_density += calculate_density(snapshot.asks);
    }
    
    metrics.price_level_density = (avg_bid_density + avg_ask_density) / 
                                (2.0 * snapshots.size());
    
    // Calculate volume concentration
    auto calculate_concentration = [](const auto& levels) {
        if (levels.empty()) return 0.0;
        
        double total_volume = std::accumulate(
            levels.begin(), levels.end(), 0.0,
            [](double sum, const auto& level) { return sum + level.volume; }
        );
        
        double top_level_volume = levels[0].volume;
        return total_volume > 0 ? top_level_volume / total_volume : 0.0;
    };
    
    metrics.volume_concentration = 0.0;
    for (const auto& snapshot : snapshots) {
        metrics.volume_concentration += (
            calculate_concentration(snapshot.bids) +
            calculate_concentration(snapshot.asks)
        ) * 0.5;
    }
    metrics.volume_concentration /= snapshots.size();
    
    // Calculate order book resiliency
    metrics.resiliency_factor = calculate_resiliency(snapshots);
    
    return metrics;
}

double AdvancedAnalytics::calculate_resiliency(
    const stable_vector<MarketMicrostructure::OrderBookSnapshot>& snapshots) {
    
    if (snapshots.size() < 2) return 0.0;
    
    std::vector<double> imbalance_changes;
    std::vector<double> spread_changes;
    
    for (size_t i = 1; i < snapshots.size(); ++i) {
        double prev_imbalance = snapshots[i-1].calculate_imbalance();
        double curr_imbalance = snapshots[i].calculate_imbalance();
        imbalance_changes.push_back(curr_imbalance - prev_imbalance);
        
        double prev_spread = snapshots[i-1].asks[0].price - snapshots[i-1].bids[0].price;
        double curr_spread = snapshots[i].asks[0].price - snapshots[i].bids[0].price;
        spread_changes.push_back(curr_spread - prev_spread);
    }
    
    // Calculate correlation between imbalance changes and spread changes
    double correlation = 0.0;
    if (!imbalance_changes.empty() && !spread_changes.empty()) {
        double imb_mean = std::accumulate(
            imbalance_changes.begin(),
            imbalance_changes.end(),
            0.0
        ) / imbalance_changes.size();
        
        double sprd_mean = std::accumulate(
            spread_changes.begin(),
            spread_changes.end(),
            0.0
        ) / spread_changes.size();
        
        double covariance = 0.0;
        double imb_var = 0.0;
        double sprd_var = 0.0;
        
        for (size_t i = 0; i < imbalance_changes.size(); ++i) {
            double imb_diff = imbalance_changes[i] - imb_mean;
            double sprd_diff = spread_changes[i] - sprd_mean;
            
            covariance += imb_diff * sprd_diff;
            imb_var += imb_diff * imb_diff;
            sprd_var += sprd_diff * sprd_diff;
        }
        
        if (imb_var > 0 && sprd_var > 0) {
            correlation = covariance / std::sqrt(imb_var * sprd_var);
        }
    }
    
    return correlation;
} 