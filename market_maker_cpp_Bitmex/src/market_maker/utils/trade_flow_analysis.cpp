#include "advanced_analytics.h"

AdvancedAnalytics::TradeFlowAnalysis AdvancedAnalytics::analyze_trade_flow(
    const stable_vector<Order>& trades,
    const stable_vector<MarketMicrostructure::OrderBookSnapshot>& snapshots) {
    
    TradeFlowAnalysis analysis;
    
    // Calculate trade size distribution
    std::vector<double> trade_sizes;
    trade_sizes.reserve(trades.size());
    
    for (const auto& trade : trades) {
        trade_sizes.push_back(trade.quantity);
    }
    
    // Calculate distribution statistics
    std::sort(trade_sizes.begin(), trade_sizes.end());
    size_t n = trade_sizes.size();
    
    analysis.avg_trade_size = std::accumulate(
        trade_sizes.begin(), trade_sizes.end(), 0.0) / n;
    
    // Calculate skewness
    double m2 = 0.0, m3 = 0.0;
    for (double size : trade_sizes) {
        double delta = size - analysis.avg_trade_size;
        m2 += delta * delta;
        m3 += delta * delta * delta;
    }
    m2 /= n;
    m3 /= n;
    
    analysis.trade_size_skewness = m3 / std::pow(m2, 1.5);
    
    // Calculate price impact matrix
    analysis.price_impact_matrix = calculate_price_impact_matrix(trades, snapshots);
    
    // Calculate impact decay curve
    analysis.impact_decay_curve = std::vector<double>(20, 0.0);
    for (size_t i = 0; i < trades.size(); ++i) {
        const auto& trade = trades[i];
        double base_price = snapshots[i].get_weighted_midprice();
        
        for (size_t j = 1; j <= 20 && i + j < snapshots.size(); ++j) {
            double price_change = snapshots[i + j].get_weighted_midprice() - base_price;
            analysis.impact_decay_curve[j-1] += std::abs(price_change);
        }
    }
    
    // Normalize decay curve
    for (double& impact : analysis.impact_decay_curve) {
        impact /= trades.size();
    }
    
    return analysis;
} 