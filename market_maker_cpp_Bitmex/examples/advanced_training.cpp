#include <market_maker/model/trainer.h>
#include <market_maker/model/data_utils.h>
#include <market_maker/model/metrics.h>
#include <market_maker/model/visualization.h>
#include <market_maker/backtest/backtest_engine.h>
#include <iostream>
#include <filesystem>

int main() {
    try {
        // Initialize model configuration
        ModelArgs args;
        args.d_model = 256;
        args.n_layer = 6;
        args.seq_len = 128;
        args.num_channels = 32;
        args.learning_rate = 0.001;
        args.num_epochs = 100;
        args.batch_size = 64;
        
        // Initialize data preprocessor
        DataPreprocessor preprocessor(args);
        
        // Load and preprocess training data
        torch::Tensor train_data = torch::randn({1000, args.num_channels, args.seq_len});
        torch::Tensor val_data = torch::randn({200, args.num_channels, args.seq_len});
        
        auto [train_x, train_y] = preprocessor.prepare_data(train_data, true);
        auto [val_x, val_y] = preprocessor.prepare_data(val_data, false);
        
        // Initialize model
        auto model = std::make_shared<SSMHippo>(args);
        
        // Initialize optimizer
        auto optimizer = std::make_shared<torch::optim::AdamW>(
            model->parameters(),
            torch::optim::AdamWOptions(args.learning_rate)
        );
        
        // Initialize trainer
        ModelTrainer trainer(args, model, optimizer);
        
        // Initialize visualizer
        ModelVisualizer::VisualizationConfig vis_config;
        vis_config.output_dir = "training_visualizations";
        vis_config.save_plots = true;
        ModelVisualizer visualizer(vis_config);
        
        // Training loop with visualization
        std::vector<float> train_losses;
        std::vector<float> val_losses;
        std::vector<MetricsCalculator::Metrics> metrics_history;
        
        trainer.train(train_data, val_data, "model_checkpoints/best_model.pt");
        
        // Visualize results
        visualizer.plot_loss_curve(train_losses, val_losses, "Training Progress");
        visualizer.plot_metrics_evolution(metrics_history, "Model Metrics");
        
        // Backtesting
        BacktestEngine::BacktestConfig backtest_config;
        backtest_config.initial_capital = 1000000;
        backtest_config.include_transaction_costs = true;
        backtest_config.include_slippage = true;
        
        auto strategy = std::make_shared<MarketMakingStrategy>(
            model,
            std::make_shared<OrderManager>(OrderManager::Config{}),
            std::make_shared<RiskManager>(RiskManager::Limits{})
        );
        
        BacktestEngine backtest(
            strategy,
            std::make_shared<RiskManager>(RiskManager::Limits{}),
            backtest_config
        );
        
        auto results = backtest.run();
        results.save_to_csv("backtest_results.csv");
        
        // Print final metrics
        std::cout << "Final Metrics:\n"
                  << "Sharpe Ratio: " << results.metrics.sharpe_ratio << "\n"
                  << "Max Drawdown: " << results.metrics.max_drawdown << "\n"
                  << "Total Return: " << results.metrics.total_return << "\n"
                  << "Win Rate: " << results.metrics.win_rate << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 