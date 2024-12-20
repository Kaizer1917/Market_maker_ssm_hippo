# Market Making with SSM-HIPPO

A sophisticated market making system that leverages State Space Models (SSM) with HiPPO operators for dynamic price prediction and market making strategies.

## Features

### Rollercoaster_girls.py Core Components

- **Predict Class**: Advanced prediction system for market making
  - Real-time price movement prediction using SSM-HIPPO
  - Dynamic state management for market conditions
  - Adaptive temporal coherence for price stability
  - Efficient forward-filling for time series alignment

### SSM Integration for Market Making

- **Dynamic State Tracking**:
  - Continuous monitoring of market state variables
  - Adaptive transition matrices for varying market conditions
  - Real-time state updates with new market data

- **Price Prediction Features**:
  - Multi-horizon forecasting capabilities
  - Channel attention for relevant feature selection
  - Patch-based processing for varying timeframes
  - Temporal coherence loss for stable predictions

- **Market Making Optimization**:
  - Dynamic bid-ask spread adjustment
  - State-aware position management
  - Risk-adjusted order sizing
  - Adaptive regularization for market volatility

## Installation

Required dependencies:
```bash
pip install torch einops numpy pandas polars scikit-learn tvm
```

## Usage

### Basic Usage

```python
from market_maker.Rollercoasters_girls import Predict

# Initialize predictor
predictor = Predict(
    lookback_window=96,
    forecast_horizon=32,
    num_features=24
)

# Make predictions
predictions = predictor.forward(market_data)
```

### Advanced Configuration

```python
# Configure SSM-HIPPO parameters
model_args = ModelArgs(
    d_model=128,          # Model dimension
    n_layer=4,            # Number of SSM layers
    seq_len=96,           # Input sequence length
    forecast_len=32,      # Prediction horizon
    num_channels=24,      # Number of market features
    patch_len=16,         # Patch size for processing
    stride=8              # Stride for patch sampling
)

# Initialize with custom configuration
predictor = Predict(
    model_args=model_args,
    use_tvm_optimization=True,
    adaptive_regularization=True
)
```

## Market Making Integration

The system is designed to work with market making strategies by providing:

1. **Real-time Predictions**:
   - Forward-filling missing data points
   - Handling irregular time series
   - Fast inference with TVM optimization

2. **State Management**:
   - Tracking market regimes
   - Managing position exposure
   - Monitoring risk metrics

3. **Adaptive Features**:
   - Dynamic learning rate adjustment
   - Adaptive regularization
   - Progressive state space expansion

## Performance Optimization

The implementation includes several optimizations:

- TVM acceleration for inference
- Efficient memory management
- Vectorized operations
- Adaptive computation paths

## Contributing

Feel free to submit issues and enhancement requests!

## C++ Implementation (cpp/)

### Core Components

1. **Market Making Engine**:
   - `MarketPredictor`: SSM-HIPPO based prediction system
   - `OrderManager`: Thread-safe order and position management
   - `RiskManager`: Real-time risk monitoring and limits
   - `StrategyManager`: Multi-threaded strategy execution

2. **Model Architecture**:
   - SSM-HIPPO layers with optimized C++ implementation
   - TVM integration for accelerated inference
   - Dynamic HiPPO transition matrices
   - Adaptive temporal coherence loss

3. **Performance Features**:
   - Lock-free data structures and queues
   - SIMD-optimized operations
   - Custom memory pooling
   - Thread pool for parallel processing

### Building and Installation

#### Requirements
- CMake 3.15+
- C++17 compliant compiler
- LibTorch
- TVM (optional)
- Boost

#### Build Instructions
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Usage Example

```cpp
#include "cpp/predictor.h"
#include "cpp/strategy.h"

int main() {
    // Configure the model
    SSMHippoConfig config{
        .d_model = 128,
        .n_layer = 4,
        .seq_len = 96,
        .d_state = 16,
        .num_channels = 24,
        .forecast_len = 32
    };

    // Initialize predictor and strategy
    auto predictor = std::make_shared<MarketPredictor>(config);
    auto order_manager = std::make_shared<OrderManager>();
    auto risk_manager = std::make_shared<RiskManager>();
    
    StoikovStrategy strategy(predictor, order_manager, risk_manager);

    // Start market making
    strategy.on_market_data(market_depth);
    return 0;
}
```

### Performance Optimizations

1. **Computational Efficiency**:
   - SIMD vectorization
   - Cache-friendly data structures
   - Zero-copy operations

2. **Memory Management**:
   - Custom allocators
   - Memory pools
   - Smart pointer optimization

3. **Concurrency**:
   - Thread pool execution
   - Lock-free algorithms
   - Atomic operations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
