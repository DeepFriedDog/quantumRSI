# Quantum RSI Trading System - Usage Guide

This guide provides step-by-step instructions for using the Quantum RSI trading system in different modes.

## Quick Start

1. **Installation**
   ```bash
   git clone https://github.com/yourusername/quantum-rsi.git
   cd quantum-rsi
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run backtest with default parameters**
   ```bash
   python -m quantum_rsi.main --mode backtest --pairs ETH/USDT --dex uniswap --start-date 2023-01-01 --end-date 2023-01-31 --initial-balance 10000 --show-trades
   ```

3. **Visualize the quantum model**
   ```bash
   python quantum_visualizer.py --symbol BTC/USDT --timeframe 1h
   ```

## Backtesting

### Basic Backtest
```bash
python -m quantum_rsi.main --mode backtest --pairs ETH/USDT --dex uniswap --start-date 2023-01-01 --end-date 2023-01-31 --initial-balance 10000
```

### Backtest with Custom Parameters
```bash
python -m quantum_rsi.main --mode backtest --pairs ETH/USDT --dex uniswap --start-date 2023-01-01 --end-date 2023-01-31 --initial-balance 10000 --v0 1.8 --a 0.45 --rsi-period 14 --rsi-high 65 --rsi-low 35 --volume-threshold 2.5 --show-trades --save-results
```

### Multi-Pair Backtest
```bash
python -m quantum_rsi.main --mode backtest --pairs ETH/USDT,BTC/USDT --dex uniswap --start-date 2023-01-01 --end-date 2023-01-31 --initial-balance 10000 --show-trades
```

## Live Trading

### Paper Trading Mode (Testnet)
```bash
python quantum_rsi_live_trader.py
```
Note: By default, the script uses testnet mode. Edit the script to set your API keys.

### Custom Parameters for Live Trading
Edit the `config` dictionary in `quantum_rsi_live_trader.py`:
```python
config = {
    # Strategy parameters
    'rsi_period': 14,
    'rsi_high': 65,          # Lower than standard for crypto markets
    'rsi_low': 35,           # Higher than standard for crypto markets
    
    # Quantum model parameters
    'v0': 1.5,               # Potential strength
    'a': 0.5,                # Well position parameter
    
    # Risk management
    'stop_loss_atr': 2.0,    # Stop loss multiplier based on ATR
    'take_profit_atr': 4.0,  # Take profit multiplier based on ATR
    'cooldown_bars': 6,      # Bars to wait after a trade
}
```

## Visualizations

### Static Quantum Model Visualization
```bash
python quantum_visualizer.py --symbol ETH/USDT --timeframe 1d --v0 1.5 --a 0.5
```

### Create Animation of Quantum Evolution
```bash
python quantum_visualizer.py --symbol BTC/USDT --timeframe 4h --animate --save btc_quantum_evolution.gif
```

### Custom Visualization Parameters
```bash
python quantum_visualizer.py --symbol ETH/USDT --timeframe 1h --limit 200 --v0 2.0 --a 0.4
```

## Parameter Tuning Guide

### Quantum Model Parameters

| Parameter | Description | Default | Conservative | Aggressive |
|-----------|-------------|---------|-------------|------------|
| v0        | Potential strength | 1.5 | 2.0 | 1.0 |
| a         | Well position | 0.5 | 0.4 | 0.6 |
| rsi_period | RSI calculation period | 14 | 14 | 7 |
| rsi_high | Overbought threshold | 65 | 70 | 60 |
| rsi_low | Oversold threshold | 35 | 30 | 40 |

### Risk Management Parameters

| Parameter | Description | Default | Conservative | Aggressive |
|-----------|-------------|---------|-------------|------------|
| stop_loss_atr | Stop loss ATR multiplier | 2.0 | 1.5 | 3.0 |
| take_profit_atr | Take profit ATR multiplier | 4.0 | 3.0 | 6.0 |
| max_position_size | Max position size (% of portfolio) | 0.03 | 0.02 | 0.05 |

### Maxwell's Demon Parameters

| Parameter | Description | Default | More Selective | Less Selective |
|-----------|-------------|---------|----------------|----------------|
| volume_threshold | Volume spike detection | 2.0 | 3.0 | 1.5 |
| vwap_period | VWAP calculation period | 20 | 30 | 10 |

## Troubleshooting

1. **No data available for backtest**
   - Check your date range
   - Verify the trading pair exists on the selected DEX
   - Make sure you have internet connection for API access

2. **Authentication errors in live trading**
   - Verify your API keys are correctly set
   - Check exchange access permissions
   - Confirm API keys have trading permissions

3. **Visualization errors**
   - Ensure you have all required packages installed
   - Try with a smaller data sample using `--limit` parameter
   - Check if you have enough memory for computation

## Examples of Results Interpretation

### Backtest Results
```
Period: 2023-01-01 to 2023-01-31
Initial Balance: $10,000.00
Final Equity: $10,350.74
Net Profit: $350.74 (3.51%)
Total Trades: 23
Win Rate: 52.17%
Profit Factor: 1.52
Max Drawdown: 2.41%
Sharpe Ratio: 1.78
```

This shows a profitable strategy with positive Sharpe ratio and reasonable drawdown.

### Quantum State Analysis

When viewing the quantum visualizer:
- The double-well potential represents the RSI landscape
- Probability density (red curve) shows where the RSI is likely to move
- Tunneling probability indicates likelihood of trend reversal
- Areas where probability density crosses the barrier suggest potential breakouts

## Contacting Support

If you encounter any issues or have questions, please:
1. Check the README and this guide
2. Look for similar issues in the GitHub repository
3. Create a new issue with detailed information about your problem

# Quantum RSI Trading Strategy Usage Guide

## Introduction

This guide provides instructions for using the Quantum RSI Trading Strategy for cryptocurrency markets. The Quantum RSI strategy is a sophisticated approach that combines quantum mechanical principles with traditional technical analysis to identify high-probability trading opportunities.

## Overview

The Quantum RSI strategy applies quantum mechanical concepts to model market behavior:

1. **RSI as a Quantum Particle**: The strategy treats RSI as a quantum particle in a double-well potential.
2. **Tunneling Probability**: Uses quantum tunneling principles to predict trend reversals.
3. **Maxwell's Demon Filter**: Applies an information-theoretic filter to detect institutional activity patterns.

## Strategy Variants

The Quantum RSI framework includes multiple parameter sets for different trading styles:

### ETH Standard
Basic implementation with moderate parameters.

### ETH Aggressive
More frequent trades with relaxed entry criteria.

### ETH Ultra-Conservative
Highly selective signals with stricter confirmation requirements.

### ETH High Win-Rate
Optimized for maximizing win rate using extreme RSI thresholds (80/20).

### ETH Quantum Enhanced 
Focuses on precision entry signals with enhanced tunneling measurement.

### ETH Optimal (Research-Based)
Combines the best elements of Quantum Enhanced and Aggressive strategies to maximize win rate while maintaining good returns. Key features include:

1. **Enhanced RSI Reversal Detection**: Waits for a confirmation of trend reversal after RSI crosses threshold levels, as supported by academic research showing higher accuracy for this approach.

2. **Asymmetric Thresholds (78/22)**: Uses optimized thresholds that balance between standard (70/30) and extreme (80/20) levels.

3. **Strict Volume Confirmation**: Requires significant volume (2.0x average) to confirm institutional participation.

4. **Managed Risk-Reward**: Uses tight stop-losses (1.2x ATR) with modest profit targets (2.0x ATR) for higher probability of reaching targets.

5. **U-shaped Pattern Recognition**: Specifically looks for U-shaped reversals in RSI, which research shows are more reliable indicators of trend changes.

## Key Parameters Explained

### RSI Parameters
- `rsi_period`: Period for RSI calculation (14 standard, 21 for reduced noise)
- `rsi_high`: Overbought threshold (70-80)
- `rsi_low`: Oversold threshold (20-30)

### Quantum Model Parameters
- `v0`: Potential barrier height (higher = more selective)
- `a`: Barrier width parameter
- `mass`: Particle mass (higher = less tunneling)
- `hbar`: Quantum constant analog (higher = more tunneling)

### Maxwell's Demon Parameters
- `volume_threshold`: Multiple of average volume required
- `vwap_deviation`: Minimum deviation from VWAP required
- `cooldown_period`: Hours to wait after a trade

## Usage Instructions

1. Install dependencies with `pip install -r requirements.txt`
2. Run a backtest with `python quantum_crypto_backtest.py`
3. To modify parameters, edit the parameter dictionaries in the main function

## Backtesting Results

The optimal strategy has achieved the following results in backtesting:
- Significantly higher win rate than standard RSI approaches
- Balanced risk-reward profile
- Reduced drawdown compared to more aggressive strategies
- Better performance during volatile market conditions

## Research Basis

The strategy is based on several academic papers:
1. "The Accuracy of Relative Strength Index (RSI) Indicator in Forecasting Foreign Exchange Price Movement" (Rossa & Mahyuni, 2022)
2. "Escape of particles in a time-dependent potential well" (Costa et al., 2011)

These papers provide evidence for:
- Higher accuracy of extreme RSI thresholds (80/20 vs 70/30)
- Importance of waiting for reversal confirmation after threshold crossing
- U-shaped pattern recognition in oscillator indicators
- Time-dependent barrier models for price movement

## Live Trading Considerations

For live trading:
1. Set appropriate risk parameters (max 1% risk per trade)
2. Use the Optimal strategy for highest win rate
3. Implement proper position sizing based on account size
4. Monitor and adjust parameters based on market conditions

## Disclaimer

This strategy is for educational and research purposes only. Past performance does not guarantee future results. Always use proper risk management techniques.

## Contact

For questions or support, please refer to the GitHub repository issues section. 