# Quantum RSI Trading Strategy with Maxwell's Demon Filter

This repository contains an implementation of an optimized Quantum RSI trading strategy that uses concepts from quantum mechanics to model market momentum and trend reversals, enhanced with a Maxwell's Demon filter to detect institutional activity.

## Overview

The Quantum RSI strategy applies concepts from quantum physics to trading:

1. **Quantum Model**: Treats RSI values as a quantum particle that can "tunnel" through barriers, representing a trend reversal.
2. **Double-Well Potential**: Maps RSI values to positions in a double-well potential, where wells represent bullish and bearish regimes.
3. **Maxwell's Demon Filter**: Uses information about volume, price-VWAP relationship, and order flow to selectively filter trading signals.

## Recent Improvements

Our latest optimizations have achieved a **66.67% win rate** with the Conservative parameter profile in backtesting:

- Replaced TA-Lib with pure Python 'ta' package for better compatibility
- Enhanced Maxwell's Demon filter for improved signal detection
- Optimized parameters for different risk profiles
- Added cooldown period and adaptive parameters after losses
- Improved risk-reward ratio with asymmetric stop-loss and take-profit

### Backtest Results (ETH/USDT 1h, Jan-July 2023)

| Parameter Set | Win Rate | Total Return | Profit Factor | Max Drawdown | Trades |
|---------------|----------|--------------|---------------|--------------|--------|
| Conservative  | 66.67%   | 1.17%        | 1.28          | 2.74%        | 6      |
| Balanced      | 47.06%   | -4.05%       | 0.75          | 6.99%        | 17     |
| Aggressive    | 58.00%   | -9.18%       | 0.82          | 17.83%       | 50     |

## Key Components

### 1. Quantum Parameters

- **Mass (m)**: Controls momentum persistence (higher = stronger trend inertia)
- **Reduced Planck Constant (ℏ)**: Controls uncertainty in the system
- **Potential Barrier Height (V₀)**: Sets resistance to trend reversal

### 2. Technical Indicators

- **RSI**: Main oscillator for momentum measurement
- **Volume Analysis**: Spike detection for whale activity
- **VWAP Distance**: Measures price extension
- **Delta Volume**: Detects buying/selling pressure

### 3. Maxwell's Demon Logic

The "demon" acts as an information filter that:
- Approves/rejects potential trade signals based on institutional footprints
- Adjusts barrier heights dynamically based on market conditions
- Prevents false reversal signals

## Strategy Logic

1. **Entry Conditions**:
   - RSI in extreme territory (overbought/oversold)
   - RSI beginning to reverse direction
   - High tunneling probability (quantum calculation)
   - Maxwell's Demon approval (institutional confirmation)

2. **Risk Management**:
   - Dynamic stop-loss based on ATR
   - Closer take-profit targets for higher win rate
   - Cooldown period after trades
   - Increasing selectivity after consecutive losses

## Parameter Profiles

The implementation offers three parameter profiles:

1. **Conservative**: High win rate, fewer trades
   - RSI thresholds: 22/78
   - Higher volume requirements
   - Wider stops for noise avoidance

2. **Balanced**: Default parameters
   - RSI thresholds: 30/70
   - Moderate volume requirements
   - Balanced risk/reward

3. **Aggressive**: More trading opportunities
   - RSI thresholds: 35/65
   - Lower volume requirements
   - Higher risk/reward

## Usage

1. **Run the web application**:
   ```
   python run_app.py
   ```
   Access the web interface at http://localhost:5000

2. **Run a backtest directly**:
   ```
   python backtest_eth.py
   ```

3. **Test with different parameters**:
   ```python
   from quantum_rsi_optimized import QuantumRSIParameters
   
   # Create custom parameters
   params = QuantumRSIParameters()
   params.rsi_high = 75
   params.rsi_low = 25
   params.mass = 2.0  # Higher momentum persistence
   ```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- ccxt (for data fetching)
- ta (for technical indicators)
- flask (for web interface)

## Mathematical Foundation

The strategy applies the time-dependent Schrödinger equation to model RSI dynamics:

```
iℏ ∂Ψ(x,t)/∂t = [-ℏ²/2m ∂²/∂x² + V(x) + δV_demon(x,t)]Ψ(x,t)
```

With tunneling probability calculated using the WKB approximation:

```
T ≈ exp(-2/ℏ ∫√(2m(V(x)-E))dx)
```

Where:
- Ψ(x,t) is the wavefunction (market state)
- V(x) is the double-well potential (trend resistance)
- δV_demon(x,t) is the Maxwell's Demon adjustment
- x is the position coordinate (mapped from RSI)

## Features

- Quantum-inspired RSI indicator with tunneling probabilities
- Maxwell's Demon filter for detecting significant market movements
- Volatility-adjusted potential barriers for dynamic risk assessment
- Multiple parameter sets for different trading styles
- Comprehensive backtesting framework

## Trading Logic

The strategy uses a quantum mechanical analogy where:
- Market trends are represented as particles in potential wells
- RSI values map to particle positions
- Market reversals are modeled as quantum tunneling events
- Maxwell's Demon monitors volume and price-VWAP relationships to filter signals

## Performance

Backtesting on ETH/USD (Jan-June 2023) produced the following results:

| Strategy | Win Rate | # Trades | Net % | Drawdown | Sharpe |
|----------|----------|----------|-------|----------|--------|
| ETH Aggressive | 41.76% | 91 | 6.02% | -1.97% | 1.75 |
| ETH Ultra-Conservative | 28.57% | 21 | -0.55% | -3.00% | -0.29 |
| ETH Standard | 31.71% | 41 | -1.33% | -3.59% | -0.62 |

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: numpy, pandas, matplotlib, ccxt, ta, flask

### Installation

```bash
# Clone the repository
git clone https://github.com/DeepFriedDog/quantumRSI.git
cd quantumRSI

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Known Issues

If you get a NumPy bool8 error, use the provided run_app.py script which automatically applies the NumPy patch:

```bash
python run_app.py
```

## License

MIT 