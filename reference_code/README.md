# Quantum RSI Trading Strategy

A cryptocurrency trading strategy that leverages quantum mechanics concepts to model and predict price movements through an enhanced RSI indicator.

## Overview

This project implements a trading strategy based on:

1. **Quantum Tunneling Analogy**: Using time-dependent particle-in-a-potential well system to model market trend reversals
2. **Maxwell's Demon Filter**: An information-based filter that detects institutional activity (whales)
3. **Enhanced RSI**: A 21-period RSI with asymmetric thresholds (70/30)

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
- Required packages: numpy, pandas, matplotlib, ccxt, backtesting

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

### Running Backtests

```bash
python quantum_crypto_backtest.py
```

## License

MIT 