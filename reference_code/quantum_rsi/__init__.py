"""
Quantum RSI Trading System

A trading system that models RSI behavior using quantum mechanics principles,
applying a time-dependent particle in a potential well model to predict
trend reversals with Maxwell's Demon mechanism for signal filtering.
"""

__version__ = "0.1.0"

# Import core components for easier access
from quantum_rsi.core.quantum_model import (
    QuantumWellPotential,
    SchrodingerSolver,
    QuantumTunneling,
    MaxwellsDemon
)
from quantum_rsi.data.dex_data import DEXDataProvider, check_web3_availability
from quantum_rsi.trading.trader import QuantumRSITrader
from quantum_rsi.backtesting.backtester import QuantumRSIBacktester, Trade 