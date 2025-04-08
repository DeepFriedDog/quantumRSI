import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime, timedelta
import matplotlib
# Use non-interactive backend for matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import traceback

# Add numpy bool8 patch to avoid AttributeError
np.bool8 = np.bool_ if not hasattr(np, 'bool8') else np.bool8

# Import the optimized implementation
from quantum_rsi_optimized import (
    QuantumRSIParameters,
    calculate_indicators,
    compute_tunneling_probability,
    maxwell_demon_filter,
    fetch_crypto_data,
    run_backtest
)

# Import the simple backtest functionality for compatibility with existing code
from simple_backtest import calculate_indicators as simple_calculate_indicators
from simple_backtest import run_simple_backtest, fetch_crypto_data as simple_fetch_data

# Import the quantum probability model
from quantum_probability_model import (
    QuantumProbabilityParameters,
    run_backtest as run_quantum_probability_backtest
)

# Import the quantum stock market model
from quantum_stock_model import (
    QuantumStockParameters,
    run_backtest as run_quantum_stock_backtest
)

# Import quantum models
from quantum_rsi_optimized import run_backtest as run_quantum_rsi_backtest
from quantum_probability_model import run_backtest as run_quantum_probability_backtest
from quantum_stock_model import run_backtest as run_quantum_stock_backtest
from quantum_oscillator_model import run_backtest as run_quantum_oscillator_backtest, QuantumOscillatorParameters
from quantum_anharmonic_oscillator_model import run_backtest as run_quantum_anharmonic_backtest, QuantumAnharmonicParameters
from quantum_tunneling_model import run_backtest as run_quantum_tunneling_backtest, QuantumTunnelingParameters

# Import the quantum hybrid strategy backtest function
from quantum_harmonic_wave_hybrid import run_quantum_hybrid_strategy_backtest

app = Flask(__name__)

# Function to run the simple backtest
def run_simple_strategy_backtest(symbol, timeframe, initial_capital, start_date, end_date, params):
    """
    Run a backtest using the simple backtesting implementation
    which doesn't rely on the problematic backtesting library
    """
    print(f"\nFetching data for {symbol} on {timeframe} timeframe from {start_date} to {end_date}...")
    
    # Use the fetch function from simple_backtest
    data = simple_fetch_data(symbol, timeframe, start_date=start_date, end_date=end_date)
    data = simple_calculate_indicators(data)
    
    print(f"Data fetched: {len(data)} candles from {data.index[0]} to {data.index[-1]}")
    
    # Run the simple backtest
    strategy_params = {
        'initial_capital': initial_capital,
        **params  # Unpack all the other parameters
    }
    
    print(f"Running simple backtest for {symbol} with quantum RSI strategy...")
    stats = run_simple_backtest(data, strategy_params, symbol=symbol, timeframe=timeframe)
    
    print(f"\n----- Simple Backtest Results for {symbol} -----")
    print(f"Total Trades: {stats['Total Trades']}")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"Net Profit: ${stats['Net Profit']:.2f} ({stats['Return [%]']:.2f}%)")
    print(f"Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")
    print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    
    # Save the trades to a CSV file
    trades_df = pd.DataFrame(stats['Trades'])
    csv_filename = f"{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if len(trades_df) > 0:
        trades_df.to_csv(f"static/results/{csv_filename}", index=False)
        print(f"Trade log saved to static/results/{csv_filename}")
    
    return stats, csv_filename

# Function to run the optimized Quantum RSI backtest
def run_optimized_strategy_backtest(symbol, timeframe, initial_capital, start_date, end_date, params):
    """
    Run a backtest using the optimized Quantum RSI implementation
    with Maxwell's Demon filter for improved win rate
    """
    print(f"\nFetching data for {symbol} on {timeframe} timeframe from {start_date} to {end_date}...")
    
    # Use the fetch function from quantum_rsi_optimized
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    print(f"Data fetched: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Run the optimized backtest
    results = run_backtest(df, params, initial_capital=initial_capital, plot=False)
    
    print(f"\n----- Optimized Quantum RSI Backtest Results for {symbol} -----")
    print(f"Total Trades: {results['trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    # Save the trades to a CSV file
    trade_df = pd.DataFrame(results['trade_log'])
    csv_filename = f"{symbol.replace('/', '_')}_{timeframe}_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if not trade_df.empty:
        trade_df.to_csv(f"static/results/{csv_filename}", index=False)
        print(f"Trade log saved to static/results/{csv_filename}")
    
    # Format the results in the expected format for compatibility
    stats = {
        'Total Trades': results['trades'],
        'Win Rate [%]': results['win_rate'],
        'Return [%]': results['total_return'],
        'Net Profit': initial_capital * results['total_return'] / 100,
        'Max Drawdown [%]': results['max_drawdown'] * 100,
        'Profit Factor': results['profit_factor'],
        'Trades': results['trade_log'],
        'Equity Curve': results['equity_curve']
    }
    
    return stats, csv_filename

# Add function to run the quantum probability backtest
def run_quantum_probability_strategy_backtest(symbol, timeframe, initial_capital, start_date, end_date, params):
    """
    Run a backtest using the Quantum Probability Theoretic model
    """
    print(f"\nFetching data for {symbol} on {timeframe} timeframe from {start_date} to {end_date}...")
    
    # Use the fetch function from quantum_probability_model
    from quantum_probability_model import fetch_crypto_data
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    print(f"Data fetched: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Run the quantum probability backtest
    results = run_quantum_probability_backtest(df, params, initial_capital=initial_capital, plot=False)
    
    print(f"\n----- Quantum Probability Model Backtest Results for {symbol} -----")
    print(f"Total Trades: {results['trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    # Save the trades to a CSV file
    trade_df = pd.DataFrame(results['trade_log'])
    csv_filename = f"{symbol.replace('/', '_')}_{timeframe}_quantum_prob_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if not trade_df.empty:
        trade_df.to_csv(f"static/results/{csv_filename}", index=False)
        print(f"Trade log saved to static/results/{csv_filename}")
    
    # Format the results in the expected format for compatibility
    stats = {
        'Total Trades': results['trades'],
        'Win Rate [%]': results['win_rate'],
        'Return [%]': results['total_return'],
        'Net Profit': initial_capital * results['total_return'] / 100,
        'Max Drawdown [%]': results['max_drawdown'] * 100,
        'Profit Factor': results['profit_factor'],
        'Trades': results['trade_log'],
        'Equity Curve': results['equity_curve']
    }
    
    return stats, csv_filename

# Add function to run the quantum stock model backtest
def run_quantum_stock_strategy_backtest(symbol, timeframe, initial_capital, start_date, end_date, params):
    """
    Run a backtest using the Quantum Stock Market model
    """
    print(f"\nFetching data for {symbol} on {timeframe} timeframe from {start_date} to {end_date}...")
    
    # Use the fetch function from quantum_stock_model
    from quantum_stock_model import fetch_crypto_data
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    print(f"Data fetched: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Run the quantum stock model backtest
    results = run_quantum_stock_backtest(df, params, initial_capital=initial_capital, plot=False)
    
    print(f"\n----- Quantum Stock Model Backtest Results for {symbol} -----")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    # Save the trades to a CSV file
    trade_df = pd.DataFrame(results['trade_log'])
    csv_filename = f"{symbol.replace('/', '_')}_{timeframe}_quantum_stock_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if not trade_df.empty:
        trade_df.to_csv(f"static/results/{csv_filename}", index=False)
        print(f"Trade log saved to static/results/{csv_filename}")
    
    # Format the results in the expected format for compatibility
    stats = {
        'Total Trades': results['total_trades'],
        'Win Rate [%]': results['win_rate'],
        'Return [%]': results['total_return'],
        'Net Profit': initial_capital * results['total_return'] / 100,
        'Max Drawdown [%]': results['max_drawdown'] * 100,
        'Profit Factor': results['profit_factor'],
        'Trades': results['trade_log'],
        'Equity Curve': results['equity']
    }
    
    return stats, csv_filename

def run_quantum_oscillator_strategy_backtest(symbol, timeframe, start_date, end_date, params):
    """Run a backtest with the Quantum Oscillator model"""
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    # --- Backtest Initialization ---
    initial_capital = 10000 # Assuming oscillator also needs a capital base for % calculations, even if not position sizing
    capital = initial_capital
    position = 0 # 1 for long, -1 for short, 0 for flat
    equity_curve = [initial_capital]
    trades = []
    entry_price = 0

    print(f"Starting Quantum Oscillator backtest with {len(df)} data points...")
    print(f"Using confidence threshold: {params.signal_threshold:.2f} for trade entry")

    # --- Backtest Loop ---
    print(f"Starting backtest loop through {len(df) - params.indicator_warmup} bars...")
    for i in range(params.indicator_warmup, len(df)):
        # <<< Early Stop Check >>>
        # Even though Oscillator doesn't explicitly track capital like others,
        # we simulate it to stop if hypothetical capital depleted.
        if capital <= 1.0:
            print(f"{df.index[i]} - Simulated capital depleted (${capital:.2f}). Stopping backtest early.")
            equity_curve.extend([capital] * (len(df) - i))
            break

        # Progress indicator
        if i % 500 == 0:
            print(f"Processing bar {i}/{len(df)} ({(i/len(df)*100):.1f}%)...")

        # Get signals and current price
        # ... (rest of signal checking and trade logic)

        # --- Update Equity Curve ---
        # Need to simulate equity change based on trades for the stop check
        # This part might be complex if the original oscillator doesn't track PnL
        # Assuming a simplified PnL tracking for the capital check:
        if position != 0:
             current_price = df['close'].iloc[i]
             pnl_ratio = (current_price - entry_price) / entry_price if position == 1 else (entry_price - current_price) / entry_price
             # Simulate capital change (this won't be accurate for performance metrics if not done properly)
             # This is primarily just to make the capital <= 0 check work.
             # Real PnL tracking should be implemented if needed for metrics.
             # capital *= (1 + pnl_ratio) # This simplistic update is likely wrong for real metrics
             equity_curve.append(equity_curve[-1] * (1 + pnl_ratio)) # Append simulated equity
        else:
             equity_curve.append(equity_curve[-1]) # No change if flat
        capital = equity_curve[-1] # Update capital for the check

    # ... (final position close and performance calculation) ...

def run_quantum_oscillator_improved_conservative_strategy_backtest(symbol, timeframe, start_date, end_date, params=None):
    """Run a backtest with the Improved Conservative Quantum Oscillator model
    
    This implementation follows David Orrell's paper "A Quantum Oscillator Model of Stock Markets"
    with optimized parameters for a conservative approach:
    
    - Higher mass parameter (reduces volatility impact)
    - Optimized hbar for better uncertainty handling
    - Market-specific frequency tuning
    - Improved signal generation with adaptive thresholds
    - Enhanced risk management
    """
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    # Create parameters if not provided
    if params is None:
        from quantum_oscillator_model import QuantumOscillatorParameters
        params = QuantumOscillatorParameters()
        
        # Conservative parameter set based on paper insights
        params.lambda_param = 0.4    # Energy level parameter (slightly higher than default)
        params.hbar = 0.85           # Lower uncertainty (more conservative)
        params.omega = 1/180         # Slower oscillations (approximately 180 days)
        
        # More conservative volatility estimate
        params.set_spread_from_volatility(0.010)  # Using 1% daily vol estimate
        
        # Conservative trade entry settings
        params.signal_threshold = 0.7    # Higher threshold for more certainty
        params.min_amplitude = 0.025     # Require larger moves 
        
        # Enhanced risk management
        params.stop_loss_atr = 1.4       # Tighter stop loss
        params.take_profit_atr = 1.7     # Higher take profit
        params.cooldown_period = 8       # More reasonable cooldown
    
    # Run the backtest
    from quantum_oscillator_model import run_backtest
    results = run_backtest(df, params)
    
    # Create a timestamp for the CSV filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"static/results/{symbol.replace('/', '_')}_{timeframe}_quantum_oscillator_improved_conservative_{timestamp}.csv"
    
    # Extract trade log and save to CSV if available
    if results and 'trade_log' in results and results['trade_log']:
        trade_df = pd.DataFrame(results['trade_log'])
        trade_df.to_csv(csv_filename, index=False)
    
    # Format the results in the expected format for compatibility with the UI
    formatted_stats = {
        'Total Trades': results.get('total_trades', 0),
        'Win Rate [%]': results.get('win_rate', 0),
        'Return [%]': results.get('total_return', 0),
        'Net Profit': results.get('total_return', 0) * df.iloc[0]['close'] / 100,  # Estimate based on initial price
        'Max Drawdown [%]': results.get('max_drawdown', 0) * 100,
        'Profit Factor': results.get('profit_factor', 0),
        'Trades': results.get('trade_log', []),
        'Equity Curve': results.get('equity', [])
    }
    
    return formatted_stats, csv_filename

def run_quantum_oscillator_improved_balanced_strategy_backtest(symbol, timeframe, start_date, end_date, params=None):
    """Run a backtest with the Improved Balanced Quantum Oscillator model
    
    This implementation follows David Orrell's paper "A Quantum Oscillator Model of Stock Markets"
    with optimized parameters for a balanced approach:
    
    - Moderate mass parameter to react to volatility appropriately
    - Paper-based hbar value for optimal uncertainty quantification
    - Harmonic frequency calibration for typical market cycles
    - Volatility-adjusted signal thresholds
    - Balanced risk/reward parameters
    """
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    # Create parameters if not provided
    if params is None:
        from quantum_oscillator_model import QuantumOscillatorParameters
        params = QuantumOscillatorParameters()
        
        # Balanced parameter set based on paper insights
        params.lambda_param = 0.5    # Energy level parameter (in paper 0.5 matches volatility patterns)
        params.hbar = 1.1            # Slightly higher uncertainty (better for capturing opportunities)
        params.omega = 1/120         # Standard oscillation (approximately 120 days)
        
        # Balanced volatility estimate
        params.set_spread_from_volatility(0.012)  # Using 1.2% daily vol estimate
        
        # Balanced trade entry settings
        params.signal_threshold = 0.6    # Moderate threshold for balanced certainty
        params.min_amplitude = 0.02      # Standard meaningful move 
        
        # Balanced risk management
        params.stop_loss_atr = 1.5       # Standard stop loss
        params.take_profit_atr = 1.5     # Balanced profit target
        params.cooldown_period = 5       # Moderate cooldown
    
    # Run the backtest
    from quantum_oscillator_model import run_backtest
    results = run_backtest(df, params)
    
    # Create a timestamp for the CSV filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"static/results/{symbol.replace('/', '_')}_{timeframe}_quantum_oscillator_improved_balanced_{timestamp}.csv"
    
    # Extract trade log and save to CSV if available
    if results and 'trade_log' in results and results['trade_log']:
        trade_df = pd.DataFrame(results['trade_log'])
        trade_df.to_csv(csv_filename, index=False)
    
    # Format the results in the expected format for compatibility with the UI
    formatted_stats = {
        'Total Trades': results.get('total_trades', 0),
        'Win Rate [%]': results.get('win_rate', 0),
        'Return [%]': results.get('total_return', 0),
        'Net Profit': results.get('total_return', 0) * df.iloc[0]['close'] / 100,  # Estimate based on initial price
        'Max Drawdown [%]': results.get('max_drawdown', 0) * 100,
        'Profit Factor': results.get('profit_factor', 0),
        'Trades': results.get('trade_log', []),
        'Equity Curve': results.get('equity', [])
    }
    
    return formatted_stats, csv_filename

def run_quantum_oscillator_improved_aggressive_strategy_backtest(symbol, timeframe, start_date, end_date, params=None):
    """Run a backtest with the Improved Aggressive Quantum Oscillator model
    
    This implementation follows David Orrell's paper "A Quantum Oscillator Model of Stock Markets"
    with optimized parameters for an aggressive approach:
    
    - Low mass parameter (higher volatility impact)
    - Higher hbar for capturing uncertainty-based opportunities
    - Faster frequency calibration for short-term market cycles
    - Dynamic signal thresholds based on energy levels
    - Aggressive risk/reward parameters
    """
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    # Create parameters if not provided
    if params is None:
        from quantum_oscillator_model import QuantumOscillatorParameters
        params = QuantumOscillatorParameters()
        
        # Aggressive parameter set based on paper insights
        params.lambda_param = 0.75   # Higher energy level parameter for more volatility
        params.hbar = 1.5            # Higher uncertainty (more aggressive)
        params.omega = 1/60          # Faster oscillations (approximately 60 days)
        
        # More aggressive volatility estimate
        params.set_spread_from_volatility(0.015)  # Using 1.5% daily vol estimate
        
        # Aggressive trade entry settings
        params.signal_threshold = 0.4    # Lower threshold for more opportunities
        params.min_amplitude = 0.015     # Lower required move size
        
        # Aggressive risk management
        params.stop_loss_atr = 1.0       # Tighter stop loss for better capital protection
        params.take_profit_atr = 2.0     # Higher take profit for better returns
        params.cooldown_period = 3       # Shorter cooldown
    
    # Run the backtest
    from quantum_oscillator_model import run_backtest
    results = run_backtest(df, params)
    
    # Create a timestamp for the CSV filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"static/results/{symbol.replace('/', '_')}_{timeframe}_quantum_oscillator_improved_aggressive_{timestamp}.csv"
    
    # Extract trade log and save to CSV if available
    if results and 'trade_log' in results and results['trade_log']:
        trade_df = pd.DataFrame(results['trade_log'])
        trade_df.to_csv(csv_filename, index=False)
    
    # Format the results in the expected format for compatibility with the UI
    formatted_stats = {
        'Total Trades': results.get('total_trades', 0),
        'Win Rate [%]': results.get('win_rate', 0),
        'Return [%]': results.get('total_return', 0),
        'Net Profit': results.get('total_return', 0) * df.iloc[0]['close'] / 100,  # Estimate based on initial price
        'Max Drawdown [%]': results.get('max_drawdown', 0) * 100,
        'Profit Factor': results.get('profit_factor', 0),
        'Trades': results.get('trade_log', []),
        'Equity Curve': results.get('equity', [])
    }
    
    return formatted_stats, csv_filename

def run_quantum_anharmonic_strategy_backtest(symbol, timeframe, start_date, end_date, params):
    """Run a backtest with the Quantum Anharmonic Oscillator model"""
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    # Run the backtest
    results = run_quantum_anharmonic_backtest(df, params)
    
    # Create a timestamp for the CSV filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"static/results/{symbol.replace('/', '_')}_{timeframe}_quantum_anharmonic_{timestamp}.csv"
    
    # Extract trade log and save to CSV if available
    if results and 'trade_log' in results and results['trade_log']:
        trade_df = pd.DataFrame(results['trade_log'])
        trade_df.to_csv(csv_filename, index=False)
    
    # Format the results in the expected format for compatibility with the UI
    formatted_stats = {
        'Total Trades': results.get('total_trades', 0),
        'Win Rate [%]': results.get('win_rate', 0),
        'Return [%]': results.get('total_return', 0),
        'Net Profit': results.get('total_return', 0) * df.iloc[0]['close'] / 100,  # Estimate based on initial price
        'Max Drawdown [%]': results.get('max_drawdown', 0) * 100,
        'Profit Factor': results.get('profit_factor', 0),
        'Trades': results.get('trade_log', []),
        'Equity Curve': results.get('equity', [])
    }
    
    return formatted_stats, csv_filename

def run_quantum_tunneling_strategy_backtest(symbol, timeframe, start_date, end_date, params):
    """Run a backtest with the Quantum Tunneling model"""
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    # Run the backtest
    results = run_quantum_tunneling_backtest(df, params)
    
    # Create a timestamp for the CSV filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"static/results/{symbol.replace('/', '_')}_{timeframe}_quantum_tunneling_{timestamp}.csv"
    
    # Extract trade log and save to CSV if available
    if results and 'trade_log' in results and results['trade_log']:
        trade_df = pd.DataFrame(results['trade_log'])
        trade_df.to_csv(csv_filename, index=False)
    
    # Format the results in the expected format for compatibility with the UI
    formatted_stats = {
        'Total Trades': results.get('total_trades', 0),
        'Win Rate [%]': results.get('win_rate', 0),
        'Return [%]': results.get('total_return', 0),
        'Net Profit': results.get('total_return', 0) * df.iloc[0]['close'] / 100,  # Estimate based on initial price
        'Max Drawdown [%]': results.get('max_drawdown', 0) * 100,
        'Profit Factor': results.get('profit_factor', 0),
        'Trades': results.get('trade_log', []),
        'Equity Curve': results.get('equity', [])
    }
    
    return formatted_stats, csv_filename

# Add function to run the quantum hybrid model backtest
def run_quantum_hybrid_strategy_backtest(symbol, timeframe, initial_capital, start_date, end_date, params=None):
    """
    Run a backtest using the Quantum Harmonic-Wave Hybrid model that combines
    the high win rate of the Conservative model with the high returns of the 
    Improved Aggressive model.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
        timeframe (str): Timeframe for the data (e.g., '1h', '4h')
        initial_capital (float): Initial capital for the backtest
        start_date (str): Start date for the backtest (YYYY-MM-DD)
        end_date (str): End date for the backtest (YYYY-MM-DD)
        params (QuantumHybridParameters, optional): Model parameters
        
    Returns:
        tuple: (stats, csv_filename) with backtest results and trade log filename
    """
    print(f"\nFetching data for {symbol} on {timeframe} timeframe from {start_date} to {end_date}...")
    
    # Import the hybrid model
    from quantum_stock_hybrid_model import QuantumHybridParameters, find_trading_signals
    
    # Use the fetch function from general utilities
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    if df is None or len(df) < 50:
        print(f"Insufficient data for {symbol} on {timeframe} timeframe")
        return None, None
    
    print(f"Data fetched: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Create default parameters if not provided
    if params is None:
        params = QuantumHybridParameters()
        
        # Adjust parameters based on market cap
        from improved_quantum_stock_testing import HIGH_MARKET_CAP, MEDIUM_MARKET_CAP, get_market_cap_based_mass
        base_mass = get_market_cap_based_mass(symbol)
        params.m = base_mass
        params.mass = base_mass
        
        # Adjust parameters based on timeframe
        if timeframe == '1h':
            params.omega = 1.1e-4  # Standard ~4-day cycle for 1h
        elif timeframe == '4h':
            params.omega = 2.5e-4  # Faster cycle for 4h
        elif timeframe == '1d':
            params.omega = 5.0e-4  # Even faster cycle for 1d
        elif timeframe == '15m':
            params.omega = 0.6e-4  # Slower cycle for 15m
        elif timeframe == '5m':
            params.omega = 0.3e-4  # Even slower cycle for 5m
        elif timeframe == '1m':
            params.omega = 0.1e-4  # Very slow cycle for 1m
    
    # Find trading signals
    df_with_signals = find_trading_signals(df, params)
    
    # Run backtest logic
    equity_curve = [initial_capital]
    trade_log = []
    current_position = 0
    entry_price = 0
    entry_time = None
    stop_loss = None
    take_profit = None
    
    # For performance metrics
    total_trades = 0
    winning_trades = 0
    total_profit_pct = 0
    total_loss_pct = 0
    max_equity = initial_capital
    min_equity = initial_capital
    current_equity = initial_capital
    max_drawdown = 0
    
    for i in range(1, len(df_with_signals)):
        current_time = df_with_signals.index[i]
        current_row = df_with_signals.iloc[i]
        previous_row = df_with_signals.iloc[i-1]
        
        # Process exit conditions if in a position
        if current_position != 0:
            # Check stop loss and take profit
            if current_position > 0:  # Long position
                # Check for stop loss hit
                if current_row['low'] <= stop_loss:
                    # Calculate profit/loss
                    profit_pct = (stop_loss / entry_price - 1) * 100
                    profit_amount = current_equity * profit_pct / 100
                    current_equity += profit_amount
                    
                    # Record the trade
                    trade_log.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'type': 'long',
                        'profit_pct': profit_pct,
                        'profit_amount': profit_amount,
                        'exit_reason': 'stop_loss'
                    })
                    
                    # Update metrics
                    total_trades += 1
                    if profit_pct > 0:
                        winning_trades += 1
                        total_profit_pct += profit_pct
                    else:
                        total_loss_pct += abs(profit_pct)
                    
                    # Reset position
                    current_position = 0
                    entry_price = 0
                    stop_loss = None
                    take_profit = None
                
                # Check for take profit hit
                elif current_row['high'] >= take_profit:
                    # Calculate profit/loss
                    profit_pct = (take_profit / entry_price - 1) * 100
                    profit_amount = current_equity * profit_pct / 100
                    current_equity += profit_amount
                    
                    # Record the trade
                    trade_log.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'type': 'long',
                        'profit_pct': profit_pct,
                        'profit_amount': profit_amount,
                        'exit_reason': 'take_profit'
                    })
                    
                    # Update metrics
                    total_trades += 1
                    if profit_pct > 0:
                        winning_trades += 1
                        total_profit_pct += profit_pct
                    else:
                        total_loss_pct += abs(profit_pct)
                    
                    # Reset position
                    current_position = 0
                    entry_price = 0
                    stop_loss = None
                    take_profit = None
                
            elif current_position < 0:  # Short position
                # Check for stop loss hit
                if current_row['high'] >= stop_loss:
                    # Calculate profit/loss
                    profit_pct = (entry_price / stop_loss - 1) * 100
                    profit_amount = current_equity * profit_pct / 100
                    current_equity += profit_amount
                    
                    # Record the trade
                    trade_log.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'type': 'short',
                        'profit_pct': profit_pct,
                        'profit_amount': profit_amount,
                        'exit_reason': 'stop_loss'
                    })
                    
                    # Update metrics
                    total_trades += 1
                    if profit_pct > 0:
                        winning_trades += 1
                        total_profit_pct += profit_pct
                    else:
                        total_loss_pct += abs(profit_pct)
                    
                    # Reset position
                    current_position = 0
                    entry_price = 0
                    stop_loss = None
                    take_profit = None
                
                # Check for take profit hit
                elif current_row['low'] <= take_profit:
                    # Calculate profit/loss
                    profit_pct = (entry_price / take_profit - 1) * 100
                    profit_amount = current_equity * profit_pct / 100
                    current_equity += profit_amount
                    
                    # Record the trade
                    trade_log.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'type': 'short',
                        'profit_pct': profit_pct,
                        'profit_amount': profit_amount,
                        'exit_reason': 'take_profit'
                    })
                    
                    # Update metrics
                    total_trades += 1
                    if profit_pct > 0:
                        winning_trades += 1
                        total_profit_pct += profit_pct
                    else:
                        total_loss_pct += abs(profit_pct)
                    
                    # Reset position
                    current_position = 0
                    entry_price = 0
                    stop_loss = None
                    take_profit = None
        
        # Process entry signals if not in a position
        if current_position == 0:
            # Buy signal
            if previous_row['signal'] == 1:
                current_position = 1
                entry_price = current_row['open']  # Enter at the open of the next candle
                entry_time = current_time
                stop_loss = previous_row['stop_loss']
                take_profit = previous_row['take_profit']
            
            # Sell signal
            elif previous_row['signal'] == -1:
                current_position = -1
                entry_price = current_row['open']  # Enter at the open of the next candle
                entry_time = current_time
                stop_loss = previous_row['stop_loss']
                take_profit = previous_row['take_profit']
        
        # Update equity curve
        equity_curve.append(current_equity)
        
        # Update max equity and drawdown
        if current_equity > max_equity:
            max_equity = current_equity
        
        drawdown = (max_equity - current_equity) / max_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            min_equity = current_equity
    
    # Close any open position at the end of the backtest
    if current_position != 0:
        last_price = df_with_signals['close'].iloc[-1]
        
        if current_position > 0:  # Long position
            profit_pct = (last_price / entry_price - 1) * 100
        else:  # Short position
            profit_pct = (entry_price / last_price - 1) * 100
        
        profit_amount = current_equity * profit_pct / 100
        current_equity += profit_amount
        
        # Record the trade
        trade_log.append({
            'entry_time': entry_time,
            'exit_time': df_with_signals.index[-1],
            'entry_price': entry_price,
            'exit_price': last_price,
            'type': 'long' if current_position > 0 else 'short',
            'profit_pct': profit_pct,
            'profit_amount': profit_amount,
            'exit_reason': 'end_of_backtest'
        })
        
        # Update metrics
        total_trades += 1
        if profit_pct > 0:
            winning_trades += 1
            total_profit_pct += profit_pct
        else:
            total_loss_pct += abs(profit_pct)
    
    # Calculate final metrics
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_return = ((current_equity - initial_capital) / initial_capital * 100)
    profit_factor = (total_profit_pct / total_loss_pct) if total_loss_pct > 0 else float('inf')
    
    # Save the trade log to a CSV file
    trade_df = pd.DataFrame(trade_log)
    csv_filename = f"{symbol.replace('/', '_')}_{timeframe}_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if not trade_df.empty:
        # Create the directory if it doesn't exist
        os.makedirs("static/results", exist_ok=True)
        trade_df.to_csv(f"static/results/{csv_filename}", index=False)
        print(f"Trade log saved to static/results/{csv_filename}")
    
    # Print results summary
    print(f"\n----- Quantum Hybrid Model Backtest Results for {symbol} -----")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # Format the results in the expected format for compatibility
    stats = {
        'Total Trades': total_trades,
        'Win Rate [%]': win_rate,
        'Return [%]': total_return,
        'Net Profit': initial_capital * total_return / 100,
        'Max Drawdown [%]': max_drawdown * 100,
        'Profit Factor': profit_factor,
        'Trades': trade_log,
        'Equity Curve': equity_curve
    }
    
    return stats, csv_filename

# List of supported cryptocurrencies
SUPPORTED_CRYPTOS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
    'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT',
    # Additional tokens for quantum strategies
    'ATOM/USDT', 'ALGO/USDT', 'LTC/USDT', 'AAVE/USDT', 'UNI/USDT', 
    'SNX/USDT', 'ZIL/USDT', 'THETA/USDT', 'ONE/USDT', 'FTM/USDT',
    'NEAR/USDT', 'ROSE/USDT', 'CHZ/USDT', 'ENJ/USDT', 'SAND/USDT',
    'RNDR/USDT', 'OCEAN/USDT', 'API3/USDT'
]

# List of supported timeframes
SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

# Dictionary to store backtest results
backtest_results = {}

@app.route('/')
def index():
    """Render the main page with backtest form"""
    # Fixed default dates for the H1 2024 period
    default_start_date = "2024-01-01"
    default_end_date = "2024-07-01"
    
    return render_template(
        'index.html',
        cryptos=SUPPORTED_CRYPTOS,
        timeframes=SUPPORTED_TIMEFRAMES,
        default_start_date=default_start_date,
        default_end_date=default_end_date
    )

@app.route('/run_backtest', methods=['POST'])
def run_backtest_endpoint():
    """Run a backtest with the provided parameters"""
    try:
        # Get parameters from form
        crypto = request.form.get('crypto', 'ETH/USDT')
        timeframe = request.form.get('timeframe', '1h')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        strategy_type = request.form.get('strategy_type', 'optimal')
        initial_capital = float(request.form.get('initial_capital', 10000))
        
        # Validate inputs
        if crypto not in SUPPORTED_CRYPTOS:
            return jsonify({'error': f'Unsupported crypto: {crypto}'}), 400
            
        if timeframe not in SUPPORTED_TIMEFRAMES:
            return jsonify({'error': f'Unsupported timeframe: {timeframe}'}), 400
        
        # Run the backtest based on selected strategy
        strategy_params = {}
        strategy_results = {}
        
        # Create the results directory if it doesn't exist
        os.makedirs('static/results', exist_ok=True)
        
        # New strategy types from the Quantum Probability model
        if strategy_type == "quantum_prob_conservative":
            # Conservative parameters - Normal distribution (Ground state)
            params = QuantumProbabilityParameters()
            params.alpha = 2.5         # Higher α parameters = stronger mean reversion
            params.delta = 0.4         # Lower δ = less influence of extreme returns
            params.h = 0.8             # Lower h = lower stochastic influence
            params.omega_level = 1     # Default to normal distribution
            params.stop_loss_atr = 1.8 # Wider stop loss
            params.take_profit_atr = 1.2 # Moderate take profit
            params.cooldown_period = 12  # Longer cooldown
            
            stats, csv_file = run_quantum_probability_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, params
            )
            strategy_results['Quantum Prob - Conservative'] = stats
            
        elif strategy_type == "quantum_prob_balanced":
            # Balanced parameters - Bimodal distribution (First excited state)
            params = QuantumProbabilityParameters()
            params.alpha = 2.0
            params.delta = 0.5
            params.h = 1.0
            params.omega_level = 3     # Use bimodal distribution
            params.stop_loss_atr = 1.5
            params.take_profit_atr = 1.3
            params.cooldown_period = 8
            
            stats, csv_file = run_quantum_probability_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, params
            )
            strategy_results['Quantum Prob - Balanced'] = stats
            
        elif strategy_type == "quantum_prob_aggressive":
            # Aggressive parameters - Trimodal distribution (Second excited state)
            params = QuantumProbabilityParameters()
            params.alpha = 1.5         # Lower α = less mean reversion
            params.delta = 0.8         # Higher δ = more influence of extreme returns
            params.h = 1.5             # Higher h = higher stochastic influence
            params.omega_level = 5     # Use trimodal distribution
            params.stop_loss_atr = 1.2 # Tighter stop loss
            params.take_profit_atr = 1.5 # Higher take profit
            params.cooldown_period = 5   # Shorter cooldown
            
            stats, csv_file = run_quantum_probability_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, params
            )
            strategy_results['Quantum Prob - Aggressive'] = stats
            
        # New strategy types from the Quantum Stock Market model
        elif strategy_type == "quantum_stock_conservative":
            # Conservative parameters - Lower mass, higher stability
            params = QuantumStockParameters()
            params.m = 2e-30         # Higher mass = less volatile
            params.hbar = 0.8        # Lower hbar = more certain
            params.F = 0.8e-19       # Lower field amplitude = less external influence
            params.omega = 0.8e-4    # Lower frequency = slower cycles
            params.stop_loss_atr = 1.8 # Wider stop loss
            params.take_profit_atr = 1.2 # Moderate take profit
            params.cooldown_period = 12  # Longer cooldown
            params.signal_threshold = 0.7 # Higher threshold for stronger signals
            
            stats, csv_file = run_quantum_stock_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, params
            )
            strategy_results['Quantum Stock - Conservative'] = stats
            
        elif strategy_type == "quantum_stock_balanced":
            # Balanced parameters - Default settings
            params = QuantumStockParameters()
            # Default settings are already balanced
            
            stats, csv_file = run_quantum_stock_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, params
            )
            strategy_results['Quantum Stock - Balanced'] = stats
            
        elif strategy_type == "quantum_stock_aggressive":
            # Aggressive parameters - Lower mass, more responsive
            params = QuantumStockParameters()
            params.m = 0.5e-30       # Lower mass = more volatility
            params.hbar = 1.2        # Higher hbar = more uncertainty
            params.F = 1.2e-19       # Higher field amplitude = more external influence
            params.omega = 1.2e-4    # Higher frequency = faster cycles
            params.stop_loss_atr = 1.2 # Tighter stop loss
            params.take_profit_atr = 1.5 # Higher take profit
            params.cooldown_period = 5   # Shorter cooldown
            params.signal_threshold = 0.5 # Lower threshold for more signals
            
            stats, csv_file = run_quantum_stock_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, params
            )
            strategy_results['Quantum Stock - Aggressive'] = stats
        
        # After the existing quantum_stock_aggressive option
        elif strategy_type == "quantum_stock_improved_conservative":
            params = QuantumStockParameters()
            # Enhanced quantum parameters
            params.m = 2.2e-30       # Increased mass parameter for more stability
            params.hbar = 0.85       # Slightly higher reduced Planck constant analog
            params.F = 1.1e-19       # Increased field amplitude for better signal detection
            params.omega = 0.8e-4    # Reduced frequency for better trend following
            
            # Enhanced trading parameters
            params.signal_threshold = 0.7  # Higher threshold for more selective entries
            params.min_amplitude = 0.025   # Higher amplitude requirement
            
            # Enhanced risk management
            params.stop_loss_atr = 1.4     # Tighter stop loss
            params.take_profit_atr = 1.8   # Higher take profit target
            params.cooldown_period = 8     # Shorter cooldown for more opportunities
            
            stats, csv_file = run_quantum_stock_improved_conservative_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, params
            )
            strategy_results['Quantum Stock - Improved Conservative'] = stats
            
        elif strategy_type == "quantum_stock_improved_balanced":
            params = QuantumStockParameters()
            # Enhanced quantum parameters
            params.m = 1.5e-30       # Balanced mass parameter
            params.hbar = 1.1        # Higher hbar for more quantum influence
            params.F = 1.0e-19       # Moderate field amplitude
            params.omega = 1.1e-4    # Slightly increased frequency for faster response
            
            # Enhanced trading parameters
            params.signal_threshold = 0.6  # Moderate threshold
            params.min_amplitude = 0.02    # Standard amplitude requirement
            
            # Enhanced risk management
            params.stop_loss_atr = 1.5     # Standard stop loss
            params.take_profit_atr = 1.5   # Balanced take profit target
            params.cooldown_period = 6     # Moderate cooldown
            
            stats, csv_file = run_quantum_stock_improved_balanced_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, params
            )
            strategy_results['Quantum Stock - Improved Balanced'] = stats
            
        elif strategy_type == "quantum_stock_improved_aggressive":
            params = QuantumStockParameters()
            # Enhanced quantum parameters
            params.m = 0.4e-30       # Lower mass for higher volatility
            params.hbar = 1.5        # Much higher hbar for stronger quantum effects
            params.F = 1.5e-19       # Higher field amplitude for stronger signals
            params.omega = 2.0e-4    # Higher frequency for faster response to market changes
            
            # Enhanced trading parameters
            params.signal_threshold = 0.4  # Lower threshold for more frequent entries
            params.min_amplitude = 0.015   # Lower amplitude requirement
            
            # Enhanced risk management
            params.stop_loss_atr = 1.0     # Tighter stop loss for aggressive management
            params.take_profit_atr = 2.5   # Much higher take profit target
            params.cooldown_period = 3     # Short cooldown for frequent trading
            
            stats, csv_file = run_quantum_stock_improved_aggressive_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, params
            )
            strategy_results['Quantum Stock - Improved Aggressive'] = stats
        
        # New strategy types from the Quantum Oscillator model
        elif strategy_type == "quantum_oscillator_conservative":
            # Create conservative quantum oscillator parameters
            params = QuantumOscillatorParameters()
            # Conservative settings
            params.lambda_param = 0.15   # Lower energy level: closer to Gaussian
            params.set_spread_from_volatility(0.01)  # Base volatility setting
            params.stop_loss_atr = 2.0   # Wider stop loss
            params.take_profit_atr = 1.0  # Lower take profit
            params.cooldown_period = 15   # Longer cooldown
            
            stats, csv_file = run_quantum_oscillator_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Oscillator - Conservative'] = stats
            
        elif strategy_type == "quantum_oscillator_balanced":
            # Create balanced quantum oscillator parameters
            params = QuantumOscillatorParameters()
            # Default balanced settings
            params.lambda_param = 0.25   # Standard energy level
            params.set_spread_from_volatility(0.01)  # Base volatility setting
            
            stats, csv_file = run_quantum_oscillator_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Oscillator - Balanced'] = stats
            
        elif strategy_type == "quantum_oscillator_aggressive":
            # Create aggressive quantum oscillator parameters
            params = QuantumOscillatorParameters()
            # Aggressive settings - significantly tuned based on Orrell's paper
            params.lambda_param = 0.75    # Much higher energy level: more fat-tailed distributions
            params.hbar = 1.5            # Higher uncertainty
            params.omega = 1/180         # Faster frequency (cycles)
            params.set_spread_from_volatility(0.018)  # Higher volatility sensitivity
            params.stop_loss_atr = 1.0   # Tighter stop loss
            params.take_profit_atr = 2.0  # Higher take profit for larger gains
            params.cooldown_period = 3    # Much shorter cooldown for more frequent trades
            
            stats, csv_file = run_quantum_oscillator_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Oscillator - Aggressive'] = stats
        
        # Original strategy types maintained for backward compatibility
        elif strategy_type == 'ultra_selective':
            # ETH Ultra-Selective parameters - Optimized for high win rate
            strategy_params = {
                'rsi_period': 14,
                'rsi_high': 78,           # Higher threshold for stronger signals
                'rsi_low': 22,            # Lower threshold for stronger signals
                'v0': 3.0,                # Higher barrier for more selective tunneling
                'a': 1.0,                 # Default value for simple backtest
                'mass': 2.0,              # More mass = higher quality tunneling
                'hbar': 0.7,              # Moderate hbar for quality
                'stop_loss_atr': 2.0,     # Wider stop loss to avoid whipsaws
                'take_profit_atr': 1.0,   # Tighter take profit for faster wins
                'cooldown_period': 24,    # Longer cooldown to avoid overtrading
                'max_consecutive_losses': 2,  # Become more selective after 2 losses
                'debug_mode': True
            }
            
            stats, csv_file = run_simple_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, strategy_params
            )
            strategy_results['Ultra Selective'] = stats
        
        elif strategy_type == 'optimal':
            # ETH Optimal parameters - Balanced for high win rate
            strategy_params = {
                'rsi_period': 14,
                'rsi_high': 72,           # Higher threshold for quality signals
                'rsi_low': 28,            # Lower threshold for quality signals
                'v0': 2.0,                # Moderate barrier
                'a': 1.0,                 # Default value for simple backtest
                'mass': 1.5,              # Moderate mass
                'hbar': 1.0,              # Moderate hbar
                'stop_loss_atr': 1.8,     # Wider stop to avoid whipsaws
                'take_profit_atr': 1.2,   # Closer take profit for quicker wins
                'cooldown_period': 12,    # Moderate cooldown
                'max_consecutive_losses': 3,  # Adapt after 3 losses
                'debug_mode': True        # Enable debug for transparency
            }
            
            stats, csv_file = run_simple_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, strategy_params
            )
            strategy_results['Optimal'] = stats
            
        elif strategy_type == 'high_precision':
            # ETH High Precision parameters - Optimized for highest win rate
            strategy_params = {
                'rsi_period': 21,         # Longer period for smoother RSI
                'rsi_high': 80,           # Very high threshold for strong overbought
                'rsi_low': 20,            # Very low threshold for strong oversold
                'v0': 3.5,                # High barrier requiring strong signals
                'a': 1.0,                 # Default value for simple backtest
                'mass': 2.5,              # Higher mass for quality tunneling
                'hbar': 0.6,              # Lower hbar for more selectivity 
                'stop_loss_atr': 2.5,     # Very wide stop loss to avoid noise
                'take_profit_atr': 1.0,   # Target-focused approach
                'cooldown_period': 36,    # Long cooldown between trades
                'max_consecutive_losses': 2,  # Very selective after losses
                'debug_mode': True
            }
            
            stats, csv_file = run_simple_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, strategy_params
            )
            strategy_results['High Precision'] = stats
            
        elif strategy_type == 'pattern_recognition':
            # ETH Pattern Recognition parameters - Focus on quality setups
            strategy_params = {
                'rsi_period': 14,
                'rsi_high': 75,           # Higher threshold
                'rsi_low': 25,            # Lower threshold
                'v0': 2.5,                # Moderate barrier
                'a': 1.0,                 # Default value for simple backtest
                'mass': 1.8,              # Moderate-high mass
                'hbar': 0.7,              # Moderate-low hbar
                'stop_loss_atr': 2.0,     # Wide stop loss for pattern completion
                'take_profit_atr': 1.2,   # Moderate take profit
                'cooldown_period': 24,    # Moderate cooldown
                'max_consecutive_losses': 3,  # Adapt after losses
                'debug_mode': True
            }
            
            stats, csv_file = run_simple_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, strategy_params
            )
            strategy_results['Pattern Recognition'] = stats
            
        elif strategy_type == 'quantum_anharmonic_aggressive':
            # Create aggressive quantum anharmonic oscillator parameters
            params = QuantumAnharmonicParameters()
            # Aggressive settings
            params.beta = 0.15      # Higher cubic term: more skew
            params.gamma = 0.08     # Higher quartic term: more kurtosis
            params.lambda_param = 0.4    # Higher energy level
            params.set_spread_from_volatility(0.012)  # Slightly higher volatility
            params.stop_loss_atr = 1.2   # Tighter stop loss
            params.take_profit_atr = 1.5  # Higher take profit
            params.cooldown_period = 5    # Shorter cooldown
            
            stats, csv_file = run_quantum_anharmonic_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Anharmonic - Aggressive'] = stats
            
        elif strategy_type == 'quantum_anharmonic_conservative':
            # Create conservative quantum anharmonic oscillator parameters
            params = QuantumAnharmonicParameters()
            # Modified conservative settings that will generate trades
            params.beta = 0.08      # Slightly higher cubic term for more skew
            params.gamma = 0.05     # Slightly higher quartic term for more kurtosis
            params.lambda_param = 0.3    # Higher energy level to generate more signals
            params.hbar = 1.1       # Slightly higher uncertainty
            params.set_spread_from_volatility(0.012)  # Slightly higher volatility setting
            params.stop_loss_atr = 1.8   # Still wider stop loss
            params.take_profit_atr = 1.2  # Slightly higher take profit
            params.cooldown_period = 10   # Reduced cooldown period
            params.signal_threshold = 0.5 # Lower threshold to generate trades
            
            stats, csv_file = run_quantum_anharmonic_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Anharmonic - Conservative'] = stats
            
        elif strategy_type == 'quantum_anharmonic_balanced':
            # Create balanced quantum anharmonic oscillator parameters
            params = QuantumAnharmonicParameters()
            # Default balanced settings - no need to change default params
            
            stats, csv_file = run_quantum_anharmonic_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Anharmonic - Balanced'] = stats
            
        elif strategy_type == 'quantum_tunneling_conservative':
            # Create quantum tunneling parameters - much more aggressive to generate trades
            params = QuantumTunnelingParameters()
            # Modified settings to actually generate trades
            params.volatility = 0.3         # Higher volatility for calculation
            params.lambda_param = 0.8       # Much lower lambda to make tunneling more likely
            params.vol_drop_threshold = 0.15  # Accept smaller volatility drops
            params.range_threshold = 0.10   # Accept wider ranges
            params.min_range_days = 5       # Require fewer days in range
            params.tunneling_threshold = 0.05 # Much lower threshold for entry
            params.stop_loss_atr = 1.5      # Moderate stop loss
            params.take_profit_atr = 1.8    # Higher take profit
            params.lookback_period = 15     # Shorter lookback
            params.cooldown_period = 5      # Shorter cooldown
            
            stats, csv_file = run_quantum_tunneling_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Tunneling - Conservative'] = stats
        
        elif strategy_type == 'quantum_tunneling_balanced':
            # Create balanced quantum tunneling parameters
            params = QuantumTunnelingParameters()
            # Default balanced settings
            params.risk_free_rate = 0.03  # Standard risk-free rate
            params.volatility = 0.3       # Standard volatility
            params.range_threshold = 0.05 # Standard range
            params.vol_drop_threshold = 0.3 # Standard volatility drop
            params.tunneling_threshold = 0.2 # Standard tunneling probability
            
            stats, csv_file = run_quantum_tunneling_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Tunneling - Balanced'] = stats
            
        elif strategy_type == 'quantum_tunneling_aggressive':
            # Create even more aggressive tunneling parameters
            params = QuantumTunnelingParameters()
            # Ultra aggressive settings
            params.volatility = 0.35        # Higher volatility
            params.lambda_param = 0.5       # Extremely low lambda for maximum tunneling probability
            params.vol_drop_threshold = 0.1  # Very small volatility drop requirement
            params.range_threshold = 0.15   # Accept much wider ranges
            params.min_range_days = 3       # Minimal range requirement
            params.tunneling_threshold = 0.03 # Extremely low threshold for entry
            params.stop_loss_atr = 1.2      # Tighter stop loss
            params.take_profit_atr = 2.0    # Higher take profit
            params.lookback_period = 10     # Very short lookback
            params.cooldown_period = 3      # Minimal cooldown
            
            stats, csv_file = run_quantum_tunneling_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Tunneling - Aggressive'] = stats
        
        elif strategy_type == 'compare_all':
            # Run all strategies and compare
            
            # Conservative
            conservative_params = QuantumRSIParameters()
            conservative_params.rsi_high = 72  # Lower from 78
            conservative_params.rsi_low = 28   # Higher from 22
            conservative_params.volume_threshold = 1.5  # Lower from 2.0
            conservative_params.take_profit_atr = 1.2  # Higher from 1.0
            conservative_params.stop_loss_atr = 1.6  # Lower from 1.8
            conservative_params.cooldown_period = 6  # Lower from default
            conservative_params.maxwell_demon_filter = True
            conservative_params.barrier_height = 1.3  # Lower barrier for easier crossover
            
            conservative_stats, conservative_csv = run_optimized_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, conservative_params
            )
            strategy_results['Conservative'] = conservative_stats
            
            # Balanced (Optimized)
            balanced_params = QuantumRSIParameters()
            # Modify default settings to generate more trades
            balanced_params.rsi_high = 68  # Lower threshold
            balanced_params.rsi_low = 32   # Higher threshold
            balanced_params.volume_threshold = 1.2
            balanced_params.cooldown_period = 4  # Reduced cooldown
            balanced_params.maxwell_demon_filter = True
            balanced_params.barrier_height = 1.0  # Lower barrier
            
            balanced_stats, balanced_csv = run_optimized_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, balanced_params
            )
            strategy_results['Balanced'] = balanced_stats
            
            # Optimal - tuned for more trades
            optimal_params = {
                'rsi_period': 14,
                'rsi_high': 65,           # Lower from 72
                'rsi_low': 35,            # Higher from 28
                'v0': 1.5,                # Lower from 2.0
                'a': 1.0,                 # Default value
                'mass': 1.2,              # Lower from 1.5
                'hbar': 1.2,              # Higher from 1.0
                'stop_loss_atr': 1.5,     # Tighter from 1.8
                'take_profit_atr': 1.4,   # Higher from 1.2
                'cooldown_period': 6,     # Shorter from 12
                'max_consecutive_losses': 3,  # Keep same
                'debug_mode': True
            }
            
            optimal_stats, optimal_csv = run_simple_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, optimal_params
            )
            strategy_results['Optimal'] = optimal_stats
            
            # High Precision - tuned for more trades but still quality signals
            hp_params = {
                'rsi_period': 14,         # Shorter from 21
                'rsi_high': 75,           # Lower from 80
                'rsi_low': 25,            # Higher from 20
                'v0': 2.8,                # Lower from 3.5
                'a': 1.0,                 # Default value
                'mass': 1.8,              # Lower from 2.5
                'hbar': 0.8,              # Higher from 0.6
                'stop_loss_atr': 2.0,     # Tighter from 2.5
                'take_profit_atr': 1.2,   # Higher from 1.0
                'cooldown_period': 18,    # Shorter from 36
                'max_consecutive_losses': 3,  # More lenient from 2
                'debug_mode': True
            }
            
            hp_stats, hp_csv = run_simple_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, hp_params
            )
            strategy_results['High Precision'] = hp_stats
            
            # Pattern Recognition - tuned for more trades
            pattern_params = {
                'rsi_period': 10,         # Shorter from 14 - more responsive
                'rsi_high': 68,           # Lower from 75
                'rsi_low': 32,            # Higher from 25
                'v0': 1.8,                # Lower from 2.5
                'a': 1.0,                 # Default value
                'mass': 1.4,              # Lower from 1.8
                'hbar': 0.9,              # Higher from 0.7 
                'stop_loss_atr': 1.7,     # Tighter from 2.0
                'take_profit_atr': 1.4,   # Higher from 1.2
                'cooldown_period': 12,    # Halved from 24
                'max_consecutive_losses': 4,  # More forgiving from 3
                'debug_mode': True
            }
            
            pattern_stats, pattern_csv = run_simple_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, pattern_params
            )
            strategy_results['Pattern Recognition'] = pattern_stats
            
            # Add Quantum Probability model strategies
            # Conservative
            qp_conservative = QuantumProbabilityParameters()
            qp_conservative.alpha = 2.5
            qp_conservative.delta = 0.4
            qp_conservative.h = 0.8
            qp_conservative.omega_level = 1
            qp_conservative.stop_loss_atr = 1.8
            qp_conservative.take_profit_atr = 1.2
            qp_conservative.cooldown_period = 12
            
            qp_cons_stats, qp_cons_csv = run_quantum_probability_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, qp_conservative
            )
            strategy_results['Quantum Prob - Conservative'] = qp_cons_stats
            
            # Balanced
            qp_balanced = QuantumProbabilityParameters()
            qp_balanced.alpha = 2.0
            qp_balanced.delta = 0.5
            qp_balanced.h = 1.0
            qp_balanced.omega_level = 3
            qp_balanced.stop_loss_atr = 1.5
            qp_balanced.take_profit_atr = 1.3
            qp_balanced.cooldown_period = 8
            
            qp_bal_stats, qp_bal_csv = run_quantum_probability_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, qp_balanced
            )
            strategy_results['Quantum Prob - Balanced'] = qp_bal_stats
            
            # Aggressive
            qp_aggressive = QuantumProbabilityParameters()
            qp_aggressive.alpha = 1.5
            qp_aggressive.delta = 0.8
            qp_aggressive.h = 1.5
            qp_aggressive.omega_level = 5
            qp_aggressive.stop_loss_atr = 1.2
            qp_aggressive.take_profit_atr = 1.5
            qp_aggressive.cooldown_period = 5
            
            qp_agg_stats, qp_agg_csv = run_quantum_probability_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, qp_aggressive
            )
            strategy_results['Quantum Prob - Aggressive'] = qp_agg_stats
            
            # Add the Quantum Stock Model strategies
            # Conservative
            qs_conservative = QuantumStockParameters()
            qs_conservative.m = 2e-30
            qs_conservative.hbar = 0.8
            qs_conservative.F = 0.8e-19
            qs_conservative.omega = 0.8e-4
            qs_conservative.stop_loss_atr = 1.8
            qs_conservative.take_profit_atr = 1.2
            qs_conservative.cooldown_period = 12
            qs_conservative.signal_threshold = 0.7
            
            qs_cons_stats, qs_cons_csv = run_quantum_stock_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, qs_conservative
            )
            strategy_results['Quantum Stock - Conservative'] = qs_cons_stats
            
            # Balanced
            qs_balanced = QuantumStockParameters()
            # Default settings
            
            qs_bal_stats, qs_bal_csv = run_quantum_stock_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, qs_balanced
            )
            strategy_results['Quantum Stock - Balanced'] = qs_bal_stats
            
            # Aggressive
            qs_aggressive = QuantumStockParameters()
            qs_aggressive.m = 0.5e-30
            qs_aggressive.hbar = 1.2
            qs_aggressive.F = 1.2e-19
            qs_aggressive.omega = 1.2e-4
            qs_aggressive.stop_loss_atr = 1.2
            qs_aggressive.take_profit_atr = 1.5
            qs_aggressive.cooldown_period = 5
            qs_aggressive.signal_threshold = 0.5
            
            qs_agg_stats, qs_agg_csv = run_quantum_stock_strategy_backtest(
                crypto, timeframe, initial_capital, start_date, end_date, qs_aggressive
            )
            strategy_results['Quantum Stock - Aggressive'] = qs_agg_stats
            
            # Quantum Anharmonic Oscillator Models
            # Conservative
            params = QuantumAnharmonicParameters()
            params.beta = 0.08      # Slightly higher cubic term for more skew
            params.gamma = 0.05     # Slightly higher quartic term for more kurtosis
            params.lambda_param = 0.3    # Higher energy level to generate more signals
            params.hbar = 1.1       # Slightly higher uncertainty
            params.set_spread_from_volatility(0.012)  # Slightly higher volatility setting
            params.stop_loss_atr = 1.8   # Still wider stop loss
            params.take_profit_atr = 1.2  # Slightly higher take profit
            params.cooldown_period = 10   # Reduced cooldown period
            params.signal_threshold = 0.5 # Lower threshold to generate trades
            qa_cons_stats, qa_cons_csv = run_quantum_anharmonic_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Anharmonic - Conservative'] = qa_cons_stats
            
            # Balanced
            params = QuantumAnharmonicParameters()
            # Default settings
            qa_bal_stats, qa_bal_csv = run_quantum_anharmonic_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Anharmonic - Balanced'] = qa_bal_stats
            
            # Aggressive
            params = QuantumAnharmonicParameters()
            params.beta = 0.15      # Higher cubic term: more skew
            params.gamma = 0.08     # Higher quartic term: more kurtosis
            params.lambda_param = 0.4    # Higher energy level
            params.set_spread_from_volatility(0.012)
            params.stop_loss_atr = 1.2
            params.take_profit_atr = 1.5
            params.cooldown_period = 5
            qa_agg_stats, qa_agg_csv = run_quantum_anharmonic_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Anharmonic - Aggressive'] = qa_agg_stats

            # Quantum Tunneling Models
            # Conservative
            params = QuantumTunnelingParameters()
            params.volatility = 0.3         # Higher volatility for calculation
            params.lambda_param = 0.8       # Much lower lambda to make tunneling more likely
            params.vol_drop_threshold = 0.15  # Accept smaller volatility drops
            params.range_threshold = 0.10   # Accept wider ranges
            params.min_range_days = 5       # Require fewer days in range
            params.tunneling_threshold = 0.05 # Much lower threshold for entry
            params.stop_loss_atr = 1.5      # Moderate stop loss
            params.take_profit_atr = 1.8    # Higher take profit
            params.lookback_period = 15     # Shorter lookback
            params.cooldown_period = 5      # Shorter cooldown
            qt_cons_stats, qt_cons_csv = run_quantum_tunneling_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Tunneling - Conservative'] = qt_cons_stats
            
            # Aggressive
            params = QuantumTunnelingParameters()
            params.volatility = 0.35        # Higher volatility
            params.lambda_param = 0.5       # Extremely low lambda for maximum tunneling probability
            params.vol_drop_threshold = 0.1  # Very small volatility drop requirement
            params.range_threshold = 0.15   # Accept much wider ranges
            params.min_range_days = 3       # Minimal range requirement
            params.tunneling_threshold = 0.03 # Extremely low threshold for entry
            params.stop_loss_atr = 1.2      # Tighter stop loss
            params.take_profit_atr = 2.0    # Higher take profit
            params.lookback_period = 10     # Very short lookback
            params.cooldown_period = 3      # Minimal cooldown
            qt_agg_stats, qt_agg_csv = run_quantum_tunneling_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Tunneling - Aggressive'] = qt_agg_stats

            # Add Quantum Oscillator models
            # Conservative
            params = QuantumOscillatorParameters()
            params.lambda_param = 0.15
            params.set_spread_from_volatility(0.01)
            params.stop_loss_atr = 2.0
            params.take_profit_atr = 1.0
            params.cooldown_period = 15
            qo_cons_stats, qo_cons_csv = run_quantum_oscillator_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Oscillator - Conservative'] = qo_cons_stats
            
            # Balanced
            params = QuantumOscillatorParameters()
            params.lambda_param = 0.25
            params.set_spread_from_volatility(0.01)
            qo_bal_stats, qo_bal_csv = run_quantum_oscillator_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Oscillator - Balanced'] = qo_bal_stats
            
            # Aggressive
            params = QuantumOscillatorParameters()
            params.lambda_param = 0.75    # Much higher energy level: more fat-tailed distributions
            params.hbar = 1.5            # Higher uncertainty
            params.omega = 1/180         # Faster frequency (cycles)
            params.set_spread_from_volatility(0.018)  # Higher volatility sensitivity
            params.stop_loss_atr = 1.0   # Tighter stop loss
            params.take_profit_atr = 2.0  # Higher take profit for larger gains
            params.cooldown_period = 3    # Much shorter cooldown for more frequent trades
            
            qo_agg_stats, qo_agg_csv = run_quantum_oscillator_strategy_backtest(
                crypto, timeframe, start_date, end_date, params
            )
            strategy_results['Quantum Oscillator - Aggressive'] = qo_agg_stats
        
        # Save results for access in other routes
        backtest_results[crypto] = strategy_results
        
        # Generate a plot of the results for the UI
        try:
            plt.figure(figsize=(14, 9))  # Larger figure for better visibility
            colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'teal', 'magenta', 'cyan', 'lime']
            i = 0
            
            for strategy, stats in strategy_results.items():
                if 'Equity Curve' in stats and stats['Equity Curve']:
                    equity_curve = stats['Equity Curve']
                    # For optimized strategies, equity curve is already calculated
                    # Normalize to percentage return for comparison
                    equity_pct = [(e - equity_curve[0]) / equity_curve[0] * 100 for e in equity_curve]
                    plt.plot(equity_pct, label=f"{strategy} ({stats['Win Rate [%]']:.1f}% win)", color=colors[i % len(colors)])
                    i += 1
            
            plt.title(f'Strategy Comparison: {crypto} {timeframe}', fontsize=14)
            plt.xlabel('Bars', fontsize=12)
            plt.ylabel('Return (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Place legend outside the plot for better visibility
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
            plt.tight_layout()  # Adjust layout to make room for the legend
            
            # Save the plot to a buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)  # Higher DPI for better quality
            buffer.seek(0)
            plt.close()
            
            # Convert to base64 for embedding in HTML
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Generate a second version with the legend embedded within the plot (for mobile viewing)
            plt.figure(figsize=(10, 6))
            i = 0
            
            for strategy, stats in strategy_results.items():
                if 'Equity Curve' in stats and stats['Equity Curve']:
                    equity_curve = stats['Equity Curve']
                    equity_pct = [(e - equity_curve[0]) / equity_curve[0] * 100 for e in equity_curve]
                    plt.plot(equity_pct, label=f"{strategy} ({stats['Win Rate [%]']:.1f}% win)", color=colors[i % len(colors)])
                    i += 1
            
            plt.title(f'Strategy Comparison: {crypto} {timeframe}')
            plt.xlabel('Bars')
            plt.ylabel('Return (%)')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=8)
            plt.tight_layout()
            
            # Save the mobile-friendly plot
            mobile_buffer = io.BytesIO()
            plt.savefig(mobile_buffer, format='png', dpi=100)
            mobile_buffer.seek(0)
            plt.close()
            
            # Convert to base64
            mobile_plot_data = base64.b64encode(mobile_buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error generating plot: {str(e)}")
            plot_data = ""
            mobile_plot_data = ""
        
        # Prepare results for the UI
        for strategy, stats in strategy_results.items():
            # Format numbers for display
            if 'Win Rate [%]' in stats:
                if isinstance(stats['Win Rate [%]'], (int, float)):
                    stats['Win Rate [%]'] = f"{stats['Win Rate [%]']:.2f}%"
            if 'Return [%]' in stats:
                if isinstance(stats['Return [%]'], (int, float)):
                    stats['Return [%]'] = f"{stats['Return [%]']:.2f}%"
            if 'Net Profit' in stats:
                if isinstance(stats['Net Profit'], (int, float)):
                    stats['Net Profit'] = f"${stats['Net Profit']:.2f}"
            if 'Max Drawdown [%]' in stats:
                if isinstance(stats['Max Drawdown [%]'], (int, float)):
                    stats['Max Drawdown [%]'] = f"{stats['Max Drawdown [%]']:.2f}%"
            if 'Profit Factor' in stats:
                if isinstance(stats['Profit Factor'], (int, float)):
                    stats['Profit Factor'] = f"{stats['Profit Factor']:.2f}"
        
        return jsonify({
            'success': True,
            'results': strategy_results,
            'plot': plot_data,
            'mobile_plot': mobile_plot_data,
            'message': f'Backtest completed for {crypto} on {timeframe} timeframe'
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Serve the trade log CSV files
@app.route('/download_trades/<path:filename>')
def download_trades(filename):
    """Serve the trade log CSV file for download"""
    return send_file(f'static/results/{filename}')

def run_quantum_oscillator_classic_strategy(df, token="Unknown"):
    """
    Run the classic Quantum Oscillator strategy on the provided DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        token (str): Token symbol for reporting
        
    Returns:
        dict: Performance statistics
    """
    print(f"Starting Quantum Oscillator backtest with {len(df)} data points...")
    
    # Create parameters for classic settings
    from quantum_oscillator_model import QuantumOscillatorParameters
    params = QuantumOscillatorParameters()
    
    # Use the standard parameters from the original model
    params.lambda_param = 0.5    # Energy level parameter
    params.hbar = 1.0            # Standard uncertainty
    params.omega = 1/100         # Standard oscillation (approximately 100 days)
    
    # Standard volatility estimate
    params.set_spread_from_volatility(0.015)  # Using 1.5% daily vol estimate
    
    # Standard trade entry settings
    params.signal_threshold = 0.55   # Original threshold
    params.min_amplitude = 0.02      # Standard meaningful move 
    
    # Standard risk management
    params.stop_loss_atr = 1.5       # Standard stop loss
    params.take_profit_atr = 1.5     # Standard profit target
    params.cooldown_period = 3       # Standard cooldown
    
    # Run the backtest
    from quantum_oscillator_model import run_backtest
    results = run_backtest(df, params)
    
    # Print results summary
    print(f"=== Quantum Oscillator Backtest Results ===")
    print(f"Total Return: {results.get('total_return', 0):.2f}%")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(f"Win Rate: {results.get('win_rate', 0):.2f}%")
    print(f"Average Win: {results.get('avg_win', 0):.2f}%")
    print(f"Average Loss: {results.get('avg_loss', 0):.2f}%")
    print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
    print(f"Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%")
    
    # Format the results in the expected format
    formatted_stats = {
        'Total Trades': results.get('total_trades', 0),
        'Win Rate [%]': results.get('win_rate', 0),
        'Return [%]': results.get('total_return', 0),
        'Net Profit': results.get('total_return', 0) * df.iloc[0]['close'] / 100 if len(df) > 0 else 0,  # Estimate based on initial price
        'Max Drawdown [%]': results.get('max_drawdown', 0) * 100,
        'Profit Factor': results.get('profit_factor', 0),
        'Trades': results.get('trade_log', []),
        'Equity Curve': results.get('equity', [])
    }
    
    return formatted_stats

if __name__ == '__main__':
    # Create results directory
    os.makedirs('static/results', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000) 