#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantum Harmonic-Wave Hybrid Model

This model combines the strengths of two quantum-inspired approaches:
1. Quantum Oscillator Model - Known for high win rates and mean reversion
2. Quantum Stock Model - Known for capturing larger returns in trending markets

The hybrid approach uses a dual-state quantum framework:
- A harmonic oscillator potential term for mean reversion (equilibrium)
- An external field (bias) term for directional market pressure (trend)

The combined model dynamically operates in either mean-reversion or trend-following 
mode based on detected market conditions across multiple timeframes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Union


class QuantumHybridParameters:
    """
    Parameters for the Quantum Harmonic-Wave Hybrid model.
    
    Combines parameters from both the Quantum Oscillator and Quantum Stock models
    with additional parameters to control the balance between them.
    """
    def __init__(self):
        # Weights for conservative vs aggressive signals
        self.conservative_weight = 0.6    # Weight for conservative signals (higher = more conservative)
        self.aggressive_weight = 0.4      # Weight for aggressive signals (higher = more aggressive)
        
        # Hybrid model specific parameters
        self.trend_threshold = 0.15       # Threshold for trend detection (lowered from 0.25)
        self.oscillator_strength = 0.8    # Strength of oscillator component (mean reversion)
        self.trend_strength = 0.6         # Strength of trend-following component
        
        # Market regime detection parameters
        self.volatility_lookback = 20     # Lookback period for volatility calculation
        self.regime_threshold = 1.2       # Threshold for regime change detection (lowered from 1.5)
        
        # Quantum oscillator parameters
        self.lambda_param = 0.3           # Controls oscillator wavelength
        self.omega = 1/180                # Oscillator frequency (typical value for 4h timeframe)
        self.hbar = 1.0                   # Planck constant analog
        
        # Quantum stock parameters
        self.m = 1.0e-30                  # Mass parameter for quantum stock model
        self.mass = 1.0e-30               # Alternative mass parameter (kept for compatibility)
        self.F = 1.0e-19                  # Field amplitude for potential
        
        # Entry and exit thresholds
        self.entry_threshold = 0.6        # Signal strength required for entry (lowered from 1.2)
        self.exit_threshold = 0.4         # Signal strength threshold for exit (lowered from 0.8)
        
        # Volatility adaptation
        self.volatility_factor = 1.0      # Factor to scale thresholds based on volatility
        self.adapt_to_volatility = True   # Whether to adjust thresholds based on volatility
        
        # Risk management
        self.stop_loss_atr = 1.5          # Stop loss in ATR units
        self.take_profit_atr = 2.0        # Take profit in ATR units
        self.cooldown_period = 3          # Bars to wait after a trade (reduced from 5)
        self.max_trades_per_day = 8       # Maximum trades per day (increased from 5)
        self.max_risk_per_trade = 0.02    # Maximum risk per trade (fraction of capital)


def calculate_rsi(prices, period=14):
    """Calculate RSI without talib"""
    # Convert to pandas Series if numpy array
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # Calculate returns and separate gains and losses
    delta = prices.diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = -loss  # Convert to positive values
    
    # First average gain and loss
    avg_gain = gain.rolling(window=period).mean().fillna(0)
    avg_loss = loss.rolling(window=period).mean().fillna(0)
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values  # Return as numpy array for compatibility

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands without talib"""
    # Convert to pandas Series if numpy array
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
        
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower

def calculate_atr(high, low, close, period=14):
    """Calculate ATR without talib"""
    # Convert to pandas Series if numpy arrays
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
        
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_market_imbalance(df: pd.DataFrame, params: QuantumHybridParameters) -> pd.DataFrame:
    """
    Calculate the market imbalance based on the hybrid quantum approach.
    
    Combines both oscillator-based mean reversion imbalance with stock model
    external field-driven imbalance to create a composite signal.
    
    Args:
        df: DataFrame with price data
        params: Model parameters
        
    Returns:
        DataFrame with market imbalance indicators
    """
    # Copy to avoid modifying the original
    result = df.copy()
    
    # --- Calculate oscillator-based market imbalance ---
    # RSI as a measure of overbought/oversold
    result['rsi'] = calculate_rsi(result['close'])
    
    # Calculate normalized RSI-based imbalance (-1 to 1 scale)
    result['rsi_imbalance'] = 2 * (result['rsi'] - 50) / 100
    
    # Bollinger Bands for mean reversion
    upper, middle, lower = calculate_bollinger_bands(result['close'])
    result['bb_upper'] = upper
    result['bb_middle'] = middle
    result['bb_lower'] = lower
    
    # BB-based imbalance (-1 to 1 scale, positive when price is high)
    # Safely handle division by zero by using a small epsilon
    epsilon = 1e-6
    bb_width = result['bb_upper'] - result['bb_middle']
    bb_width = bb_width.replace(0, epsilon)  # Replace zeros with small value
    result['bb_imbalance'] = (result['close'] - result['bb_middle']) / bb_width
    
    # Clip to range [-1, 1]
    result['bb_imbalance'] = np.clip(result['bb_imbalance'], -1, 1)
    
    # --- Calculate stock model-based market imbalance ---
    # Momentum indicators
    result['momentum'] = result['close'].pct_change(10).fillna(0)  # Changed to pct_change with fillna
    
    # Handle potential NaN values in momentum calculation
    momentum_mean = result['momentum'].rolling(30).mean().fillna(0)
    momentum_std = result['momentum'].rolling(30).std().fillna(0.001)  # Avoid division by zero
    result['norm_momentum'] = (result['momentum'] - momentum_mean) / momentum_std
    result['norm_momentum'] = result['norm_momentum'].fillna(0)
    result['norm_momentum'] = np.clip(result['norm_momentum'], -3, 3) / 3  # Normalize to [-1, 1]
    
    # Moving average relationship
    result['ma_fast'] = result['close'].rolling(10).mean().fillna(method='bfill')  # Fill NaN values
    result['ma_slow'] = result['close'].rolling(30).mean().fillna(method='bfill')  # Fill NaN values
    
    # Avoid division by zero in ma_imbalance calculation
    ma_denom = result['ma_slow'].replace(0, epsilon)
    result['ma_imbalance'] = (result['ma_fast'] - result['ma_slow']) / ma_denom
    result['ma_imbalance'] = np.clip(result['ma_imbalance'] * 20, -1, 1)  # Scale and clip to [-1, 1]
    
    # --- Combine signals into a hybrid market imbalance ---
    # Oscillator component (mean reversion)
    oscillator_imbalance = (result['rsi_imbalance'] + result['bb_imbalance']) / 2
    
    # Stock model component (trend following)
    stock_imbalance = (result['norm_momentum'] + result['ma_imbalance']) / 2
    
    # Calculate market volatility
    result['volatility'] = result['close'].pct_change().rolling(params.volatility_lookback).std().fillna(0)
    
    # Determine regime (higher volatility = more weight to oscillator for safety)
    vol_rolling_mean = result['volatility'].rolling(100).mean().fillna(result['volatility'])
    result['volatility_regime'] = result['volatility'] / vol_rolling_mean
    result['volatility_regime'] = result['volatility_regime'].fillna(1.0)
    
    # Dynamic weighting based on volatility and trend strength
    trend_weight = np.minimum(1.0, np.abs(stock_imbalance) * params.trend_strength)
    oscillator_weight = params.oscillator_strength * (2.0 - trend_weight)
    
    # Combine signals with dynamic weights
    result['hybrid_imbalance'] = (
        oscillator_weight * oscillator_imbalance +
        trend_weight * stock_imbalance
    ) / (oscillator_weight + trend_weight)
    
    # Signal strength - higher when both oscillator and trend agree
    agreement_factor = np.abs(np.sign(oscillator_imbalance) * np.sign(stock_imbalance))
    result['signal_strength'] = np.abs(result['hybrid_imbalance']) * (1 + agreement_factor * 0.5)
    
    return result


def calculate_energy_levels(df: pd.DataFrame, params: QuantumHybridParameters) -> pd.DataFrame:
    """
    Calculate quantum energy levels for price movement.
    
    Uses concepts from quantum mechanics to identify potential breakout levels
    and areas where mean reversion is likely.
    
    Args:
        df: DataFrame with market imbalance data
        params: Model parameters
        
    Returns:
        DataFrame with energy levels and transition probabilities
    """
    result = df.copy()
    
    # --- Calculate price volatility (proportional to energy) ---
    result['atr'] = calculate_atr(result['high'], result['low'], result['close'])
    
    # Normalized ATR (relative to recent levels)
    result['norm_atr'] = result['atr'] / result['atr'].rolling(50).mean()
    result['norm_atr'] = result['norm_atr'].fillna(1.0)
    
    # --- Calculate quantum energy levels ---
    # Base energy level (proportional to volatility)
    result['base_energy'] = result['norm_atr'] * params.hbar * params.omega
    
    # Energy levels - discrete steps of volatility
    energy_step = params.hbar * params.omega
    
    # Ground state energy
    result['ground_energy'] = 0.5 * energy_step
    
    # First excited state energy threshold
    result['excited_energy'] = 1.5 * energy_step
    
    # Current energy estimate based on recent price action
    result['current_energy'] = result['base_energy'] * (1 + np.abs(result['hybrid_imbalance']))
    
    # Probability of staying in the ground state (mean reversion zone)
    result['ground_state_prob'] = np.exp(-np.abs(result['hybrid_imbalance']) * 2)
    
    # Probability of transition to excited state (breakout potential)
    result['excitation_prob'] = 1 - result['ground_state_prob']
    
    # External field influence (shifts the energy levels)
    result['external_field'] = params.F * result['ma_imbalance']
    
    # Adjusted energy levels based on external field
    field_effect = result['external_field'] ** 2 / (2 * params.m * params.omega ** 2)
    result['adjusted_ground'] = result['ground_energy'] - field_effect
    result['adjusted_excited'] = result['excited_energy'] - field_effect
    
    # Expected equilibrium shift due to external field
    result['equilibrium_shift'] = result['external_field'] / (params.m * params.omega ** 2)
    
    return result


def generate_hybrid_signals(df: pd.DataFrame, params: QuantumHybridParameters) -> pd.DataFrame:
    """
    Generate trading signals based on the hybrid quantum model.
    
    Combines oscillator and stock model signals with adaptive weighting
    based on the detected market regime.
    
    Args:
        df: DataFrame with energy levels and market imbalance
        params: Model parameters
        
    Returns:
        DataFrame with trading signals
    """
    result = df.copy()
    
    # --- Generate basic signal components ---
    # Conservative signal (higher win rate, based on oscillator model)
    conservative_signal = -result['bb_imbalance']  # Negative because we want to buy low, sell high
    
    # Aggressive signal (higher return potential, based on stock model)
    aggressive_signal = result['ma_imbalance'] + 0.5 * result['norm_momentum']
    
    # --- Market regime detection ---
    # Trending vs ranging market detection
    result['is_trending'] = np.abs(result['ma_imbalance']) > params.trend_threshold
    
    # Adapt weights based on the market regime
    conservative_weight = np.where(
        result['is_trending'],
        params.conservative_weight * 0.7,  # Less weight to conservative in trends
        params.conservative_weight * 1.3   # More weight to conservative in ranges
    )
    
    aggressive_weight = np.where(
        result['is_trending'],
        params.aggressive_weight * 1.3,    # More weight to aggressive in trends
        params.aggressive_weight * 0.7     # Less weight to aggressive in ranges
    )
    
    # Normalize weights
    total_weight = conservative_weight + aggressive_weight
    conservative_weight = conservative_weight / total_weight
    aggressive_weight = aggressive_weight / total_weight
    
    # --- Combine signals with adaptive weighting ---
    result['raw_signal'] = (
        conservative_weight * conservative_signal +
        aggressive_weight * aggressive_signal
    )
    
    # Apply quantum regime considerations
    # In high excitation probability, amplify signal in direction of external field
    excitation_effect = result['excitation_prob'] * np.sign(result['equilibrium_shift']) * 0.5
    result['quantum_signal'] = result['raw_signal'] + excitation_effect
    
    # Signal smoothing
    result['signal'] = result['quantum_signal'].rolling(3).mean()
    result['signal'] = result['signal'].fillna(0)
    
    # --- Generate entry/exit signals ---
    # Long entry: positive signal and high enough strength - LOWERED THRESHOLD
    result['long_entry'] = (
        (result['signal'] > params.entry_threshold * 0.8) & 
        (result['signal_strength'] > params.entry_threshold * 0.8) &
        (~result['is_trending'] | (result['is_trending'] & result['ma_imbalance'] > 0))
    )
    
    # Short entry: negative signal and high enough strength - LOWERED THRESHOLD
    result['short_entry'] = (
        (result['signal'] < -params.entry_threshold * 0.8) & 
        (result['signal_strength'] > params.entry_threshold * 0.8) &
        (~result['is_trending'] | (result['is_trending'] & result['ma_imbalance'] < 0))
    )
    
    # Long exit: signal weakens or turns negative - LOWERED THRESHOLD
    result['long_exit'] = (
        (result['signal'] < params.exit_threshold * 0.8) | 
        (result['signal'] < -0.2)  # Previously 0, now -0.2 to be more forgiving
    )
    
    # Short exit: signal weakens or turns positive - LOWERED THRESHOLD
    result['short_exit'] = (
        (result['signal'] > -params.exit_threshold * 0.8) | 
        (result['signal'] > 0.2)  # Previously 0, now 0.2 to be more forgiving
    )
    
    # Apply cooldown to avoid overtrading
    for i in range(len(result)):
        if i < params.cooldown_period:
            continue
        
        # Check if entry happened in the cooldown period
        recent_entries = (
            result['long_entry'].iloc[i-params.cooldown_period:i].any() or
            result['short_entry'].iloc[i-params.cooldown_period:i].any()
        )
        
        if recent_entries:
            result.loc[result.index[i], 'long_entry'] = False
            result.loc[result.index[i], 'short_entry'] = False
    
    return result


def find_trading_signals(df: pd.DataFrame, params: Optional[QuantumHybridParameters] = None) -> pd.DataFrame:
    """
    Main function to find trading signals using the Quantum Harmonic-Wave Hybrid model.
    
    Args:
        df: DataFrame with OHLCV data
        params: Model parameters (optional, uses defaults if None)
        
    Returns:
        DataFrame with all indicators and signals
    """
    if params is None:
        params = QuantumHybridParameters()
    
    # Calculate market imbalance
    result = calculate_market_imbalance(df, params)
    
    # Calculate energy levels
    result = calculate_energy_levels(result, params)
    
    # Generate trading signals
    result = generate_hybrid_signals(result, params)
    
    # Add stop loss and take profit levels
    result['stop_loss_long'] = result['close'] * (1 - params.stop_loss_atr * result['atr'] / result['close'])
    result['take_profit_long'] = result['close'] * (1 + params.take_profit_atr * result['atr'] / result['close'])
    
    result['stop_loss_short'] = result['close'] * (1 + params.stop_loss_atr * result['atr'] / result['close'])
    result['take_profit_short'] = result['close'] * (1 - params.take_profit_atr * result['atr'] / result['close'])
    
    return result


def plot_signals(df: pd.DataFrame, start_idx: int = 0, end_idx: Optional[int] = None) -> None:
    """
    Plot the price chart with signals for visual analysis.
    
    Args:
        df: DataFrame with signals
        start_idx: Starting index for plotting
        end_idx: Ending index for plotting (optional)
    """
    if end_idx is None:
        end_idx = len(df)
    
    plot_df = df.iloc[start_idx:end_idx].copy()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Price chart with BB bands
    ax1.plot(plot_df.index, plot_df['close'], label='Close Price')
    ax1.plot(plot_df.index, plot_df['bb_upper'], 'r--', alpha=0.3, label='Upper BB')
    ax1.plot(plot_df.index, plot_df['bb_middle'], 'g--', alpha=0.3, label='Middle BB')
    ax1.plot(plot_df.index, plot_df['bb_lower'], 'r--', alpha=0.3, label='Lower BB')
    
    # Plot buy signals
    buy_points = plot_df[plot_df['long_entry']]
    ax1.scatter(buy_points.index, buy_points['close'], marker='^', color='green', s=100, label='Buy Signal')
    
    # Plot sell signals
    sell_points = plot_df[plot_df['short_entry']]
    ax1.scatter(sell_points.index, sell_points['close'], marker='v', color='red', s=100, label='Sell Signal')
    
    # Plot trending periods
    trend_periods = plot_df[plot_df['is_trending']]
    ax1.scatter(trend_periods.index, trend_periods['close'], marker='o', color='orange', s=20, alpha=0.3, label='Trending')
    
    ax1.set_title('Quantum Harmonic-Wave Hybrid Model Signals')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Signal plot
    ax2.plot(plot_df.index, plot_df['signal'], label='Signal')
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.axhline(y=1, color='g', linestyle='--', alpha=0.3)
    ax2.axhline(y=-1, color='r', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Signal Value')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Energy levels and probabilities
    ax3.plot(plot_df.index, plot_df['excitation_prob'], 'orange', label='Excitation Probability')
    ax3.plot(plot_df.index, plot_df['ground_state_prob'], 'blue', label='Ground State Probability')
    ax3.set_ylabel('Probability')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


def backtest_hybrid_strategy(df: pd.DataFrame, params: Optional[QuantumHybridParameters] = None, 
                          initial_capital: float = 10000.0, plot: bool = True) -> Tuple[Dict, pd.DataFrame]:
    """
    Backtest the Quantum Harmonic-Wave Hybrid strategy.
    
    Args:
        df: DataFrame with OHLCV data
        params: Model parameters (optional)
        initial_capital: Initial capital for backtesting
        plot: Whether to plot the backtest results
        
    Returns:
        Tuple of (performance_metrics, trades_df)
    """
    if params is None:
        params = QuantumHybridParameters()
    
    # Find trading signals
    result = find_trading_signals(df, params)
    
    # Initialize tracking variables
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = 0
    trades = []
    current_trade = {}
    
    # Output some debug info about signals
    long_signals = result['long_entry'].sum()
    short_signals = result['short_entry'].sum()
    print(f"Found {long_signals} long entry signals and {short_signals} short entry signals")
    
    # Run backtest
    for i in range(50, len(result)):  # Skip the first 50 rows to allow indicators to stabilize
        date = result.index[i]
        close = result['close'].iloc[i]
        
        # Check for exits if in a position
        if position == 1:  # Long position
            # Check stop loss and take profit
            if (close <= current_trade['stop_loss'] or 
                close >= current_trade['take_profit'] or 
                result['long_exit'].iloc[i]):
                
                # Calculate profit/loss
                profit = (close - entry_price) * current_trade['size']
                capital += profit
                
                # Record trade
                current_trade['exit_date'] = date
                current_trade['exit_price'] = close
                current_trade['profit'] = profit
                current_trade['return'] = profit / (entry_price * current_trade['size']) * 100
                trades.append(current_trade)
                
                # Reset position
                position = 0
                
        elif position == -1:  # Short position
            # Check stop loss and take profit
            if (close >= current_trade['stop_loss'] or 
                close <= current_trade['take_profit'] or 
                result['short_exit'].iloc[i]):
                
                # Calculate profit/loss
                profit = (entry_price - close) * current_trade['size']
                capital += profit
                
                # Record trade
                current_trade['exit_date'] = date
                current_trade['exit_price'] = close
                current_trade['profit'] = profit
                current_trade['return'] = profit / (entry_price * current_trade['size']) * 100
                trades.append(current_trade)
                
                # Reset position
                position = 0
        
        # Check for entries if not in a position
        if position == 0:
            if result['long_entry'].iloc[i]:
                # Enter long position
                entry_price = close
                # Use a smaller risk per trade to allow more trades
                position_size = min(
                    (capital * params.max_risk_per_trade) / (close * 0.02),  # 2% risk per trade
                    capital * 0.25 / close  # Never use more than 25% of capital
                )
                position = 1
                
                # Record trade
                current_trade = {
                    'entry_date': date,
                    'entry_price': entry_price,
                    'type': 'LONG',
                    'size': position_size,
                    'stop_loss': max(result['stop_loss_long'].iloc[i], entry_price * 0.95),  # Max 5% stop
                    'take_profit': min(result['take_profit_long'].iloc[i], entry_price * 1.1)  # Min 10% take profit
                }
                
            elif result['short_entry'].iloc[i]:
                # Enter short position
                entry_price = close
                # Use a smaller risk per trade to allow more trades
                position_size = min(
                    (capital * params.max_risk_per_trade) / (close * 0.02),  # 2% risk per trade
                    capital * 0.25 / close  # Never use more than 25% of capital
                )
                position = -1
                
                # Record trade
                current_trade = {
                    'entry_date': date,
                    'entry_price': entry_price,
                    'type': 'SHORT',
                    'size': position_size,
                    'stop_loss': min(result['stop_loss_short'].iloc[i], entry_price * 1.05),  # Max 5% stop
                    'take_profit': max(result['take_profit_short'].iloc[i], entry_price * 0.9)  # Min 10% take profit
                }
    
    # Close any open position at the end
    if position != 0:
        close = result['close'].iloc[-1]
        date = result.index[-1]
        
        if position == 1:  # Long position
            profit = (close - entry_price) * current_trade['size']
        else:  # Short position
            profit = (entry_price - close) * current_trade['size']
            
        capital += profit
        
        # Record trade
        current_trade['exit_date'] = date
        current_trade['exit_price'] = close
        current_trade['profit'] = profit
        current_trade['return'] = profit / (entry_price * current_trade['size']) * 100
        trades.append(current_trade)
    
    # Create trades DataFrame
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Calculate performance metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        total_return = (capital - initial_capital) / initial_capital * 100
        
        avg_profit = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trades_df[trades_df['profit'] > 0]['profit'].sum() / 
                            trades_df[trades_df['profit'] < 0]['profit'].sum()) if losing_trades > 0 else float('inf')
        
        # Calculate drawdown
        equity_curve = [initial_capital]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['profit'])
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (running_max - equity_curve) / running_max * 100
        max_drawdown = np.max(drawdown)
        
        # Prepare performance summary
        performance = {
            'Initial Capital': initial_capital,
            'Final Capital': capital,
            'Total Return [%]': total_return,
            'Total Trades': total_trades,
            'Win Rate [%]': win_rate,
            'Profit Factor': profit_factor,
            'Average Profit': avg_profit,
            'Average Loss': avg_loss,
            'Max Drawdown [%]': max_drawdown
        }
        
        # Optional plotting
        if plot:
            plot_signals(result)
            
            # Plot equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(np.array(equity_curve))
            plt.title('Equity Curve')
            plt.xlabel('Trade #')
            plt.ylabel('Capital')
            plt.grid(True)
            plt.show()
            
        return performance, trades_df
    
    else:
        print("No trades were found. Check signal thresholds and data quality.")
        # No trades executed
        return {
            'Initial Capital': initial_capital,
            'Final Capital': capital,
            'Total Return [%]': 0,
            'Total Trades': 0,
            'Win Rate [%]': 0,
            'Profit Factor': 0,
            'Average Profit': 0,
            'Average Loss': 0,
            'Max Drawdown [%]': 0
        }, pd.DataFrame()


def run_quantum_hybrid_strategy_backtest(symbol: str, timeframe: str, initial_capital: float, 
                                      start_date: str, end_date: str, 
                                      params: Optional[QuantumHybridParameters] = None, 
                                      plot: bool = False) -> Tuple[Dict, pd.DataFrame]:
    """
    Run a backtest for the Quantum Harmonic-Wave Hybrid model with the given parameters.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe for candlestick data (e.g., '1h', '4h')
        initial_capital: Initial capital for the backtest
        start_date: Start date for the backtest (format: 'YYYY-MM-DD')
        end_date: End date for the backtest (format: 'YYYY-MM-DD')
        params: Parameters for the model (optional)
        plot: Whether to plot the backtest results
        
    Returns:
        Tuple of (performance_metrics, trades_df)
    """
    from app import fetch_crypto_data
    
    # Fetch data
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    if df is None or len(df) == 0:
        print(f"No data available for {symbol} on {timeframe} from {start_date} to {end_date}")
        return None, None
    
    # Run backtest
    print(f"Running Quantum Harmonic-Wave Hybrid backtest for {symbol} on {timeframe}...")
    performance, trades_df = backtest_hybrid_strategy(df, params, initial_capital, plot)
    
    # Print summary
    print("\n--- Quantum Harmonic-Wave Hybrid Strategy Results ---")
    for key, value in performance.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    return performance, trades_df 