#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantum Harmonic-Wave Hybrid Model with EMA Slope Regime Filter

This module implements the original hybrid quantum model combining aspects of
the Quantum Stock Conservative and Aggressive models. It uses an EMA slope
filter to determine the market regime (trend direction) and only allows trades
in the direction of the identified trend. Exits are handled by the exit signal
or the initial ATR-based stop loss.

Enhancements compared to original:
1. EMA Slope Regime Filter: Determines trend direction to filter entries.
2. Standard ATR Stop Loss: Uses a standard initial ATR stop loss (no trailing).
"""

import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import norm
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Helper function to calculate EMA slope
def calculate_ema_slope(series: pd.Series, period: int = 5) -> pd.Series:
    """Calculate the slope of the EMA using linear regression over a rolling window."""
    # Ensure the series is float type
    series = series.astype(float)
    # Create sequence for linear regression
    x = np.arange(period)
    # Apply rolling linear regression
    slopes = series.rolling(window=period).apply(
        lambda y: np.polyfit(x, y, 1)[0] if not np.isnan(y).any() and len(y) == period else np.nan,
        raw=True
    )
    return slopes

class QuantumHybridEnhancedParameters:
    """
    Parameter class for the Quantum Harmonic-Wave Hybrid model with EMA Slope Filter.
    """
    def __init__(self):
        # --- Core Quantum Parameters ---
        self.hbar = 1.0
        self.m = 1.0
        self.mass = 1.0
        self.omega = 1.0e-4
        self.lambda_param = 0.5

        # --- Hybrid Weight Parameters ---
        self.conservative_weight = 0.55
        self.aggressive_weight = 0.45

        # --- Signal Generation Parameters ---
        self.entry_threshold = 0.6
        self.exit_threshold = 0.5 # Threshold for the *opposite* signal to trigger an exit
        self.volatility_adaptation = True
        self.momentum_factor = 0.6

        # --- Base Risk Management Parameters ---
        self.stop_loss_atr_factor = 1.4 # Multiplier for initial ATR stop loss
        self.atr_period = 14
        self.rsi_period = 14
        self.rsi_threshold_upper = 75 # Used in conservative signal calculation
        self.rsi_threshold_lower = 28 # Used in conservative signal calculation
        self.rsi_weight = 0.25 # Implicitly used in conservative signal calculation

        # --- Market Imbalance Parameters ---
        self.volume_ema_period = 20
        self.imbalance_lookback = 6
        self.imbalance_threshold = 0.4 # Used? Check calculation

        # --- Enhancement Parameters (EMA Slope Filter) ---
        self.use_regime_filter = True # Enable EMA Slope filter
        self.regime_ema_period = 50   # Period for the trend EMA
        self.regime_slope_period = 5    # Lookback period for calculating slope
        self.regime_slope_threshold = 1e-6 # Minimum absolute slope to define trend

        # --- Removed Parameters ---
        # self.use_atr_trailing_stop = False # Removed trailing stop
        # self.trailing_stop_atr_multiplier = 1.8 # Removed trailing stop

        # --- Other Parameters ---
        self.dynamic_timeframe = True # Placeholder, not directly used in backtest function
        self.risk_pct = 1.0 # Percentage of capital to risk per trade
        self.use_adaptive_position_sizing = True # Enable position sizing based on risk_pct and stop distance

        # --- Realism Parameters (New) ---
        self.commission_pct = 0.00075 # Example: 0.075% Taker fee on Binance Futures
        self.slippage_pct = 0.0005  # Example: 0.05% slippage per side (0.1% total round trip)


# --- Calculation Functions (Mostly unchanged from previous version) ---

def calculate_market_imbalance(df, params):
    # (Same as in original quantum_stock_hybrid_model.py)
    df['volume_ema'] = df['volume'].ewm(span=params.volume_ema_period, adjust=False).mean()
    df['price_direction'] = np.sign(df['close'].diff(1)).fillna(0)
    df['buying_volume'] = df['volume'] * (df['price_direction'] > 0).astype(float)
    df['selling_volume'] = df['volume'] * (df['price_direction'] < 0).astype(float)
    rolling_vol_sum = df['volume'].rolling(window=params.imbalance_lookback).sum()
    df['buying_pressure'] = df['buying_volume'].rolling(window=params.imbalance_lookback).sum() / (rolling_vol_sum + 1e-8)
    df['selling_pressure'] = df['selling_volume'].rolling(window=params.imbalance_lookback).sum() / (rolling_vol_sum + 1e-8)
    df['market_imbalance'] = df['buying_pressure'] - df['selling_pressure']
    df['market_imbalance'].fillna(0, inplace=True)
    imbalance_rolling_min = df['market_imbalance'].rolling(window=100).min()
    imbalance_rolling_max = df['market_imbalance'].rolling(window=100).max()
    df['market_imbalance_normalized'] = 2 * (df['market_imbalance'] - imbalance_rolling_min) / \
                                       (imbalance_rolling_max - imbalance_rolling_min + 1e-8) - 1
    df['market_imbalance_normalized'].fillna(0, inplace=True)
    df['market_imbalance_normalized'] = df['market_imbalance_normalized'].clip(-1, 1)
    return df

def calculate_volatility(df, params):
     # (Same as in original quantum_stock_hybrid_model.py - calculates 'atr')
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].ewm(span=params.atr_period, adjust=False).mean() # Use EWM for smoother ATR
    df['returns'] = df['close'].pct_change()
    df['return_volatility'] = df['returns'].rolling(window=20).std()
    df['normalized_atr'] = df['atr'] / (df['close'] + 1e-8)
    df['volatility'] = (df['normalized_atr'] + df['return_volatility']) / 2
    df['volatility'] = df['volatility'].fillna(df['volatility'].mean())
    return df

def calculate_energy_levels(df, params):
    # (Same as in original quantum_stock_hybrid_model.py)
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['price_displacement'] = (df['close'] - rolling_mean) / (rolling_std + 1e-8)
    df['price_velocity'] = df['price_displacement'].diff(1)
    df['kinetic_energy'] = 0.5 * params.m * df['price_velocity']**2
    df['potential_energy'] = 0.5 * params.m * params.omega**2 * df['price_displacement']**2
    df['total_energy'] = df['kinetic_energy'] + df['potential_energy']
    df['quantum_level'] = np.round(df['total_energy'] / (params.hbar * params.omega + 1e-8) - 0.5)
    df['quantum_level'] = df['quantum_level'].clip(0, 10) 
    max_levels = 10  
    df['normalized_energy'] = df['quantum_level'] / max_levels
    df['energy_change'] = df['quantum_level'].diff(1)
    df['significant_energy_change'] = (abs(df['energy_change']) >= 1).astype(int)
    return df

def calculate_conservative_signals(df, params):
    # (Same as in original quantum_stock_hybrid_model.py)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(span=params.rsi_period, adjust=False).mean() # Use EWM
    avg_loss = loss.ewm(span=params.rsi_period, adjust=False).mean() # Use EWM
    rs = avg_gain / (avg_loss + 1e-8) 
    df['rsi'] = 100 - (100 / (1 + rs))
    rolling_min = df['close'].rolling(window=50).min()
    rolling_max = df['close'].rolling(window=50).max()
    df['price_position'] = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
    df['conservative_signal_raw'] = (
        0.45 * (1 - df['normalized_energy']) +  
        0.35 * df['market_imbalance_normalized'] +  
        0.2 * (1 - df['price_position'])  
    )
    rsi_factor = np.where(
        df['rsi'] < params.rsi_threshold_lower, 1 + (params.rsi_threshold_lower - df['rsi']) / 50, 0.8 )
    df['conservative_signal_raw'] = df['conservative_signal_raw'] * rsi_factor
    rolling_min_sig = df['conservative_signal_raw'].rolling(window=20).min()
    rolling_max_sig = df['conservative_signal_raw'].rolling(window=20).max()
    df['conservative_signal'] = (df['conservative_signal_raw'] - rolling_min_sig) / \
                                (rolling_max_sig - rolling_min_sig + 1e-8)
    df['conservative_signal'] = df['conservative_signal'].fillna(0.5).clip(0, 1)
    return df

def calculate_aggressive_signals(df, params):
    # (Same as in original quantum_stock_hybrid_model.py)
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_diff'] = (df['ema_fast'] - df['ema_slow']) / (df['close'] + 1e-8) # Normalized difference

    # MACD calculation (can be kept as it's used in the signal)
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal_line'] = df['macd'].ewm(span=9, adjust=False).mean() # Renamed for clarity
    df['macd_hist'] = df['macd'] - df['macd_signal_line']
    df['macd_direction'] = np.sign(df['macd_hist'].diff(3)).fillna(0) # Using sign of change over 3 periods

    # Calculate RSI if not already present (needed for signal)
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(span=params.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=params.rsi_period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'].fillna(50, inplace=True) # Handle initial NaNs

    df['aggressive_signal_raw'] = (
        0.35 * df['market_imbalance_normalized'] +  # Market structure
        0.25 * df['significant_energy_change'] +   # Energy state change
        0.25 * df['ema_diff'] +                    # Short-term trend (normalized)
        0.15 * df['macd_direction'] +              # MACD momentum
        0.15 * (df['rsi'] - 50) / 50               # RSI momentum (normalized)
    )

    # Apply volatility adaptation if enabled
    if params.volatility_adaptation:
        vol_mean = df['volatility'].rolling(window=100).mean()
        vol_ratio = df['volatility'] / (vol_mean + 1e-8)
        # Cap the volatility scaling factor to prevent extreme values
        vol_scaling = vol_ratio.clip(0.5, 2.0)
        df['aggressive_signal_raw'] = df['aggressive_signal_raw'] * vol_scaling

    # Normalize the raw signal to be between 0 and 1 using rolling min/max
    rolling_min_agg = df['aggressive_signal_raw'].rolling(window=10).min()
    rolling_max_agg = df['aggressive_signal_raw'].rolling(window=10).max()
    df['aggressive_signal'] = (df['aggressive_signal_raw'] - rolling_min_agg) / \
                              (rolling_max_agg - rolling_min_agg + 1e-8)

    # Fill initial NaNs and clip the final signal
    df['aggressive_signal'] = df['aggressive_signal'].fillna(0.5).clip(0, 1)

    return df

def calculate_regime_filter(df, params):
    """ Calculate the regime filter based on EMA slope. """
    print("    Calculating regime filter...")
    try:
        if params.use_regime_filter:
            ema_col = f'ema_{params.regime_ema_period}'
            df[ema_col] = df['close'].ewm(span=params.regime_ema_period, adjust=False).mean()
            df['ema_slope'] = calculate_ema_slope(df[ema_col], period=params.regime_slope_period)

            # Define regime: 1 for uptrend, -1 for downtrend, 0 for flat/chop
            df['regime'] = 0 # Default to flat
            df.loc[df['ema_slope'] > params.regime_slope_threshold, 'regime'] = 1
            df.loc[df['ema_slope'] < -params.regime_slope_threshold, 'regime'] = -1
            df['regime'].fillna(0, inplace=True) # Fill NaNs from slope calculation
            print(f"    Regime filter calculated (EMA{params.regime_ema_period}, Slope{params.regime_slope_period}, Threshold {params.regime_slope_threshold}).")
        else:
            # If filter is disabled, allow all trades
            df['regime'] = 1 # Or could use 0 if that's preferred default state
            df['ema_slope'] = 0
            df[f'ema_{params.regime_ema_period}'] = np.nan # Ensure column exists if plotting needs it
            print("    Regime filter disabled.")
    except Exception as e:
        print(f"    ERROR calculating regime filter: {e}")
        # Decide how to handle error: maybe default to no trades (0) or allow all (1)?
        df['regime'] = 0
        df['ema_slope'] = 0
        df[f'ema_{params.regime_ema_period}'] = np.nan

    return df


def run_quantum_hybrid_enhanced_strategy_backtest(symbol, timeframe, initial_capital, start_date, end_date, params=None):
    """
    Run the backtest for the Quantum Hybrid strategy with EMA Slope Regime Filter.
    Uses standard ATR initial stop loss.
    """
    if params is None:
        params = QuantumHybridEnhancedParameters()

    # --- Data Fetching ---
    from app import fetch_crypto_data # Assuming fetch_crypto_data is in app module
    print(f"Fetching data for {symbol} on {timeframe} timeframe from {start_date} to {end_date}...")
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    if df is None or df.empty:
        print(f"Error fetching data for {symbol} {timeframe} or data is empty.")
        return None, None, None # Return None for stats, df, trades_df
    print(f"Data fetched: {len(df)} candles from {df.index.min()} to {df.index.max()}")

    # --- Indicator Calculations (with detailed logging) ---
    print("  Calculating volatility...")
    try:
        df = calculate_volatility(df, params)
        print("  Volatility calculated.")
    except Exception as e:
        print(f"  ERROR calculating volatility: {e}")
        return None, None, None

    print("  Calculating market imbalance...")
    try:
        df = calculate_market_imbalance(df, params)
        print("  Market imbalance calculated.")
    except Exception as e:
        print(f"  ERROR calculating market imbalance: {e}")
        return None, None, None

    print("  Calculating energy levels...")
    try:
        df = calculate_energy_levels(df, params)
        print("  Energy levels calculated.")
    except Exception as e:
        print(f"  ERROR calculating energy levels: {e}")
        return None, None, None

    print("  Calculating conservative signals...")
    try:
        df = calculate_conservative_signals(df, params)
        print("  Conservative signals calculated.")
    except Exception as e:
        print(f"  ERROR calculating conservative signals: {e}")
        return None, None, None

    print("  Calculating aggressive signals...")
    try:
        df = calculate_aggressive_signals(df, params)
        print("  Aggressive signals calculated.")
    except Exception as e:
        print(f"  ERROR calculating aggressive signals: {e}")
        return None, None, None

    # --- Regime Filter Calculation ---
    try:
        df = calculate_regime_filter(df, params) # Calculate regime based on EMA slope
    except Exception as e:
        print(f"  ERROR calculating regime filter: {e}")
        return None, None, None # Stop if regime calculation fails


    # --- Combined Signal ---
    print("  Calculating combined signal...")
    try:
        df['combined_signal'] = (params.conservative_weight * df['conservative_signal'] +
                                 params.aggressive_weight * df['aggressive_signal'])
        df['combined_signal'] = df['combined_signal'].fillna(0.5).clip(0, 1) # Ensure bounds 0-1
        # Calculate an "exit" signal (inverse for exiting opposite positions)
        df['exit_signal'] = 1.0 - df['combined_signal']
        print("  Combined signal calculated.")
    except Exception as e:
        print(f"  ERROR calculating combined signal: {e}")
        return None, None, None

    # --- Backtesting Loop ---
    print("Starting backtest loop...")
    capital = initial_capital
    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    equity_curve = [initial_capital]
    trades = []
    position_size_pct = params.risk_pct / 100.0
    entry_time = None # Initialize entry_time outside the loop

    for i in range(1, len(df)):
        # <<< Early Stop Check >>>
        if capital <= 1.0:
            print(f"{df.index[i]} - Capital depleted (${capital:.2f}). Stopping backtest early.")
            equity_curve.extend([capital] * (len(df) - i))
            break

        current_price = df['close'].iloc[i]
        current_atr = df['atr'].iloc[i]
        current_regime = df['regime'].iloc[i] if params.use_regime_filter else 1
        entry_signal = df['combined_signal'].iloc[i]
        exit_signal = df['exit_signal'].iloc[i]

        # --- Exit Logic ---
        if position > 0: # In a long position
            if current_price <= stop_loss:
                # Apply slippage to exit price (favorable for SL)
                exit_price = stop_loss * (1 - params.slippage_pct)
                pnl_before_fees = position * (exit_price - entry_price)
                commission = abs(position * exit_price * params.commission_pct) + entry_commission # Total commission
                pnl_after_fees = pnl_before_fees - commission
                capital += pnl_after_fees
                # Use df.index[i] for exit time, ensure entry_time is available
                trade_entry_time = entry_time if entry_time is not None else df.index[i] # Fallback if entry_time not set
                trades.append({'entry_time': trade_entry_time, 'exit_time': df.index[i], 'entry_price': entry_price, 'exit_price': exit_price, 'size': position, 'pnl': pnl_after_fees, 'commission': commission, 'type': 'Stop Loss Long'})
                print(f"{df.index[i]} - Stop Loss Hit (Long): Exit at {exit_price:.4f}, PnL: {pnl_after_fees:.2f} (Fees: {commission:.2f})")
                position = 0
                entry_time = None # Reset entry time
                entry_commission = 0 # Reset entry commission
            elif exit_signal > params.exit_threshold:
                # Apply slippage to exit price (unfavorable for market exit)
                exit_price = current_price * (1 - params.slippage_pct)
                pnl_before_fees = position * (exit_price - entry_price)
                commission = abs(position * exit_price * params.commission_pct) + entry_commission
                pnl_after_fees = pnl_before_fees - commission
                capital += pnl_after_fees
                trade_entry_time = entry_time if entry_time is not None else df.index[i]
                trades.append({'entry_time': trade_entry_time, 'exit_time': df.index[i], 'entry_price': entry_price, 'exit_price': exit_price, 'size': position, 'pnl': pnl_after_fees, 'commission': commission, 'type': 'Exit Signal Long'})
                print(f"{df.index[i]} - Exit Signal (Long): Exit at {exit_price:.4f}, PnL: {pnl_after_fees:.2f} (Fees: {commission:.2f})")
                position = 0
                entry_time = None # Reset entry time
                entry_commission = 0 # Reset entry commission

        elif position < 0: # In a short position
            if current_price >= stop_loss:
                # Apply slippage to exit price (favorable for SL)
                exit_price = stop_loss * (1 + params.slippage_pct)
                pnl_before_fees = position * (exit_price - entry_price)
                commission = abs(position * exit_price * params.commission_pct) + entry_commission
                pnl_after_fees = pnl_before_fees - commission
                capital += pnl_after_fees
                trade_entry_time = entry_time if entry_time is not None else df.index[i]
                trades.append({'entry_time': trade_entry_time, 'exit_time': df.index[i], 'entry_price': entry_price, 'exit_price': exit_price, 'size': position, 'pnl': pnl_after_fees, 'commission': commission, 'type': 'Stop Loss Short'})
                print(f"{df.index[i]} - Stop Loss Hit (Short): Exit at {exit_price:.4f}, PnL: {pnl_after_fees:.2f} (Fees: {commission:.2f})")
                position = 0
                entry_time = None # Reset entry time
                entry_commission = 0 # Reset entry commission
            elif exit_signal > params.exit_threshold:
                # Apply slippage to exit price (unfavorable for market exit)
                exit_price = current_price * (1 + params.slippage_pct)
                pnl_before_fees = position * (exit_price - entry_price)
                commission = abs(position * exit_price * params.commission_pct) + entry_commission
                pnl_after_fees = pnl_before_fees - commission
                capital += pnl_after_fees
                trade_entry_time = entry_time if entry_time is not None else df.index[i]
                trades.append({'entry_time': trade_entry_time, 'exit_time': df.index[i], 'entry_price': entry_price, 'exit_price': exit_price, 'size': position, 'pnl': pnl_after_fees, 'commission': commission, 'type': 'Exit Signal Short'})
                print(f"{df.index[i]} - Exit Signal (Short): Exit at {exit_price:.4f}, PnL: {pnl_after_fees:.2f} (Fees: {commission:.2f})")
                position = 0
                entry_time = None # Reset entry time
                entry_commission = 0 # Reset entry commission

        # --- Entry Logic ---
        if position == 0:
            sl_atr = current_atr if current_atr > 0 else df['atr'].mean()

            if entry_signal > params.entry_threshold and current_regime == 1:
                # Apply slippage to entry price (unfavorable)
                entry_price = current_price * (1 + params.slippage_pct)
                stop_loss = entry_price - params.stop_loss_atr_factor * sl_atr
                # Ensure SL is below entry
                if stop_loss >= entry_price: stop_loss = entry_price * (1 - 0.001)
                risk_per_share = entry_price - stop_loss
                if risk_per_share <= 0: risk_per_share = entry_price * 0.01 # Fallback risk
                
                # Position size calculation (remains the same logic)
                if params.use_adaptive_position_sizing: size = (capital * position_size_pct) / risk_per_share
                else: size = capital * 0.1 / entry_price # Basic sizing fallback
                
                # Calculate entry commission and deduct from capital
                entry_commission = abs(size * entry_price * params.commission_pct)
                capital -= entry_commission
                
                position = size
                entry_time = df.index[i] # Record entry time
                print(f"{df.index[i]} - Long Entry: Price={entry_price:.4f}, SL={stop_loss:.4f}, Size={position:.4f}, Fees:{entry_commission:.2f}")

            elif entry_signal < (1.0 - params.entry_threshold) and current_regime == -1:
                # Apply slippage to entry price (unfavorable)
                entry_price = current_price * (1 - params.slippage_pct)
                stop_loss = entry_price + params.stop_loss_atr_factor * sl_atr
                # Ensure SL is above entry
                if stop_loss <= entry_price: stop_loss = entry_price * (1 + 0.001)
                risk_per_share = stop_loss - entry_price
                if risk_per_share <= 0: risk_per_share = entry_price * 0.01 # Fallback risk
                
                # Position size calculation
                if params.use_adaptive_position_sizing: size = (capital * position_size_pct) / risk_per_share
                else: size = capital * 0.1 / entry_price # Basic sizing fallback
                
                # Calculate entry commission and deduct from capital
                entry_commission = abs(size * entry_price * params.commission_pct)
                capital -= entry_commission
                
                position = -size
                entry_time = df.index[i] # Record entry time
                print(f"{df.index[i]} - Short Entry: Price={entry_price:.4f}, SL={stop_loss:.4f}, Size={position:.4f}, Fees:{entry_commission:.2f}")

        # --- Update Equity Curve ---
        if capital <= 0:
            print(f"Capital wiped out at {df.index[i]}. Stopping backtest.")
            capital = 0
            equity_curve.append(capital)
            # Need to ensure equity curve matches length later if breaking early
            break

        current_equity = capital
        if position > 0: current_equity += position * (current_price - entry_price)
        elif position < 0: current_equity += position * (current_price - entry_price)
        equity_curve.append(current_equity if current_equity > 0 else 0)

    # --- Final Position Close --- (This block should be OUTSIDE the loop)
    if position > 0:
        # Apply slippage and commission on final close
        exit_price = df['close'].iloc[-1] * (1 - params.slippage_pct)
        pnl_before_fees = position * (exit_price - entry_price)
        commission = abs(position * exit_price * params.commission_pct) + entry_commission
        pnl_after_fees = pnl_before_fees - commission
        capital += pnl_after_fees
        trade_entry_time = entry_time if entry_time is not None else df.index[-1] # Use last known entry or fallback
        trades.append({'entry_time': trade_entry_time, 'exit_time': df.index[-1], 'entry_price': entry_price, 'exit_price': exit_price, 'size': position, 'pnl': pnl_after_fees, 'commission': commission, 'type': 'Close Long EOD'})
        print(f"{df.index[-1]} - Closing Long position at EOD: PnL {pnl_after_fees:.2f} (Fees: {commission:.2f})")
        equity_curve[-1] = capital
    elif position < 0:
        # Apply slippage and commission on final close
        exit_price = df['close'].iloc[-1] * (1 + params.slippage_pct)
        pnl_before_fees = position * (exit_price - entry_price)
        commission = abs(position * exit_price * params.commission_pct) + entry_commission
        pnl_after_fees = pnl_before_fees - commission
        capital += pnl_after_fees
        trade_entry_time = entry_time if entry_time is not None else df.index[-1]
        trades.append({'entry_time': trade_entry_time, 'exit_time': df.index[-1], 'entry_price': entry_price, 'exit_price': exit_price, 'size': position, 'pnl': pnl_after_fees, 'commission': commission, 'type': 'Close Short EOD'})
        print(f"{df.index[-1]} - Closing Short position at EOD: PnL {pnl_after_fees:.2f} (Fees: {commission:.2f})")
        equity_curve[-1] = capital

    # --- Performance Calculation --- (This block should be OUTSIDE the loop)
    print("Calculating performance metrics...")
    # Adjust equity curve length if loop broke early
    if len(equity_curve) < len(df) + 1:
        equity_curve.extend([equity_curve[-1]] * (len(df) + 1 - len(equity_curve)))

    # Assign to DataFrame
    if len(equity_curve) == len(df) + 1:
         df['equity'] = equity_curve[:-1]
    else:
         print(f"Error: Equity curve length ({len(equity_curve)}) still incorrect after adjustment.")
         # Fallback: Assign what we have, padded
         df['equity'] = equity_curve[:len(df)] + [equity_curve[-1]] * (len(df) - len(equity_curve))
         df['equity'] = df['equity'][:len(df)] # Ensure exact length

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("No trades executed.")
        stats = {
            "total_trades": 0, "win_rate": 0, "total_return": 0,
            "max_drawdown": 0, "profit_factor": 0, "risk_adjusted_return": 0
        }
        trades_df = pd.DataFrame(columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'size', 'pnl', 'type'])
        return stats, df, trades_df

    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

    trades_df['pnl_pct'] = (trades_df['pnl'] / initial_capital) * 100
    total_return = ( (capital / initial_capital) - 1 ) * 100
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] < 0]
    win_rate = (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    gross_profit = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max Drawdown calculation
    equity_index = df.index[:len(equity_curve)-1] # Index should match df length
    equity_series = pd.Series(equity_curve[:-1], index=equity_index)

    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / (rolling_max + 1e-8)
    max_drawdown = abs(drawdown.min()) * 100 if not drawdown.empty else 0

    risk_adjusted_return = total_return / (max_drawdown + 1e-8)

    stats = {
        "total_trades": len(trades_df),
        "win_rate": win_rate,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "risk_adjusted_return": risk_adjusted_return,
    }

    return stats, df, trades_df

# --- Example Usage & Plotting (Optional) ---
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import os
    from app import fetch_crypto_data # Re-import for standalone run

    # Example parameters
    symbol = 'ETH/USDT'
    timeframe = '1h'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    initial_capital = 10000

    # Use enhanced parameters
    params = QuantumHybridEnhancedParameters()
    # Example adjustments (optional):
    # params.use_regime_filter = True
    # params.regime_slope_threshold = 1e-7
    # params.stop_loss_atr_factor = 2.0

    print(f"Running Enhanced Quantum Hybrid backtest for {symbol} {timeframe}...")
    stats, df_results, trades_df = run_quantum_hybrid_enhanced_strategy_backtest(
        symbol, timeframe, initial_capital, start_date, end_date, params
    )

    if stats is None:
        print("Backtest failed to produce results.")
    else:
        print("\n----- Backtest Performance Metrics -----")
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")

        print("\nPlotting comprehensive results...")

        # Create plot directory if it doesn't exist
        plot_dir = "static/plots"
        os.makedirs(plot_dir, exist_ok=True)

        # Increase number of subplots for more indicators + drawdown
        fig, axes = plt.subplots(8, 1, figsize=(16, 28), sharex=True,
                                 gridspec_kw={'height_ratios': [4, 1, 1, 1, 1, 1, 2, 2]})
        fig.suptitle(f'Quantum Hybrid Enhanced Strategy ({symbol} {timeframe}) - Comprehensive Analysis\n'
                     f'EMA Slope Filter: {params.use_regime_filter}, Period={params.regime_ema_period}, Slope={params.regime_slope_period}, Thresh={params.regime_slope_threshold:.1e}',
                     fontsize=16)

        # 1. Price, EMA Trend, and Trades
        ax_price = axes[0]
        ax_price.plot(df_results.index, df_results['close'], label='Close Price', color='black', alpha=0.8, linewidth=1)
        if params.use_regime_filter and f'ema_{params.regime_ema_period}' in df_results.columns:
            ax_price.plot(df_results.index, df_results[f'ema_{params.regime_ema_period}'], label=f'EMA {params.regime_ema_period}', color='blue', linestyle='--', alpha=0.7)
        ax_price.set_ylabel('Price')
        ax_price.set_title('Price, Trend EMA, and Trades')
        ax_price.grid(True, linestyle='--', alpha=0.5)

        # Plot trades on price chart
        if not trades_df.empty:
            long_entries = trades_df[trades_df['size'] > 0]
            short_entries = trades_df[trades_df['size'] < 0]
            ax_price.plot(long_entries['entry_time'], long_entries['entry_price'], '^', markersize=8, color='lime', label='Long Entry', alpha=0.9)
            ax_price.plot(short_entries['entry_time'], short_entries['entry_price'], 'v', markersize=8, color='red', label='Short Entry', alpha=0.9)
            ax_price.plot(trades_df['exit_time'], trades_df['exit_price'], 'x', markersize=6, color='orange', label='Exit', alpha=0.9)
        ax_price.legend(loc='upper left')

        # 2. EMA Slope and Regime
        ax_slope = axes[1]
        if params.use_regime_filter and 'ema_slope' in df_results.columns:
            ax_regime = ax_slope.twinx() # Secondary y-axis for regime
            color_slope = 'purple'
            ax_slope.plot(df_results.index, df_results['ema_slope'], label=f'EMA {params.regime_ema_period} Slope ({params.regime_slope_period}p)', color=color_slope, alpha=0.8)
            ax_slope.axhline(params.regime_slope_threshold, color=color_slope, linestyle=':', alpha=0.7)
            ax_slope.axhline(-params.regime_slope_threshold, color=color_slope, linestyle=':', alpha=0.7)
            ax_slope.set_ylabel('EMA Slope', color=color_slope)
            ax_slope.tick_params(axis='y', labelcolor=color_slope)
            ax_slope.grid(True, axis='y', linestyle='--', alpha=0.3)

            color_regime = 'gray'
            ax_regime.fill_between(df_results.index, 0, 1, where=df_results['regime'] == 1, color='green', alpha=0.15, label='Uptrend')
            ax_regime.fill_between(df_results.index, -1, 0, where=df_results['regime'] == -1, color='red', alpha=0.15, label='Downtrend')
            # ax_regime.plot(df_results.index, df_results['regime'], label='Regime', color=color_regime, linestyle='-.', alpha=0.6, drawstyle='steps-post') # Optional line plot for regime
            ax_regime.set_ylabel('Regime', color=color_regime)
            ax_regime.tick_params(axis='y', labelcolor=color_regime)
            ax_regime.set_ylim(-1.1, 1.1)
            ax_regime.set_yticks([-1, 0, 1])
            ax_regime.legend(loc='upper right')
            ax_slope.set_title('EMA Slope and Market Regime')
        else:
             ax_slope.text(0.5, 0.5, 'Regime Filter Disabled', horizontalalignment='center', verticalalignment='center', transform=ax_slope.transAxes)
             ax_slope.set_yticks([])


        # 3. Combined Signal
        ax_signal = axes[2]
        ax_signal.plot(df_results.index, df_results['combined_signal'], label='Combined Signal', color='cyan', alpha=0.8)
        ax_signal.axhline(params.entry_threshold, color='lime', linestyle='--', alpha=0.7, label=f'Long Entry ({params.entry_threshold:.2f})')
        ax_signal.axhline(1.0 - params.entry_threshold, color='tomato', linestyle='--', alpha=0.7, label=f'Short Entry ({(1.0 - params.entry_threshold):.2f})')
        # ax_signal.axhline(params.exit_threshold, color='orange', linestyle=':', alpha=0.7, label=f'Exit ({params.exit_threshold:.2f})') # Consider adding exit signal plot too
        ax_signal.set_ylabel('Signal (0-1)')
        ax_signal.set_ylim(-0.05, 1.05)
        ax_signal.set_title('Combined Quantum Signal')
        ax_signal.grid(True, linestyle='--', alpha=0.5)
        ax_signal.legend(loc='upper left', fontsize='small')

        # 4. RSI
        ax_rsi = axes[3]
        if 'rsi' in df_results.columns:
            ax_rsi.plot(df_results.index, df_results['rsi'], label=f'RSI ({params.rsi_period})', color='orange', alpha=0.8)
            ax_rsi.axhline(params.rsi_threshold_upper, color='red', linestyle=':', alpha=0.7, label=f'Upper ({params.rsi_threshold_upper})')
            ax_rsi.axhline(params.rsi_threshold_lower, color='green', linestyle=':', alpha=0.7, label=f'Lower ({params.rsi_threshold_lower})')
            ax_rsi.axhline(50, color='gray', linestyle='--', alpha=0.5)
            ax_rsi.set_ylabel('RSI')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_title('RSI')
            ax_rsi.grid(True, linestyle='--', alpha=0.5)
            ax_rsi.legend(loc='upper left', fontsize='small')
        else:
             ax_rsi.text(0.5, 0.5, 'RSI Not Calculated/Available', horizontalalignment='center', verticalalignment='center', transform=ax_rsi.transAxes)
             ax_rsi.set_yticks([])

        # 5. MACD Histogram
        ax_macd = axes[4]
        if 'macd_hist' in df_results.columns:
            colors = ['green' if x >= 0 else 'red' for x in df_results['macd_hist']]
            ax_macd.bar(df_results.index, df_results['macd_hist'], label='MACD Histogram', color=colors, alpha=0.7)
            ax_macd.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax_macd.set_ylabel('MACD Hist')
            ax_macd.set_title('MACD Histogram (12, 26, 9)')
            ax_macd.grid(True, linestyle='--', alpha=0.5)
            # ax_macd.legend(loc='upper left', fontsize='small') # Legend less useful for histogram bars
        else:
             ax_macd.text(0.5, 0.5, 'MACD Not Calculated/Available', horizontalalignment='center', verticalalignment='center', transform=ax_macd.transAxes)
             ax_macd.set_yticks([])

        # 6. Normalized Energy
        ax_energy = axes[5]
        if 'normalized_energy' in df_results.columns:
            ax_energy.plot(df_results.index, df_results['normalized_energy'], label='Normalized Energy', color='magenta', alpha=0.8)
            ax_energy.set_ylabel('Energy (0-1)')
            ax_energy.set_ylim(-0.05, 1.05)
            ax_energy.set_title('Normalized Quantum Energy Level')
            ax_energy.grid(True, linestyle='--', alpha=0.5)
            ax_energy.legend(loc='upper left', fontsize='small')
        else:
            ax_energy.text(0.5, 0.5, 'Energy Not Calculated/Available', horizontalalignment='center', verticalalignment='center', transform=ax_energy.transAxes)
            ax_energy.set_yticks([])

        # 7. Market Imbalance
        ax_imbalance = axes[6]
        if 'market_imbalance_normalized' in df_results.columns:
            ax_imbalance.plot(df_results.index, df_results['market_imbalance_normalized'], label='Market Imbalance (Norm)', color='brown', alpha=0.8)
            ax_imbalance.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax_imbalance.set_ylabel('Imbalance (-1 to 1)')
            ax_imbalance.set_ylim(-1.05, 1.05)
            ax_imbalance.set_title(f'Normalized Market Imbalance ({params.imbalance_lookback}p)')
            ax_imbalance.grid(True, linestyle='--', alpha=0.5)
            ax_imbalance.legend(loc='upper left', fontsize='small')
        else:
            ax_imbalance.text(0.5, 0.5, 'Imbalance Not Calculated/Available', horizontalalignment='center', verticalalignment='center', transform=ax_imbalance.transAxes)
            ax_imbalance.set_yticks([])

        # 8. Equity Curve and Drawdown
        ax_equity = axes[7]
        ax_drawdown = ax_equity.twinx() # Secondary y-axis for drawdown

        if 'equity' in df_results.columns:
            equity_series = df_results['equity']
            # Equity Curve plot
            color_equity = 'teal'
            ax_equity.plot(equity_series.index, equity_series, label='Equity Curve', color=color_equity, linewidth=1.5)
            ax_equity.set_ylabel('Portfolio Value', color=color_equity)
            ax_equity.tick_params(axis='y', labelcolor=color_equity)
            ax_equity.set_title('Equity Curve and Drawdown')
            ax_equity.grid(True, linestyle='--', alpha=0.5)

            # Drawdown calculation and plot
            color_dd = 'red'
            rolling_max_equity = equity_series.cummax()
            drawdown_series = (equity_series - rolling_max_equity) / (rolling_max_equity + 1e-8) * 100 # In percentage
            ax_drawdown.fill_between(drawdown_series.index, drawdown_series, 0, label='Drawdown (%)', color=color_dd, alpha=0.3)
            ax_drawdown.set_ylabel('Drawdown (%)', color=color_dd)
            ax_drawdown.tick_params(axis='y', labelcolor=color_dd)
            ax_drawdown.set_ylim(bottom=min(drawdown_series.min() - 5, -10), top=5) # Dynamic bottom limit

            # Combine legends
            lines, labels = ax_equity.get_legend_handles_labels()
            lines2, labels2 = ax_drawdown.get_legend_handles_labels()
            ax_drawdown.legend(lines + lines2, labels + labels2, loc='lower left', fontsize='small')
        else:
            ax_equity.text(0.5, 0.5, 'Equity Data Not Available', horizontalalignment='center', verticalalignment='center', transform=ax_equity.transAxes)
            ax_equity.set_yticks([])
            ax_drawdown.set_yticks([])


        # Formatting x-axis for the last subplot
        plt.xlabel('Date')
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout rect

        # Save the plot
        plot_filename = os.path.join(plot_dir, f"enhanced_hybrid_COMPREHENSIVE_{symbol.replace('/', '')}_{timeframe}_{start_date}_to_{end_date}.png")
        plt.savefig(plot_filename, dpi=150) # Increase DPI for better quality
        print(f"Comprehensive plot saved to {plot_filename}")
        # plt.show() # Comment out if running in non-interactive environment


