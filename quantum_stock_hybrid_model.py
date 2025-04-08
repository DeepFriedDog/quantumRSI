#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantum Harmonic-Wave Hybrid Model

This module implements a hybrid quantum model combining the best aspects of
the Quantum Stock Conservative model (high win rate) and the 
Quantum Stock Improved Aggressive model (high returns).

The hybrid approach leverages the mathematical frameworks from both:
1. Orrell's Quantum Oscillator Model with discrete energy levels
2. Zhang's Quantum Stock Model with wave function distribution

It combines conservative entry criteria with aggressive profit targets and
dynamically adjusts to market conditions.
"""

import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import norm
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class QuantumHybridParameters:
    """
    Parameter class for the Quantum Harmonic-Wave Hybrid model that combines 
    the conservative and aggressive approaches.
    """
    def __init__(self):
        # Core quantum parameters
        self.hbar = 1.0         # Reduced Planck constant (conservative approach)
        self.m = 1.0            # Mass parameter (will be adjusted by market cap)
        self.mass = 1.0         # Alternative name for mass parameter
        self.omega = 1.0e-4     # Angular frequency (oscillation frequency)
        self.lambda_param = 0.5 # Quantum coupling constant
        
        # Hybrid weight parameters - adjust for better balance
        self.conservative_weight = 0.55  # Weight for conservative signal (entry) - reduced from 0.65
        self.aggressive_weight = 0.45    # Weight for aggressive signal (exit) - increased from 0.35
        
        # Signal generation parameters
        self.entry_threshold = 0.6       # Threshold for entry signals - reduced from 0.65
        self.exit_threshold = 0.5        # Threshold for exit signals - increased from 0.45
        self.volatility_adaptation = True  # Adapt thresholds based on volatility
        self.momentum_factor = 0.6       # Weight for momentum component - increased from 0.5
        
        # Risk management parameters
        self.stop_loss_atr_factor = 1.4  # Stop loss as multiple of ATR - reduced from 1.5
        self.take_profit_atr_factor = 2.7  # Take profit as multiple of ATR - increased from 2.5
        self.atr_period = 14             # Period for ATR calculation
        self.rsi_period = 14             # Period for RSI calculation
        self.rsi_threshold_upper = 75    # Upper RSI threshold - increased from 70
        self.rsi_threshold_lower = 28    # Lower RSI threshold - decreased from 30
        self.rsi_weight = 0.25           # Weight for RSI in signal calculation - increased from 0.2
        
        # Market imbalance parameters
        self.volume_ema_period = 20      # Period for volume EMA
        self.imbalance_lookback = 6      # Lookback for market imbalance - increased from 5
        self.imbalance_threshold = 0.4   # Threshold for significant imbalance - decreased from 0.5
        
        # Advanced parameters
        self.dynamic_timeframe = True    # Adjust parameters based on timeframe
        self.risk_pct = 1.0              # Risk percentage per trade
        self.use_adaptive_position_sizing = True  # Use adaptive position sizing

def calculate_market_imbalance(df, params):
    """
    Calculate market imbalance based on volume and price direction.
    Implements Eq. 3 from the quantum harmonic oscillator paper.
    """
    # Calculate volume EMA
    df['volume_ema'] = df['volume'].ewm(span=params.volume_ema_period, adjust=False).mean()
    
    # Calculate price direction
    df['price_direction'] = np.sign(df['close'].diff(1)).fillna(0)
    
    # Calculate buying and selling volume
    df['buying_volume'] = df['volume'] * (df['price_direction'] > 0).astype(float)
    df['selling_volume'] = df['volume'] * (df['price_direction'] < 0).astype(float)
    
    # Calculate buying and selling pressure EMAs
    df['buying_pressure'] = df['buying_volume'].rolling(window=params.imbalance_lookback).sum() / \
                           df['volume'].rolling(window=params.imbalance_lookback).sum()
    df['selling_pressure'] = df['selling_volume'].rolling(window=params.imbalance_lookback).sum() / \
                            df['volume'].rolling(window=params.imbalance_lookback).sum()
                            
    # Calculate market imbalance
    df['market_imbalance'] = df['buying_pressure'] - df['selling_pressure']
    df['market_imbalance'].fillna(0, inplace=True)
    
    # Calculate normalized market imbalance
    df['market_imbalance_normalized'] = 2 * (df['market_imbalance'] - df['market_imbalance'].rolling(window=100).min()) / \
                                       (df['market_imbalance'].rolling(window=100).max() - df['market_imbalance'].rolling(window=100).min() + 1e-8) - 1
    df['market_imbalance_normalized'].fillna(0, inplace=True)
    df['market_imbalance_normalized'] = df['market_imbalance_normalized'].clip(-1, 1)
    
    return df

def calculate_volatility(df, params):
    """
    Calculate volatility using ATR and price return volatility.
    Implements Eq. 8 from the quantum harmonic oscillator paper.
    """
    # Calculate True Range
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=params.atr_period).mean()
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Calculate return volatility
    df['return_volatility'] = df['returns'].rolling(window=20).std()
    
    # Calculate normalized ATR (as a percentage of price)
    df['normalized_atr'] = df['atr'] / df['close']
    
    # Calculate hybrid volatility measure
    df['volatility'] = (df['normalized_atr'] + df['return_volatility']) / 2
    
    # Fill NaN values
    df['volatility'] = df['volatility'].fillna(df['volatility'].mean())
    
    return df

def calculate_energy_levels(df, params):
    """
    Calculate energy levels using quantum harmonic oscillator model.
    E_n = hbar * omega * (n + 1/2)
    """
    # Calculate price displacement
    df['price_displacement'] = df['close'] - df['close'].rolling(window=20).mean()
    df['price_displacement'] = df['price_displacement'] / df['close'].rolling(window=20).std()
    
    # Calculate kinetic energy component
    df['price_velocity'] = df['price_displacement'].diff(1)
    df['kinetic_energy'] = 0.5 * params.m * df['price_velocity']**2
    
    # Calculate potential energy component
    df['potential_energy'] = 0.5 * params.m * params.omega**2 * df['price_displacement']**2
    
    # Calculate total energy
    df['total_energy'] = df['kinetic_energy'] + df['potential_energy']
    
    # Calculate quantum energy levels
    df['quantum_level'] = np.round(df['total_energy'] / (params.hbar * params.omega) - 0.5)
    df['quantum_level'] = df['quantum_level'].clip(0, 10)  # Cap at reasonable values
    
    # Calculate normalized energy (between 0 and 1)
    max_levels = 10  # Maximum number of levels to consider
    df['normalized_energy'] = df['quantum_level'] / max_levels
    
    # Calculate energy transition
    df['energy_change'] = df['quantum_level'].diff(1)
    df['significant_energy_change'] = (abs(df['energy_change']) >= 1).astype(int)
    
    return df

def calculate_conservative_signals(df, params):
    """
    Calculate trading signals using the conservative approach, focusing on high win rate.
    """
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=params.rsi_period).mean()
    avg_loss = loss.rolling(window=params.rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate conservative signal component
    df['price_position'] = (df['close'] - df['close'].rolling(window=50).min()) / \
                          (df['close'].rolling(window=50).max() - df['close'].rolling(window=50).min() + 1e-8)
    
    # Conservative approach: more emphasis on energy levels and less on market imbalance
    df['conservative_signal_raw'] = (
        0.45 * (1 - df['normalized_energy']) +  # Lower energy levels are better for entry (increased from 0.4)
        0.35 * df['market_imbalance_normalized'] +  # Market imbalance (increased from 0.3)
        0.2 * (1 - df['price_position'])  # Lower price relative to range (reduced from 0.3)
    )
    
    # Apply RSI filter with more aggressive scaling
    rsi_factor = np.where(
        df['rsi'] < params.rsi_threshold_lower,
        1 + (params.rsi_threshold_lower - df['rsi']) / 50,  # More aggressive scaling (changed from 100)
        0.8  # Default factor when RSI is not oversold
    )
    
    df['conservative_signal_raw'] = df['conservative_signal_raw'] * rsi_factor
    
    # Normalize between 0 and 1
    df['conservative_signal'] = df['conservative_signal_raw'].rolling(window=20).apply(
        lambda x: (x[-1] - min(x)) / (max(x) - min(x) + 1e-8) if max(x) > min(x) else 0.5
    ).fillna(0.5)
    
    return df

def calculate_aggressive_signals(df, params):
    """
    Calculate trading signals using the aggressive approach, focusing on high returns.
    """
    # Calculate momentum indicators
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_diff'] = (df['ema_fast'] - df['ema_slow']) / df['close']
    
    # Add MACD for better trend detection
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_direction'] = np.sign(df['macd_hist'].diff(3))
    
    # Calculate aggressive signal component
    # Aggressive approach: more emphasis on market imbalance and energy transitions
    df['aggressive_signal_raw'] = (
        0.35 * df['market_imbalance_normalized'] +  # Market imbalance (increased from 0.3)
        0.25 * df['significant_energy_change'] +    # Energy transitions (quantum jumps) (decreased from 0.3)
        0.25 * df['ema_diff'] +                     # Momentum component (increased from 0.2)
        0.15 * df['macd_direction'] +               # MACD direction (new component)
        0.15 * (df['rsi'] - 50) / 50                # RSI component (decreased from 0.2)
    )
    
    # Apply volatility scaling for aggressive approach with improved scaling
    if params.volatility_adaptation:
        # Higher volatility means more opportunity for aggressive approach
        vol_ratio = df['volatility'] / df['volatility'].rolling(window=100).mean()
        # Cap the volatility scaling to prevent extreme values
        vol_scaling = vol_ratio.clip(0.5, 2.0)
        df['aggressive_signal_raw'] = df['aggressive_signal_raw'] * vol_scaling
    
    # Normalize between 0 and 1
    df['aggressive_signal'] = df['aggressive_signal_raw'].rolling(window=10).apply(
        lambda x: (x[-1] - min(x)) / (max(x) - min(x) + 1e-8) if max(x) > min(x) else 0.5
    ).fillna(0.5)
    
    return df

def find_trading_signals(df, params):
    """
    Find trading signals using the hybrid quantum model approach.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        params (QuantumHybridParameters): Model parameters
        
    Returns:
        pd.DataFrame: DataFrame with added signal columns
    """
    # Ensure dataframe has required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Calculate market imbalance
    df = calculate_market_imbalance(df, params)
    
    # Calculate volatility
    df = calculate_volatility(df, params)
    
    # Calculate energy levels
    df = calculate_energy_levels(df, params)
    
    # Calculate conservative and aggressive signals
    df = calculate_conservative_signals(df, params)
    df = calculate_aggressive_signals(df, params)
    
    # Calculate hybrid signal with dynamic weighting based on timeframe
    # Get timeframe information by looking at time difference between rows
    time_diff = pd.Series(df.index).diff().median().total_seconds() / 3600  # in hours
    
    # Adjust weights based on timeframe
    if time_diff >= 24:  # Daily timeframe
        conservative_weight = params.conservative_weight * 0.8  # Favor aggressive approach on daily
        aggressive_weight = 1 - conservative_weight
    elif time_diff >= 4:  # 4h timeframe
        conservative_weight = params.conservative_weight * 0.9  # Slightly favor aggressive on 4h
        aggressive_weight = 1 - conservative_weight
    else:  # 1h or smaller
        conservative_weight = params.conservative_weight * 1.1  # Favor conservative on smaller timeframes
        aggressive_weight = 1 - conservative_weight
    
    df['entry_signal'] = conservative_weight * df['conservative_signal'] + \
                         (1 - conservative_weight) * df['aggressive_signal']
    
    df['exit_signal'] = (1 - aggressive_weight) * df['conservative_signal'] + \
                        aggressive_weight * df['aggressive_signal']
    
    # Determine entry and exit thresholds with better scaling
    if params.volatility_adaptation:
        # Adjust thresholds based on volatility with improved scaling
        vol_ratio = df['volatility'] / df['volatility'].rolling(window=100).mean()
        # Cap the volatility ratio to prevent extreme values
        vol_ratio = vol_ratio.clip(0.7, 1.5)
        entry_threshold = params.entry_threshold / vol_ratio
        exit_threshold = params.exit_threshold * vol_ratio
    else:
        entry_threshold = params.entry_threshold
        exit_threshold = params.exit_threshold
    
    # Apply timeframe-specific threshold adjustments
    if time_diff >= 24:  # Daily timeframe
        entry_threshold = entry_threshold * 0.95  # Lower threshold for daily (more signals)
    elif time_diff >= 4:  # 4h timeframe
        entry_threshold = entry_threshold * 0.98  # Slightly lower for 4h
    else:  # 1h or smaller
        entry_threshold = entry_threshold * 1.05  # Higher threshold for smaller timeframes (less signals)
    
    # Initialize signal column
    df['signal'] = 0
    
    # Generate buy signals with improved filtering
    long_condition = (
        (df['entry_signal'] > entry_threshold) & 
        (df['market_imbalance_normalized'] > 0.1) &  # Require stronger imbalance
        (df['rsi'] < params.rsi_threshold_upper) &   # Not overbought
        (df['close'] > df['close'].rolling(10).mean())  # Price above short-term MA (trend filter)
    )
    
    # Generate sell signals with improved filtering
    short_condition = (
        (df['entry_signal'] > entry_threshold) &
        (df['market_imbalance_normalized'] < -0.1) &  # Require stronger imbalance
        (df['rsi'] > params.rsi_threshold_lower) &    # Not oversold
        (df['close'] < df['close'].rolling(10).mean())  # Price below short-term MA (trend filter)
    )
    
    # Apply signals
    df.loc[long_condition, 'signal'] = 1
    df.loc[short_condition, 'signal'] = -1
    
    # Calculate stop loss and take profit levels with adaptive sizing
    df['stop_loss'] = np.nan
    df['take_profit'] = np.nan
    
    # Dynamic ATR factors based on timeframe
    if time_diff >= 24:  # Daily timeframe
        sl_factor = params.stop_loss_atr_factor * 1.1  # Wider stops for daily
        tp_factor = params.take_profit_atr_factor * 1.2  # Wider targets for daily
    elif time_diff >= 4:  # 4h timeframe
        sl_factor = params.stop_loss_atr_factor * 1.05
        tp_factor = params.take_profit_atr_factor * 1.1
    else:  # 1h or smaller
        sl_factor = params.stop_loss_atr_factor
        tp_factor = params.take_profit_atr_factor
    
    # For long positions
    long_mask = df['signal'] == 1
    df.loc[long_mask, 'stop_loss'] = df.loc[long_mask, 'close'] * (1 - sl_factor * df.loc[long_mask, 'normalized_atr'])
    df.loc[long_mask, 'take_profit'] = df.loc[long_mask, 'close'] * (1 + tp_factor * df.loc[long_mask, 'normalized_atr'])
    
    # For short positions
    short_mask = df['signal'] == -1
    df.loc[short_mask, 'stop_loss'] = df.loc[short_mask, 'close'] * (1 + sl_factor * df.loc[short_mask, 'normalized_atr'])
    df.loc[short_mask, 'take_profit'] = df.loc[short_mask, 'close'] * (1 - tp_factor * df.loc[short_mask, 'normalized_atr'])
    
    # Fill forward stop loss and take profit for visualization
    df['stop_loss'] = df['stop_loss'].fillna(method='ffill')
    df['take_profit'] = df['take_profit'].fillna(method='ffill')
    
    return df 

def run_quantum_hybrid_strategy_backtest(symbol, timeframe, initial_capital, start_date, end_date, params=None):
    # ... (data fetching and indicator calculation) ...

    # --- Backtest Initialization ---
    capital = initial_capital
    position = 0
    equity_curve = [initial_capital]
    trades = []
    entry_price = 0
    entry_time = None
    risk_pct = getattr(params, 'risk_pct', 0.01)
    use_adaptive_sizing = getattr(params, 'use_adaptive_position_sizing', True)
    entry_commission = 0

    print("Starting backtest loop...")
    # --- Backtest Loop ---
    for i in range(1, len(df)):
        # <<< Early Stop Check >>>
        if capital <= 1.0:
            print(f"{df.index[i]} - Capital depleted (${capital:.2f}). Stopping backtest early.")
            equity_curve.extend([capital] * (len(df) - i))
            break

        current_price = df['close'].iloc[i]
        current_atr = df['atr'].iloc[i] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else df['atr'].mean()
        hybrid_signal = df['hybrid_signal'].iloc[i]
        # Assume exit signal is inverse of entry (or define explicitly if needed)
        exit_signal = 1.0 - hybrid_signal

        # --- Exit Logic ---
        # ... (rest of exit logic)

        # --- Entry Logic ---
        # ... (rest of entry logic)

        # Update Equity Curve
        if position != 0:
            current_value = capital + position * (current_price - entry_price)
            equity_curve.append(current_value)
        else:
            equity_curve.append(capital)

    # ... (final position close and performance calculation) ... 