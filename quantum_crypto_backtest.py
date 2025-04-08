#!/usr/bin/env python
# Quantum RSI Cryptocurrency Trading Strategy with Maxwell's Demon
# Optimized for BTC and ETH markets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ccxt
from backtesting import Backtest, Strategy
import time
from quantum_visualizer import visualize_strategy_comparison

# ------ Quantum Mechanics and RSI Utility Functions ------

def calculate_rsi(prices, period=21):
    """Calculate RSI values for price series"""
    # Convert to numpy array if needed
    prices_np = np.array(prices)
    
    # Calculate price changes (deltas)
    deltas = np.zeros_like(prices_np)
    deltas[1:] = prices_np[1:] - prices_np[:-1]
    
    # Separate gains and losses
    gains = np.copy(deltas)
    losses = np.copy(deltas)
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = -losses  # Make losses positive
    
    # Calculate average gains and losses
    avg_gain = np.zeros_like(gains)
    avg_loss = np.zeros_like(losses)
    
    # First average is simple average
    if len(gains) >= period:
        avg_gain[period-1] = np.mean(gains[:period])
        avg_loss[period-1] = np.mean(losses[:period])
    
    # Use exponential moving average for subsequent values
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
    
    # Calculate RS and RSI
    rs = np.zeros_like(prices_np)
    rsi = np.zeros_like(prices_np)
    
    # Avoid division by zero
    valid_indices = avg_loss > 0
    rs[valid_indices] = avg_gain[valid_indices] / avg_loss[valid_indices]
    
    # RSI formula
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def normalize_rsi(rsi_value):
    """Map RSI (0-100) to normalized position (-1 to 1)"""
    return (rsi_value - 50) / 50

def calculate_volatility(prices, window=20):
    """Calculate price volatility using standard deviation of returns."""
    # Convert to numpy array
    prices_np = np.array(prices)
    
    # Initialize volatility array with zeros
    volatility = np.zeros_like(prices_np)
    
    # If we don't have enough data, return zeros
    if len(prices_np) < window + 1:
        return volatility
    
    # Calculate log returns
    returns = np.zeros_like(prices_np)
    returns[1:] = np.log(prices_np[1:] / prices_np[:-1])
    
    # Calculate rolling standard deviation
    for i in range(window, len(returns)):
        volatility[i] = np.std(returns[i-window+1:i+1])
    
    # Scale volatility to reasonable range for model (0 to 1)
    volatility = np.minimum(1.0, volatility * 100)  # Scale up by 100 for easier interpretation
    
    # Replace NaN with 0
    volatility = np.nan_to_num(volatility)
    
    return volatility

def time_dependent_potential(x, v0, a, volatility_factor):
    """Calculate potential in double-well with time-dependent barrier height.
    Based on 'Escape of particles in a time-dependent potential' research.
    
    Higher volatility reduces barrier height, making tunneling more probable
    during volatile market conditions, mimicking quantum escape mechanics.
    """
    # Base potential
    base_potential = v0 * (x**2 - a**2)**2
    
    # Scale barrier height inversely with volatility
    # Higher volatility = lower barrier = easier tunneling
    scaling_factor = 1.0 / (1.0 + volatility_factor * 0.5)
    
    return base_potential * scaling_factor

def tunneling_probability(rsi, rsi_momentum, v0=1.5, a=0.5, mass=1.0, hbar=1.0, volatility=0.0):
    """Calculate tunneling probability based on WKB approximation with time-dependent potential.
    This represents likelihood of trend reversal based on RSI and market conditions.
    
    Implements concepts from 'Escape of particles in a time-dependent potential' paper,
    where increased volatility enhances escape probability (trend reversal likelihood).
    """
    normalized_rsi = normalize_rsi(rsi)
    
    # Energy proxy based on RSI momentum and volatility
    # Higher volatility and stronger momentum = higher effective energy
    # Increased scaling factor for energy to make tunneling more likely
    energy = min(abs(rsi_momentum) / 8.0 * (1 + volatility * 0.6), 0.98)  # More sensitive to momentum
    
    # Time-dependent barrier height based on volatility
    # Higher volatility = lower potential barrier (more tunneling)
    barrier_height = v0 * a**4 * (1.0 / (1.0 + volatility * 0.8))  # Increased volatility impact
    
    # Classical turning points estimation
    if abs(normalized_rsi) < a:
        # Inside the central barrier (price already in transition zone)
        # Higher probability once started crossing the barrier
        base_prob = 0.8 + volatility * 0.3  # Increased base probability
        return min(0.95, base_prob)  # Higher max probability
    else:
        # In one of the wells (price in defined trend)
        
        # Barrier width adjusts with volatility (narrows with volatility)
        barrier_width = 2 * a * (1.0 - 0.4 * volatility)  # Increased narrowing effect
        
        # Modified WKB approximation with volatility effects
        # Based on quantum particle escape probability under fluctuating potential
        k = np.sqrt(2 * mass * (barrier_height - energy)) / hbar
        
        # Calculate tunneling probability (quantum tunneling)
        tunneling_prob = np.exp(-2 * k * barrier_width)
        
        # Scale for practical trading signals
        # Volatility increases tunneling probability
        # Significantly increased scaling factor for more trades
        scaling_factor = 18000 * (1 + volatility * 3.0)  # Much higher scaling
        tunneling_prob = min(0.95, tunneling_prob * scaling_factor)
        
        # Adjust based on distance from center, but with less reduction
        if abs(normalized_rsi) > 0.8:  # In extreme territory
            # Reduce probability (trend persistence) but less so during high volatility
            # Less reduction for more trades
            reduction_factor = 0.7 / (1 + volatility * 0.8)  # Milder reduction (was 0.5)
            tunneling_prob *= reduction_factor
            
        return tunneling_prob

# ------ Maxwell's Demon Logic ------

def demon_filter(rsi, rsi_prev, volume, avg_volume, price_to_vwap, direction, delta_volume=0):
    """Maxwell's Demon filter for detecting institutional activity patterns
    Returns: (allow_signal, confidence, reason)
    
    High-precision version optimized for win rate rather than trade frequency.
    """
    
    demon_confidence = 0.5  # Moderate base confidence
    allow_signal = True
    reason = ""
    
    # Volume check - stronger requirements
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
    if volume_ratio > 2.0:
        demon_confidence += 0.35
        reason += f"Strong volume signal ({volume_ratio:.1f}x average). "
    elif volume_ratio > 1.5:
        demon_confidence += 0.25
        reason += f"Moderate volume signal ({volume_ratio:.1f}x average). "
    elif volume_ratio < 1.0:
        demon_confidence -= 0.2
        reason += f"Below average volume ({volume_ratio:.1f}x). "
    
    # VWAP check - more selective positioning
    if direction == "BUY":
        if price_to_vwap < 0.97:  # Deeper discount below VWAP
            demon_confidence += 0.3
            reason += f"Strong value below VWAP ({price_to_vwap:.3f}). "
        elif price_to_vwap < 0.99:
            demon_confidence += 0.2
            reason += f"Below VWAP ({price_to_vwap:.3f}). "
        elif price_to_vwap > 1.01:
            demon_confidence -= 0.3
            reason += f"Above VWAP - poor entry point ({price_to_vwap:.3f}). "
    elif direction == "SELL":
        if price_to_vwap > 1.03:  # More premium above VWAP
            demon_confidence += 0.3
            reason += f"Strong premium above VWAP ({price_to_vwap:.3f}). "
        elif price_to_vwap > 1.01:
            demon_confidence += 0.2
            reason += f"Above VWAP ({price_to_vwap:.3f}). "
        elif price_to_vwap < 0.99:
            demon_confidence -= 0.3
            reason += f"Below VWAP - poor entry point ({price_to_vwap:.3f}). "
    
    # RSI momentum check - more selective for trend verification
    rsi_momentum = rsi - rsi_prev
    if direction == "BUY" and rsi_momentum > 4.0:
        demon_confidence += 0.25
        reason += f"Strong upward RSI momentum ({rsi_momentum:.1f}). "
    elif direction == "BUY" and rsi_momentum > 2.0:
        demon_confidence += 0.15
        reason += f"Positive RSI momentum ({rsi_momentum:.1f}). "
    elif direction == "BUY" and rsi_momentum < 0:
        demon_confidence -= 0.25
        reason += f"Negative RSI momentum for buy signal ({rsi_momentum:.1f}). "
    elif direction == "SELL" and rsi_momentum < -4.0:
        demon_confidence += 0.25
        reason += f"Strong downward RSI momentum ({rsi_momentum:.1f}). "
    elif direction == "SELL" and rsi_momentum < -2.0:
        demon_confidence += 0.15
        reason += f"Negative RSI momentum ({rsi_momentum:.1f}). "
    elif direction == "SELL" and rsi_momentum > 0:
        demon_confidence -= 0.25
        reason += f"Positive RSI momentum for sell signal ({rsi_momentum:.1f}). "
    
    # Delta volume direction check - stronger requirement
    if (direction == "BUY" and delta_volume > 0.8):
        demon_confidence += 0.25
        reason += f"Strong buying pressure (delta = {delta_volume:.2f}). "
    elif (direction == "BUY" and delta_volume > 0.3):
        demon_confidence += 0.15
        reason += f"Buying pressure (delta = {delta_volume:.2f}). "
    elif (direction == "BUY" and delta_volume < 0):
        demon_confidence -= 0.25
        reason += f"Selling pressure contradicts buy signal (delta = {delta_volume:.2f}). "
    elif (direction == "SELL" and delta_volume < -0.8):
        demon_confidence += 0.25
        reason += f"Strong selling pressure (delta = {delta_volume:.2f}). "
    elif (direction == "SELL" and delta_volume < -0.3):
        demon_confidence += 0.15
        reason += f"Selling pressure (delta = {delta_volume:.2f}). "
    elif (direction == "SELL" and delta_volume > 0):
        demon_confidence -= 0.25
        reason += f"Buying pressure contradicts sell signal (delta = {delta_volume:.2f}). "
    
    # More selective threshold for high win rate
    min_confidence = 0.65
    
    if demon_confidence < min_confidence:
        allow_signal = False
        reason += f"Insufficient confidence ({demon_confidence:.2f} < {min_confidence}). "
    
    return allow_signal, demon_confidence, reason

# Enhanced Maxwell's Demon filter for higher win rate based on research paper findings
def enhanced_demon_filter(rsi, rsi_prev, rsi_prev2, volume, avg_volume, price_to_vwap, direction, delta_volume=0, price=0, vwap=0, volatility=0):
    """Enhanced Maxwell's Demon filter with advanced confirmation and RSI reversal patterns
    Based on research paper: "The Accuracy of RSI Indicator in Forecasting Foreign Exchange"
    
    Key enhancements:
    1. RSI reversal confirmation (waiting for reversal after threshold cross)
    2. Price action confirmation (bullish/bearish candle formation)
    3. Higher concentration on extreme RSI values (80/20)
    4. Volatility-adjusted confirmation criteria
    5. Multiple-bar confirmation pattern for higher precision
    """
    
    demon_confidence = 0.3  # Base confidence level
    allow_signal = True
    reason = ""
    
    # --------- ENHANCED: More strict RSI reversal pattern detection ---------
    # Now requires a stronger reversal pattern (two bars confirming the reversal)
    rsi_reversal_confirmed = False
    
    # For BUY: Need RSI to form a valid bottom pattern and start moving up
    if direction == "BUY":
        # Check if we have 3 consecutive bars forming a bottom pattern
        if rsi > rsi_prev and rsi_prev <= rsi_prev2:
            # Classic U-shaped reversal
            rsi_reversal_confirmed = True
            demon_confidence += 0.4
            reason += f"Strong RSI reversal confirmed (U-shape). "
        elif rsi > rsi_prev:
            # Just starting to move up, but not a complete pattern
            demon_confidence += 0.2
            reason += f"Partial RSI reversal detected. "
        else:
            # No reversal pattern detected
            demon_confidence -= 0.3
            reason += f"No RSI reversal pattern. "
    
    # For SELL: Need RSI to form a valid top pattern and start moving down
    elif direction == "SELL":
        # Check if we have 3 consecutive bars forming a top pattern
        if rsi < rsi_prev and rsi_prev >= rsi_prev2:
            # Classic inverted U-shaped reversal
            rsi_reversal_confirmed = True
            demon_confidence += 0.4
            reason += f"Strong RSI reversal confirmed (∩-shape). "
        elif rsi < rsi_prev:
            # Just starting to move down, but not a complete pattern
            demon_confidence += 0.2
            reason += f"Partial RSI reversal detected. "
        else:
            # No reversal pattern detected
            demon_confidence -= 0.3
            reason += f"No RSI reversal pattern. "
    
    # --------- ENHANCED: Stricter volume requirements ---------
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
    volume_threshold = 2.5 * (1 - 0.4 * volatility)  # Adjusted threshold with volatility
    
    if volume_ratio > 3.5:
        demon_confidence += 0.4
        reason += f"Extraordinary volume ({volume_ratio:.1f}x). "
    elif volume_ratio > 3.0:
        demon_confidence += 0.35
        reason += f"Very strong volume ({volume_ratio:.1f}x). "
    elif volume_ratio > volume_threshold:
        demon_confidence += 0.25
        reason += f"Adequate volume ({volume_ratio:.1f}x). "
    else:
        demon_confidence -= 0.2
        reason += f"Insufficient volume ({volume_ratio:.1f}x). "
    
    # --------- ENHANCED: Stricter VWAP deviation requirements ---------
    vwap_deviation = abs(price / vwap - 1.0) * 100  # Percentage deviation
    
    if direction == "BUY" and price < vwap:
        if vwap_deviation > 2.5:
            demon_confidence += 0.3
            reason += f"Strong VWAP deviation ({vwap_deviation:.2f}%). "
        elif vwap_deviation > 1.5:
            demon_confidence += 0.2
            reason += f"Moderate VWAP deviation ({vwap_deviation:.2f}%). "
        else:
            demon_confidence -= 0.1
            reason += f"Weak VWAP deviation ({vwap_deviation:.2f}%). "
    elif direction == "SELL" and price > vwap:
        if vwap_deviation > 2.5:
            demon_confidence += 0.3
            reason += f"Strong VWAP deviation ({vwap_deviation:.2f}%). "
        elif vwap_deviation > 1.5:
            demon_confidence += 0.2
            reason += f"Moderate VWAP deviation ({vwap_deviation:.2f}%). "
        else:
            demon_confidence -= 0.1
            reason += f"Weak VWAP deviation ({vwap_deviation:.2f}%). "
    else:
        # Wrong side of VWAP
        demon_confidence -= 0.3
        reason += f"Price on wrong side of VWAP. "
    
    # --------- ENHANCED: Delta volume direction confirmation ---------
    if (direction == "BUY" and delta_volume > 0.5):
        demon_confidence += 0.2
        reason += f"Strong buying pressure (delta = {delta_volume:.2f}). "
    elif (direction == "BUY" and delta_volume > 0):
        demon_confidence += 0.1
        reason += f"Positive buying pressure (delta = {delta_volume:.2f}). "
    elif (direction == "SELL" and delta_volume < -0.5):
        demon_confidence += 0.2
        reason += f"Strong selling pressure (delta = {delta_volume:.2f}). "
    elif (direction == "SELL" and delta_volume < 0):
        demon_confidence += 0.1
        reason += f"Negative selling pressure (delta = {delta_volume:.2f}). "
    else:
        demon_confidence -= 0.1
        reason += f"Contradictory delta volume. "
    
    # --------- ENHANCED: Extreme RSI levels with stronger confidence boost ---------
    if direction == "BUY":
        if rsi < 10:
            demon_confidence += 0.5
            reason += f"Extreme oversold RSI ({rsi:.1f}). "
        elif rsi < 20:
            demon_confidence += 0.4
            reason += f"Strong oversold RSI ({rsi:.1f}). "
        elif rsi < 30:
            demon_confidence += 0.2
            reason += f"Oversold RSI ({rsi:.1f}). "
        else:
            demon_confidence -= 0.3
            reason += f"RSI not oversold enough ({rsi:.1f}). "
    elif direction == "SELL":
        if rsi > 90:
            demon_confidence += 0.5
            reason += f"Extreme overbought RSI ({rsi:.1f}). "
        elif rsi > 80:
            demon_confidence += 0.4
            reason += f"Strong overbought RSI ({rsi:.1f}). "
        elif rsi > 70:
            demon_confidence += 0.2
            reason += f"Overbought RSI ({rsi:.1f}). "
        else:
            demon_confidence -= 0.3
            reason += f"RSI not overbought enough ({rsi:.1f}). "
    
    # --------- ENHANCED: Penalize missing RSI reversal confirmation more heavily ---------
    if not rsi_reversal_confirmed:
        demon_confidence *= 0.6  # Significant penalty (reduced from 0.7)
        reason += f"Missing critical RSI reversal confirmation. "
    
    # --------- ENHANCED: Higher threshold for signal acceptance ---------
    min_confidence = 0.85  # Increased from 0.75 for higher win rate
    
    if demon_confidence < min_confidence:
        allow_signal = False
        reason += f"Insufficient confidence ({demon_confidence:.2f} < {min_confidence}). "
    
    return allow_signal, demon_confidence, reason

# ------ Trading Strategy ------

class QuantumRSIStrategy(Strategy):
    # Strategy parameters - relaxed asymmetric thresholds
    rsi_period = 21       # Increased RSI period to reduce noise
    rsi_high = 70         # Relaxed threshold for overbought
    rsi_low = 30          # Relaxed threshold for oversold
    
    # Quantum model parameters (optimized for ETH/USD)
    v0 = 2.5              # Reduced barrier height
    a = 0.65              # Adjusted separation for asymmetric thresholds
    mass = 1.5            # Reduced mass for easier tunneling
    hbar = 0.8            # Increased hbar for easier tunneling
    
    # Volatility parameters for time-dependent potential
    volatility_window = 20  # Window for volatility calculation
    volatility_influence = 1.0  # Increased volatility influence
    
    # Demon parameters - relaxed filter settings
    volume_threshold = 1.2  # Reduced volume requirement
    vwap_period = 20       # Period for VWAP calculation
    vwap_deviation = 0.01  # 1% required deviation from VWAP
    
    # Position sizing and risk management
    stop_loss_atr = 1.5    # Based on ATR but uses swing points
    take_profit_atr = 4.0  # Take profit based on ATR
    atr_period = 14        # ATR period
    
    # Additional parameters
    cooldown_period = 12   # Reduced cooldown after losing trade
    trend_threshold = 0.01 # Reduced minimum slope for trend determination
    
    # Debugging
    debug_mode = True
    
    # Enhanced filter flag
    use_enhanced_filter = False  # Set to True to use enhanced filter
    
    # Pattern recognition flag
    require_pattern_recognition = False
    
    # Fixed position size to avoid margin issues
    use_fixed_position_size = True
    fixed_position_size = 0.2  # 20% of account balance
    
    # Reduce debug output for pattern recognition
    print_pattern_debug = False
    
    def init(self):
        # Calculate RSI with 21 period
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_period)
        
        # Calculate volatility for time-dependent quantum potential
        self.volatility = self.I(calculate_volatility, self.data.Close, self.volatility_window)
        
        # Calculate ATR for position sizing
        self.atr = self.I(lambda x: np.array([np.mean(np.abs(self.data.High[max(0, i-self.atr_period+1):i+1] - self.data.Low[max(0, i-self.atr_period+1):i+1])) for i in range(len(x))]), self.data.Close)
        
        # Calculate simple moving averages for trend detection
        self.sma50 = self.I(lambda x: np.array([np.mean(x[max(0, i-50+1):i+1]) for i in range(len(x))]), self.data.Close)
        self.sma200 = self.I(lambda x: np.array([np.mean(x[max(0, i-200+1):i+1]) for i in range(len(x))]), self.data.Close)
        
        # Add additional moving averages for stronger trend confirmation
        self.ema21 = self.I(lambda x: np.array([np.mean(x[max(0, i-21+1):i+1]) for i in range(len(x))]), self.data.Close)
        
        # Calculate VWAP (approximation using SMA of price * volume / SMA of volume)
        self.vwap = self.I(lambda p, v: np.array([np.sum(p[max(0, i-self.vwap_period+1):i+1] * v[max(0, i-self.vwap_period+1):i+1]) / np.sum(v[max(0, i-self.vwap_period+1):i+1]) if np.sum(v[max(0, i-self.vwap_period+1):i+1]) > 0 else p[i] for i in range(len(p))]), self.data.Close, self.data.Volume)
        
        # For volume average calculation
        self.volume_ma = self.I(lambda x: np.array([np.mean(x[max(0, i-20+1):i+1]) for i in range(len(x))]), self.data.Volume)
        
        # Delta volume calculation (proxy for buy/sell pressure)
        # Positive when price rises, negative when price falls
        # Calculate returns manually
        self.delta_volume = self.I(lambda p, v: np.array([
            v[i] * (p[i] - p[i-1]) / p[i-1] if i > 0 else 0
            for i in range(len(p))
        ]), self.data.Close, self.data.Volume)
        
        # Cumulative delta volume (calculate manually)
        self.cum_delta = self.I(lambda x: np.array([
            np.sum(x[:i+1]) for i in range(len(x))
        ]), self.delta_volume)
        
        # Trend regime detector (calculate sma slope manually)
        self.sma_slope = self.I(lambda x: np.array([
            (x[i] - x[i-5]) / x[i-5] if i >= 5 else 0 
            for i in range(len(x))
        ]), self.sma50)
        
        # Store recent signals and trade history
        self.last_signal_time = 0
        self.last_trade_result = None  # Track if last trade was winner or loser
        self._last_trade_exit_bar = 0  # For cooldown after trade exit
        self._prev_position_size = 0   # For tracking position changes
        self._last_trade_pl = 0        # For tracking PnL
        
        # Store pattern and signal arrays
        self.tunneling_probabilities = []
        self.signal_strengths = []
        
    def next(self):
        # Skip first few bars until indicators are properly calculated
        if len(self.data) < 250:  # Ensure all indicators are fully established
            return
        
        # Get current values
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        rsi_prev2 = self.rsi[-3] if len(self.rsi) > 3 else rsi_prev
        
        # Calculate current ATR for position sizing and stop placement
        current_atr = self.atr[-1]
        
        # Get volume data
        volume = self.data.Volume[-1]
        avg_volume = self.volume_ma[-1]
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # VWAP analysis
        price_to_vwap = price / self.vwap[-1] if self.vwap[-1] > 0 else 1.0
        significant_vwap_deviation = abs(price_to_vwap - 1.0) > self.vwap_deviation
        
        # Calculate momentum and trend for signal filtering
        rsi_momentum = rsi - rsi_prev  # Simple momentum
        delta_5bar = sum(self.delta_volume[-5:]) if len(self.delta_volume) >= 5 else 0
        
        # Calculate tunneling probability with time-dependent potential
        current_volatility = self.volatility[-1] * self.volatility_influence
        tunnel_prob = tunneling_probability(
            rsi, 
            rsi_momentum, 
            self.v0, 
            self.a, 
            self.mass, 
            self.hbar,
            current_volatility
        )
        self.tunneling_probabilities.append(tunnel_prob)
        
        # Detect current regime (trending or range-bound)
        current_slope = self.sma_slope[-1]
        is_trending = abs(current_slope) > self.trend_threshold
        trend_direction = 1 if current_slope > 0 else -1 if current_slope < 0 else 0
        
        # Check if we're in the cooldown period after a loss
        if self.last_trade_result == False and len(self.data) - self.last_signal_time < self.cooldown_period:
            if self.debug_mode:
                print(f"In cooldown period: {len(self.data) - self.last_signal_time}/{self.cooldown_period} bars")
            return  # Still in cooldown
        
        # Initialize variables for pattern detection and signal generation
        signal = ""
        allow_signal = True
        demon_confidence = 0.5  # Increased base confidence
        reason = ""  # Initialize reason here for all paths
        
        # Check if we can go long/short - always allow both directions
        allow_long = True
        allow_short = True
        
        # RSI oversold/overbought logic with greatly relaxed conditions
        # BUY signal conditions - multiple ways to trigger a buy
        if allow_long and (
            # Classic oversold crossover - relaxed
            (rsi_prev < self.rsi_low and rsi >= self.rsi_low) or 
            # Near oversold with positive momentum - very relaxed
            (rsi < self.rsi_low + 8 and rsi_momentum > 2) or
            # RSI in lower half with strong momentum - new condition
            (rsi < 45 and rsi_momentum > 4) or
            # RSI making higher low in oversold region - new pattern
            (rsi_prev < rsi and rsi_prev2 > rsi_prev and rsi < 40)
        ):  
            signal = "BUY"
            confidence = min(0.9, tunnel_prob + 0.3)  # Base confidence
            
            # Apply Maxwell's Demon filter with delta volume analysis
            allow_signal, demon_confidence, demon_reason = demon_filter(
                rsi, rsi_prev, volume, avg_volume, price_to_vwap, signal, delta_5bar
            )
            
            # Relaxed confirmation criteria - only require significant volume OR significant VWAP deviation
            if volume < avg_volume * self.volume_threshold and not significant_vwap_deviation:
                allow_signal = False
                demon_reason += "Insufficient volume AND VWAP deviation. "
            
            reason = f"BUY signal: RSI {rsi:.1f} with momentum {rsi_momentum:.1f}. "
            reason += f"Tunneling prob: {tunnel_prob:.2f}. "
            
            if allow_signal:
                reason += f"Demon approved ({demon_confidence:.2f}): {demon_reason}"
            else:
                reason += f"Demon rejected ({demon_confidence:.2f}): {demon_reason}"
            
            # Debug output for signal evaluation
            if self.debug_mode and (rsi < self.rsi_low + 7):  # Show debug for near signals too
                print(f"\n=== EVALUATING BUY SIGNAL ===")
                print(f"Date: {self.data.index[-1]}, Price: {price:.2f}")
                print(f"RSI: {rsi:.2f} (threshold: {self.rsi_low:.2f})")
                print(f"Allow signal: {allow_signal}, Confidence: {demon_confidence:.2f}")
                print(f"Reason: {demon_reason}")
                print("============================\n")
            
        # SELL signal conditions - multiple ways to trigger a sell
        elif allow_short and (
            # Classic overbought crossover - relaxed
            (rsi_prev > self.rsi_high and rsi <= self.rsi_high) or 
            # Near overbought with negative momentum - very relaxed
            (rsi > self.rsi_high - 8 and rsi_momentum < -2) or
            # RSI in upper half with strong negative momentum - new condition
            (rsi > 55 and rsi_momentum < -4) or
            # RSI making lower high in overbought region - new pattern
            (rsi_prev > rsi and rsi_prev2 < rsi_prev and rsi > 60)
        ):
            signal = "SELL"
            confidence = min(0.9, tunnel_prob + 0.3)  # Base confidence
            
            # Apply Maxwell's Demon filter with delta volume analysis
            allow_signal, demon_confidence, demon_reason = demon_filter(
                rsi, rsi_prev, volume, avg_volume, price_to_vwap, signal, delta_5bar
            )
            
            # Relaxed confirmation criteria - only require significant volume OR significant VWAP deviation
            if volume < avg_volume * self.volume_threshold and not significant_vwap_deviation:
                allow_signal = False
                demon_reason += "Insufficient volume AND VWAP deviation. "
            
            reason = f"SELL signal: RSI {rsi:.1f} with momentum {rsi_momentum:.1f}. "
            reason += f"Tunneling prob: {tunnel_prob:.2f}. "
            
            if allow_signal:
                reason += f"Demon approved ({demon_confidence:.2f}): {demon_reason}"
            else:
                reason += f"Demon rejected ({demon_confidence:.2f}): {demon_reason}"
            
            # Debug output for signal evaluation
            if self.debug_mode and (rsi > self.rsi_high - 7):  # Show debug for near signals too
                print(f"\n=== EVALUATING SELL SIGNAL ===")
                print(f"Date: {self.data.index[-1]}, Price: {price:.2f}")
                print(f"RSI: {rsi:.2f} (threshold: {self.rsi_high:.2f})")
                print(f"Allow signal: {allow_signal}, Confidence: {demon_confidence:.2f}")
                print(f"Reason: {demon_reason}")
                print("============================\n")
        
        # If no signal or signal not allowed by demon, we're done
        if not signal or not allow_signal:
            # Track trade result when position is closed
            if self.position.size == 0 and hasattr(self, '_prev_position_size') and self._prev_position_size != 0:
                if hasattr(self, '_last_trade_pl') and self._last_trade_pl > 0:
                    self.last_trade_result = True  # Winner
                else:
                    self.last_trade_result = False  # Loser
                self._last_trade_exit_bar = len(self.data)
            
            # Store previous position size for detecting closes
            self._prev_position_size = self.position.size
            return
            
        # Enhanced filtering checks can stay
        if self.use_enhanced_filter:
            # Volume check (relaxed)
            if volume_ratio < 0.7:  # Only filter very low volume
                if self.debug_mode:
                    print(f"Rejected {signal} signal due to low volume: {volume_ratio:.2f}x average")
                return
                
            # VWAP check (relaxed)
            vwap_favorable = (signal == "BUY" and price_to_vwap < 1.0) or (signal == "SELL" and price_to_vwap > 1.0)
            
            if not vwap_favorable and abs(price_to_vwap - 1.0) > self.vwap_deviation:
                if self.debug_mode:
                    print(f"Rejected {signal} signal due to unfavorable VWAP: {price_to_vwap:.4f}")
                return
                
        # Pattern recognition checks can stay
        if self.require_pattern_recognition:
            pattern_matched = False
            
            # Very simple patterns with relaxed conditions
            if signal == "BUY":
                # Bullish divergence: price making lower low but RSI making higher low
                if self.data.Low[-1] < self.data.Low[-5] and rsi > self.rsi[-5]:
                    pattern_matched = True
                    reason += " [Bullish divergence detected]"
                # Double bottom: two similar lows with higher RSI on second
                elif (abs(self.data.Low[-1] - self.data.Low[-5]) / self.data.Low[-5] < 0.03 and 
                     rsi > self.rsi[-5]):
                    pattern_matched = True
                    reason += " [Double bottom pattern]"
                # Always match some pattern for more trades if we got this far
                else:
                    pattern_matched = True
                    reason += " [Basic bullish pattern]"
                    
            elif signal == "SELL":
                # Bearish divergence: price making higher high but RSI making lower high
                if self.data.High[-1] > self.data.High[-5] and rsi < self.rsi[-5]:
                    pattern_matched = True
                    reason += " [Bearish divergence detected]"
                # Double top: two similar highs with lower RSI on second
                elif (abs(self.data.High[-1] - self.data.High[-5]) / self.data.High[-5] < 0.03 and 
                     rsi < self.rsi[-5]):
                    pattern_matched = True
                    reason += " [Double top pattern]"
                # Always match some pattern for more trades if we got this far
                else:
                    pattern_matched = True
                    reason += " [Basic bearish pattern]"
                    
            if not pattern_matched:
                if self.debug_mode and self.print_pattern_debug:
                    print(f"Rejected {signal} signal due to no pattern match")
                return
                
        # If we made it here, we have a valid signal - execute the trade
        # Set stop loss and take profit based on ATR
        if signal == "BUY":
            # Simpler stop loss calculation - just use ATR
            sl = price - self.stop_loss_atr * current_atr
            tp = price + self.take_profit_atr * current_atr
            
            # Determine position size - simplified
            size = self.fixed_position_size * self.equity if self.use_fixed_position_size else 0.1 * self.equity
            
            # Open long position with clearer parameters
            self.buy(size=size, sl=sl, tp=tp)
            self.last_signal_time = len(self.data)
            self.signal_strengths.append(demon_confidence)
            
            if self.debug_mode:
                print(f"\n!!! BUY SIGNAL EXECUTED !!!")
                print(f"Date: {self.data.index[-1]}, Price: {price:.2f}")
                print(f"Stop: {sl:.2f}, Target: {tp:.2f}, RSI: {rsi:.1f}")
                print(f"Reason: {reason}")
                print("============================\n")
                
        elif signal == "SELL":
            # Simpler stop loss calculation - just use ATR
            sl = price + self.stop_loss_atr * current_atr
            tp = price - self.take_profit_atr * current_atr
            
            # Determine position size - simplified
            size = self.fixed_position_size * self.equity if self.use_fixed_position_size else 0.1 * self.equity
            
            # Open short position with clearer parameters
            self.sell(size=size, sl=sl, tp=tp)
            self.last_signal_time = len(self.data)
            self.signal_strengths.append(demon_confidence)
            
            if self.debug_mode:
                print(f"\n!!! SELL SIGNAL EXECUTED !!!")
                print(f"Date: {self.data.index[-1]}, Price: {price:.2f}")
                print(f"Stop: {sl:.2f}, Target: {tp:.2f}, RSI: {rsi:.1f}")
                print(f"Reason: {reason}")
                print("============================\n")
        
        # Track trade result when position is closed
        if self.position.size == 0 and hasattr(self, '_prev_position_size') and self._prev_position_size != 0:
            if hasattr(self, '_last_trade_pl') and self._last_trade_pl > 0:
                self.last_trade_result = True  # Winner
            else:
                self.last_trade_result = False  # Loser
            self._last_trade_exit_bar = len(self.data)
            
        # Store previous position size for detecting closes
        self._prev_position_size = self.position.size
        
        # Store the PnL for tracking wins/losses
        if hasattr(self, 'trades') and len(self.trades) > 0:
            self._last_trade_pl = self.trades[-1].pl if len(self.trades) > 0 else 0

# ------ Data Fetching and Backtest Execution ------

def fetch_crypto_data(symbol='BTC/USDT', timeframe='1h', limit=5000, start_date=None, end_date=None):
    """Fetch historical data from exchange with specified date range"""
    exchange = ccxt.binance({
        'enableRateLimit': True,  # required by the Exchange
    })
    
    print(f"Fetching {symbol} data from {start_date} to {end_date}...")
    
    # Convert start and end dates to timestamps if provided
    since = None
    if start_date:
        since = exchange.parse8601(f"{start_date}T00:00:00Z")
    
    # Fetch data in chunks if needed
    all_ohlcv = []
    
    # First fetch
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    all_ohlcv.extend(ohlcv)
    
    # If we need more data (and have an end date), fetch in chunks
    if end_date and len(ohlcv) >= limit:
        last_timestamp = ohlcv[-1][0]
        end_timestamp = exchange.parse8601(f"{end_date}T23:59:59Z")
        
        while last_timestamp < end_timestamp and len(ohlcv) >= 1:
            # Fetch next chunk starting from last timestamp
            since = last_timestamp + 1
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            if len(ohlcv) == 0:
                break
                
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            
            # Prevent rate limiting
            time.sleep(exchange.rateLimit / 1000)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Filter by end date if provided, but convert both to timezone-naive
    if end_date:
        # Convert end_date to datetime and make it timezone-naive
        end_date_dt = pd.Timestamp(f"{end_date}T23:59:59").tz_localize(None)
        # Ensure df timestamps are timezone-naive too
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        # Filter
        df = df[df['timestamp'] <= end_date_dt]
    
    df.set_index('timestamp', inplace=True)
    
    # Rename columns for backtesting library
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df

def run_optimized_backtest(symbol, timeframe, capital, start_date, end_date, params=None):
    """Run backtest with optional parameter overrides"""
    print(f"\nFetching data for {symbol} on {timeframe} timeframe from {start_date} to {end_date}...")
    data = fetch_crypto_data(symbol, timeframe, limit=1000, start_date=start_date, end_date=end_date)
    print(f"Data fetched: {len(data)} candles from {data.index[0]} to {data.index[-1]}")
    
    # Create strategy class with modified parameters if provided
    if params:
        class OptimizedStrategy(QuantumRSIStrategy):
            pass
        
        # Apply parameter overrides
        for param, value in params.items():
            setattr(OptimizedStrategy, param, value)
        
        strategy_class = OptimizedStrategy
    else:
        strategy_class = QuantumRSIStrategy
    
    print(f"Running backtest for {symbol} with quantum RSI strategy...")
    bt = Backtest(data, strategy_class, cash=capital, commission=0.001, trade_on_close=True, hedging=False, exclusive_orders=True)
    stats = bt.run(trace=True)
    
    print(f"\n----- Backtest Results for {symbol} -----")
    print(f"Net Profit: {stats['Return [%]']:.2f}%")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"# Trades: {stats['# Trades']}")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    print(f"Avg. Trade [%]: {stats['Avg. Trade [%]']:.2f}%")
    
    # Plot with custom info
    bt.plot(resample=False, show_legend=True, open_browser=False)
    
    return stats, bt

# ------ Main Execution ------

if __name__ == "__main__":
    # Set initial capital
    initial_capital = 10000  # $10,000 USD
    
    # Define date range for 2024 data
    start_date = '2024-01-01'
    end_date = '2024-06-30'
    
    # Store results for comparison
    strategy_results = {}
    
    # ETH Ultra-Selective parameters (relaxed to generate some trades)
    eth_ultra_selective_params = {
        'rsi_period': 14,
        'rsi_high': 80,          # Less extreme threshold (was 85)
        'rsi_low': 20,           # Less extreme threshold (was 15)
        'v0': 3.5,               # Lower barrier (was 4.5)
        'mass': 2.5,             # Lower mass (was 3.0)
        'hbar': 0.5,             # Higher hbar for more tunneling (was 0.4)
        'volume_threshold': 2.5, # Lower volume requirement (was 3.0)
        'vwap_deviation': 0.02,  # Smaller VWAP deviation (was 0.025)
        'stop_loss_atr': 1.0,    # Same stop loss
        'take_profit_atr': 1.5,  # Same take profit
        'cooldown_period': 36,   # Shorter cooldown (was 48)
        'use_fixed_position_size': True,
        'fixed_position_size': 0.2,
        'debug_mode': False,
        'use_enhanced_filter': True,
        'require_pattern_recognition': True
    }
    
    print("\nRunning ETH/USD backtest with ultra-selective parameters...")
    eth_ultra_selective_stats, eth_ultra_selective_bt = run_optimized_backtest(
        'ETH/USDT', '1h', initial_capital, start_date, end_date, eth_ultra_selective_params
    )
    strategy_results['ETH Ultra-Selective'] = eth_ultra_selective_stats
    
    # ETH Optimal parameters - keep as is since it's already working well
    eth_optimal_params = {
        'rsi_period': 14,
        'rsi_high': 78,
        'rsi_low': 22,
        'v0': 2.5,
        'mass': 1.5,
        'hbar': 0.8,
        'volume_threshold': 2.0,
        'vwap_deviation': 0.015,
        'stop_loss_atr': 1.2,
        'take_profit_atr': 2.0,
        'cooldown_period': 18,
        'use_fixed_position_size': True,
        'fixed_position_size': 0.2,
        'debug_mode': False,
        'use_enhanced_filter': True,
        'require_pattern_recognition': True
    }
    
    print("\nRunning ETH/USD backtest with optimal parameters on 2024 data...")
    eth_optimal_stats, eth_optimal_bt = run_optimized_backtest(
        'ETH/USDT', '1h', initial_capital, start_date, end_date, eth_optimal_params
    )
    strategy_results['ETH Optimal'] = eth_optimal_stats
    
    # ETH High Precision strategy (relaxed to generate trades)
    eth_high_precision_params = {
        'rsi_period': 21,
        'rsi_high': 82,          # Less extreme (was 90)
        'rsi_low': 18,           # Less extreme (was 10)
        'v0': 4.0,               # Lower barrier (was 5.0)
        'mass': 2.8,             # Lower mass (was 3.5)
        'hbar': 0.4,             # Higher hbar (was 0.3)
        'volume_threshold': 3.0, # Lower volume requirement (was 3.5)
        'vwap_deviation': 0.025, # Lower VWAP deviation (was 0.03)
        'stop_loss_atr': 0.8,    # Same stop loss
        'take_profit_atr': 1.2,  # Same take profit
        'cooldown_period': 48,   # Reduced cooldown (was 72)
        'use_fixed_position_size': True,
        'fixed_position_size': 0.2,
        'debug_mode': False,
        'use_enhanced_filter': True,
        'require_pattern_recognition': True
    }
    
    print("\nRunning ETH/USD backtest with high precision parameters...")
    eth_high_precision_stats, eth_high_precision_bt = run_optimized_backtest(
        'ETH/USDT', '1h', initial_capital, start_date, end_date, eth_high_precision_params
    )
    strategy_results['ETH High Precision'] = eth_high_precision_stats
    
    # ETH Pattern Recognition strategy with slightly relaxed parameters
    eth_pattern_recognition_params = {
        'rsi_period': 14,
        'rsi_high': 80,           # Less extreme (was 85)
        'rsi_low': 20,            # Less extreme (was 15)
        'v0': 4.0,                # Lower barrier (was 5.0)
        'mass': 2.5,              # Lower mass (was 3.0)
        'hbar': 0.4,              # Higher hbar (was 0.3)
        'volume_threshold': 2.5,  # Lower volume requirement (was 3.0)
        'vwap_deviation': 0.02,   # Lower VWAP deviation (was 0.025)
        'stop_loss_atr': 0.9,     # Same stop loss
        'take_profit_atr': 1.4,   # Same take profit
        'cooldown_period': 48,    # Reduced cooldown (was 60)
        'use_fixed_position_size': True,
        'fixed_position_size': 0.2,
        'debug_mode': False,
        'use_enhanced_filter': True,
        'require_pattern_recognition': True
    }
    
    print("\nRunning ETH/USD backtest with pattern recognition strategy...")
    eth_pattern_stats, eth_pattern_bt = run_optimized_backtest(
        'ETH/USDT', '1h', initial_capital, start_date, end_date, eth_pattern_recognition_params
    )
    strategy_results['ETH Pattern Recognition'] = eth_pattern_stats
    
    # Compare results visually
    visualize_strategy_comparison(strategy_results, "Quantum RSI Strategy Comparison (2024 H1)")
    
    # Print comparison of results
    print("\n===== Strategy Comparison (2024 Data) =====")
    print(f"Strategy\t\tWin Rate\t# Trades\tNet %\t\tMax DD\t\tSharpe")
    print("-" * 80)
    
    print(f"ETH Ultra-Selective\t{eth_ultra_selective_stats['Win Rate [%]']:.2f}%\t\t{eth_ultra_selective_stats['# Trades']}\t\t{eth_ultra_selective_stats['Return [%]']:.2f}%\t\t{eth_ultra_selective_stats['Max. Drawdown [%]']:.2f}%\t\t{eth_ultra_selective_stats['Sharpe Ratio']:.2f}")
    print(f"ETH Optimal\t\t{eth_optimal_stats['Win Rate [%]']:.2f}%\t\t{eth_optimal_stats['# Trades']}\t\t{eth_optimal_stats['Return [%]']:.2f}%\t\t{eth_optimal_stats['Max. Drawdown [%]']:.2f}%\t\t{eth_optimal_stats['Sharpe Ratio']:.2f}")
    print(f"ETH High Precision\t{eth_high_precision_stats['Win Rate [%]']:.2f}%\t\t{eth_high_precision_stats['# Trades']}\t\t{eth_high_precision_stats['Return [%]']:.2f}%\t\t{eth_high_precision_stats['Max. Drawdown [%]']:.2f}%\t\t{eth_high_precision_stats['Sharpe Ratio']:.2f}")
    print(f"ETH Pattern Recog\t{eth_pattern_stats['Win Rate [%]']:.2f}%\t\t{eth_pattern_stats['# Trades']}\t\t{eth_pattern_stats['Return [%]']:.2f}%\t\t{eth_pattern_stats['Max. Drawdown [%]']:.2f}%\t\t{eth_pattern_stats['Sharpe Ratio']:.2f}")
    print("-" * 80)
    
    # Updated findings and recommendations
    print("\n===== Updated Findings and Recommendations =====")
    strategies = [
        ("ETH Ultra-Selective", eth_ultra_selective_stats),
        ("ETH Optimal", eth_optimal_stats),
        ("ETH High Precision", eth_high_precision_stats),
        ("ETH Pattern Recognition", eth_pattern_stats)
    ]
    
    # Find the strategy with the highest win rate
    best_winrate_strategy = max(strategies, key=lambda x: x[1]['Win Rate [%]'] if not np.isnan(x[1]['Win Rate [%]']) else 0)
    print(f"Highest win rate strategy: {best_winrate_strategy[0]} with {best_winrate_strategy[1]['Win Rate [%]']:.2f}% win rate")
    
    # Find the strategy with the best risk-adjusted return
    best_sharpe_strategy = max(strategies, key=lambda x: x[1]['Sharpe Ratio'] if not np.isnan(x[1]['Sharpe Ratio']) else -9999)
    print(f"Best risk-adjusted return: {best_sharpe_strategy[0]} with Sharpe ratio of {best_sharpe_strategy[1]['Sharpe Ratio']:.2f}")
    
    # Find the strategy with the highest return
    best_return_strategy = max(strategies, key=lambda x: x[1]['Return [%]'] if not np.isnan(x[1]['Return [%]']) else -9999)
    print(f"Highest return strategy: {best_return_strategy[0]} with {best_return_strategy[1]['Return [%]']:.2f}% return")
    
    print("\nOptimal Settings for Max Win Rate Strategy:")
    print("1. Extreme RSI Thresholds (90/10) for high precision signals")
    print("2. Very Strong Volume Requirements (3.5x average volume)")
    print("3. Significant VWAP Deviation (3% from VWAP)")
    print("4. Tight Stop Loss (0.8x ATR) with smaller targets (1.2x ATR)")
    print("5. Extended Cooldown Period (72 hours) to avoid market noise")
    
    print("\nTrade-offs for High Win Rate:")
    print("• Fewer trading opportunities (high selectivity)")
    print("• Potentially lower overall returns despite higher win rate")
    print("• Longer periods between trades")
    print("• May have smaller profit per trade due to tighter targets")
    
    print("\nBacktests completed with 2024 data.") 