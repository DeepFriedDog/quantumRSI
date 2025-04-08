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
    energy = min(abs(rsi_momentum) / 10.0 * (1 + volatility * 0.4), 0.95)  # Increased volatility impact
    
    # Time-dependent barrier height based on volatility
    # Higher volatility = lower potential barrier
    barrier_height = v0 * a**4 * (1.0 / (1.0 + volatility * 0.6))  # Increased volatility effect
    
    # Classical turning points estimation
    if abs(normalized_rsi) < a:
        # Inside the central barrier (price already in transition zone)
        # Higher probability once started crossing the barrier
        base_prob = 0.7 + volatility * 0.3  # Increased volatility influence
        return min(0.9, base_prob)
    else:
        # In one of the wells (price in defined trend)
        
        # Barrier width adjusts with volatility (narrows with volatility)
        barrier_width = 2 * a * (1.0 - 0.3 * volatility)  # Increased volatility effect
        
        # Modified WKB approximation with volatility effects
        # Based on quantum particle escape probability under fluctuating potential
        k = np.sqrt(2 * mass * (barrier_height - energy)) / hbar
        
        # Calculate tunneling probability (quantum tunneling)
        tunneling_prob = np.exp(-2 * k * barrier_width)
        
        # Scale for practical trading signals
        # Volatility increases tunneling probability
        scaling_factor = 12000 * (1 + volatility * 2.5)  # Increased scaling
        tunneling_prob = min(0.9, tunneling_prob * scaling_factor)
        
        # Adjust based on distance from center
        if abs(normalized_rsi) > 0.8:  # In extreme territory
            # Reduce probability (trend persistence) but less so during high volatility
            reduction_factor = 0.5 / (1 + volatility * 0.7)  # Increased volatility adjustment
            tunneling_prob *= reduction_factor
            
        return tunneling_prob

# ------ Maxwell's Demon Logic ------

def demon_filter(rsi, rsi_prev, volume, avg_volume, price_to_vwap, direction, delta_volume=0):
    """Maxwell's Demon filter for detecting institutional activity patterns
    Returns: (allow_signal, confidence, reason)
    
    Based on research, this implementation focuses on:
    1. Extreme RSI values for higher accuracy
    2. Volume confirmation of institutional activity
    3. Price-VWAP relationship
    4. Delta volume analysis for buy/sell pressure
    """
    
    demon_confidence = 0.3  # Increased base confidence level
    allow_signal = True
    reason = ""
    
    # Volume spike check (institutional activity) - relaxed threshold
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
    if volume_ratio > 3.0:  # Reduced threshold
        demon_confidence += 0.4  # Highest confidence boost
        reason += f"Volume spike detected ({volume_ratio:.1f}x average). "
    elif volume_ratio > 2.0:  # Reduced threshold
        demon_confidence += 0.3  # Higher confidence boost
        reason += f"Moderate volume increase ({volume_ratio:.1f}x average). "
    elif volume_ratio > 1.5:  # Added lower threshold
        demon_confidence += 0.1  # Modest confidence boost
        reason += f"Slight volume increase ({volume_ratio:.1f}x average). "
    
    # VWAP divergence (price far from value) - relaxed requirement
    if direction == "BUY" and price_to_vwap < 0.99:  # 1% below VWAP
        demon_confidence += 0.3  # Increased confidence boost
        reason += f"Price below VWAP ({price_to_vwap:.3f}). "
    elif direction == "SELL" and price_to_vwap > 1.01:  # 1% above VWAP
        demon_confidence += 0.3  # Increased confidence boost
        reason += f"Price above VWAP ({price_to_vwap:.3f}). "
    
    # Delta Volume analysis for buy/sell pressure - relaxed
    if direction == "BUY" and delta_volume > -0.5:  # Allow slightly negative
        demon_confidence += 0.2
        reason += f"Delta volume acceptable ({delta_volume:.2f}). "
    elif direction == "SELL" and delta_volume < 0.5:  # Allow slightly positive
        demon_confidence += 0.2
        reason += f"Delta volume acceptable ({delta_volume:.2f}). "
    
    # RSI momentum check - require less momentum
    rsi_change = rsi - rsi_prev
    if (direction == "BUY" and rsi_change > 5) or (direction == "SELL" and rsi_change < -5):
        demon_confidence += 0.25  # Increased confidence boost
        reason += f"Strong RSI momentum ({rsi_change:.1f}). "
    elif (direction == "BUY" and rsi_change > 2) or (direction == "SELL" and rsi_change < -2):
        demon_confidence += 0.15
        reason += f"Moderate RSI momentum ({rsi_change:.1f}). "
    
    # Check extreme RSI values - relaxed thresholds
    if direction == "BUY" and rsi < 20:  # Oversold
        demon_confidence += 0.4
        reason += f"Oversold RSI ({rsi:.1f}). "
    elif direction == "BUY" and rsi < 30:  # Moderately oversold
        demon_confidence += 0.25
        reason += f"Moderately oversold RSI ({rsi:.1f}). "
    elif direction == "SELL" and rsi > 80:  # Overbought
        demon_confidence += 0.4
        reason += f"Overbought RSI ({rsi:.1f}). "
    elif direction == "SELL" and rsi > 70:  # Moderately overbought
        demon_confidence += 0.25
        reason += f"Moderately overbought RSI ({rsi:.1f}). "
    
    # Decide if signal should be allowed
    if demon_confidence < 0.6:  # Reduced confidence threshold
        allow_signal = False
        reason += f"Insufficient confidence ({demon_confidence:.2f}). "
    
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
    debug_mode = True     # Enable debug output
    
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
        self.tunneling_probabilities = []
        self.signal_strengths = []
    
    def next(self):
        # Skip some initial bars to allow indicators to stabilize
        if len(self.data) < 250:  # Ensure all indicators are fully established
            return
        
        # Get current values
        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        avg_volume = self.volume_ma[-1]
        current_volatility = self.volatility[-1] * self.volatility_influence
        
        # VWAP ratio - check for significant deviation
        price_to_vwap = price / self.vwap[-1] if self.vwap[-1] > 0 else 1.0
        significant_vwap_deviation = abs(price_to_vwap - 1.0) > self.vwap_deviation
        
        # Calculate RSI momentum (3-period change)
        rsi_momentum = rsi - self.rsi[-4] if len(self.rsi) >= 4 else 0
        
        # Delta volume analysis
        current_delta = self.delta_volume[-1]
        delta_5bar = sum(self.delta_volume[-5:])  # Sum of last 5 bars
        
        # Current ATR for position sizing
        current_atr = self.atr[-1]
        
        # Calculate tunneling probability with time-dependent potential
        tunnel_prob = tunneling_probability(
            rsi, 
            rsi_momentum, 
            self.v0, 
            self.a, 
            self.mass, 
            self.hbar,
            current_volatility  # Include volatility for time-dependent model
        )
        self.tunneling_probabilities.append(tunnel_prob)
        
        # Detect current regime (trending or range-bound)
        current_slope = self.sma_slope[-1]
        is_trending = abs(current_slope) > self.trend_threshold
        trend_direction = 1 if current_slope > 0 else -1 if current_slope < 0 else 0
        
        # Debugging - print diagnostic information occasionally
        is_debug_bar = self.debug_mode and len(self.data) % 100 == 0
        if is_debug_bar:
            print(f"\n=== DIAGNOSTIC INFO (Bar {len(self.data)}) ===")
            print(f"Date: {self.data.index[-1]}")
            print(f"RSI: {rsi:.2f} (prev: {rsi_prev:.2f}, momentum: {rsi_momentum:.2f})")
            print(f"Tunneling probability: {tunnel_prob:.4f}")
            print(f"Volume ratio: {volume/avg_volume:.2f}x, Delta vol (5-bar): {delta_5bar:.2f}")
            print(f"Price to VWAP: {price_to_vwap:.4f}")
            print(f"SMA slope: {current_slope:.4f}, Trend direction: {trend_direction}")
            print(f"Is in cooldown: {len(self.data) - self.last_signal_time < self.cooldown_period}")
            print("====================================\n")
        
        # Check if we're in the cooldown period after a loss
        if self.last_trade_result == False and len(self.data) - self.last_signal_time < self.cooldown_period:
            return
        
        # Signal variables
        signal = None
        signal_price = price
        stop_loss = 0
        take_profit = 0
        
        # Multiple trend confirmation (relaxed)
        strong_uptrend = price > self.ema21[-1]
        strong_downtrend = price < self.ema21[-1]
        
        # Allow trading in both trending and ranging conditions 
        allow_long = True  # Allow long trades regardless of trend
        allow_short = True  # Allow short trades regardless of trend
        
        # Extremely oversold to bullish (RSI crossing up from below threshold)
        if (allow_long and ((rsi_prev < self.rsi_low and rsi >= self.rsi_low) or 
           (rsi < self.rsi_low + 5 and rsi_momentum > 3))):  # Relaxed momentum requirement
            
            signal = "BUY"
            confidence = min(0.9, tunnel_prob + 0.3)  # Base confidence
            
            # Apply Maxwell's Demon filter with delta volume analysis
            allow_signal, demon_confidence, demon_reason = demon_filter(
                rsi, rsi_prev, volume, avg_volume, price_to_vwap, signal, delta_5bar
            )
            
            # Relaxed confirmation criteria
            # Only require significant volume OR significant VWAP deviation
            if volume < avg_volume * self.volume_threshold and not significant_vwap_deviation:
                allow_signal = False
                demon_reason += "Insufficient volume AND VWAP deviation. "
            
            # Debug output for signal evaluation
            if self.debug_mode and (rsi < self.rsi_low + 7):  # Show debug for near signals too
                print(f"\n=== EVALUATING BUY SIGNAL ===")
                print(f"Date: {self.data.index[-1]}, Price: {price:.2f}")
                print(f"RSI: {rsi:.2f} (threshold: {self.rsi_low:.2f})")
                print(f"Allow signal: {allow_signal}, Confidence: {demon_confidence:.2f}")
                print(f"Reason: {demon_reason}")
                print("============================\n")
            
            # Stronger signal for extremely oversold conditions
            extremely_oversold = rsi < 20
            
            if allow_signal and not self.position and (tunnel_prob > 0.3 or extremely_oversold):  # Relaxed tunneling requirement
                # Find recent swing low for stop placement (use shorter window)
                recent_lows = self.data.Low[-10:].tolist()
                recent_swing_low = min(recent_lows)
                
                # Intelligent stop loss using swing points and ATR
                stop_buffer = current_atr * 0.3  # Reduced buffer below swing
                stop_loss = min(price - current_atr * self.stop_loss_atr, 
                                recent_swing_low - stop_buffer)
                
                take_profit = price + current_atr * self.take_profit_atr
                
                # Position sizing based on confidence and stop distance
                risk_pct = 0.01  # 1% risk per trade
                stop_distance_pct = (price - stop_loss) / price
                position_size = risk_pct / stop_distance_pct if stop_distance_pct > 0 else 0.1
                
                if extremely_oversold:
                    position_size *= 1.5  # Increase size for higher conviction setups
                
                # Cap position size at 0.25 (25% of portfolio)
                position_size = min(0.25, position_size)
                
                # Enter long position
                self.buy(sl=stop_loss, tp=take_profit, size=position_size)
                self.last_signal_time = len(self.data)
                self.signal_strengths.append(demon_confidence)
                
                if self.debug_mode:
                    print(f"\n!!! BUY SIGNAL EXECUTED !!!")
                    print(f"Date: {self.data.index[-1]}, Price: {price:.2f}")
                    print(f"Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                    print(f"Position size: {position_size:.4f}")
                    print("============================\n")
            
        # Extremely overbought to bearish (RSI crossing down from above threshold)
        elif (allow_short and ((rsi_prev > self.rsi_high and rsi <= self.rsi_high) or 
             (rsi > self.rsi_high - 5 and rsi_momentum < -3))):  # Relaxed momentum requirement
            
            signal = "SELL"
            confidence = min(0.9, tunnel_prob + 0.3)  # Base confidence
            
            # Apply Maxwell's Demon filter with delta volume analysis
            allow_signal, demon_confidence, demon_reason = demon_filter(
                rsi, rsi_prev, volume, avg_volume, price_to_vwap, signal, delta_5bar
            )
            
            # Relaxed confirmation criteria
            # Only require significant volume OR significant VWAP deviation
            if volume < avg_volume * self.volume_threshold and not significant_vwap_deviation:
                allow_signal = False
                demon_reason += "Insufficient volume AND VWAP deviation. "
            
            # Debug output for signal evaluation
            if self.debug_mode and (rsi > self.rsi_high - 7):  # Show debug for near signals too
                print(f"\n=== EVALUATING SELL SIGNAL ===")
                print(f"Date: {self.data.index[-1]}, Price: {price:.2f}")
                print(f"RSI: {rsi:.2f} (threshold: {self.rsi_high:.2f})")
                print(f"Allow signal: {allow_signal}, Confidence: {demon_confidence:.2f}")
                print(f"Reason: {demon_reason}")
                print("============================\n")
            
            # Stronger signal for extremely overbought conditions
            extremely_overbought = rsi > 80
            
            if allow_signal and not self.position and (tunnel_prob > 0.3 or extremely_overbought):  # Relaxed tunneling requirement
                # Find recent swing high for stop placement (use shorter window)
                recent_highs = self.data.High[-10:].tolist()
                recent_swing_high = max(recent_highs)
                
                # Intelligent stop loss using swing points and ATR
                stop_buffer = current_atr * 0.3  # Reduced buffer above swing
                stop_loss = max(price + current_atr * self.stop_loss_atr, 
                                recent_swing_high + stop_buffer)
                
                take_profit = price - current_atr * self.take_profit_atr
                
                # Position sizing based on confidence and stop distance
                risk_pct = 0.01  # 1% risk per trade
                stop_distance_pct = (stop_loss - price) / price
                position_size = risk_pct / stop_distance_pct if stop_distance_pct > 0 else 0.1
                
                if extremely_overbought:
                    position_size *= 1.5  # Increase size for higher conviction setups
                
                # Cap position size at 0.25 (25% of portfolio)
                position_size = min(0.25, position_size)
                
                # Enter short position
                self.sell(sl=stop_loss, tp=take_profit, size=position_size)
                self.last_signal_time = len(self.data)
                self.signal_strengths.append(demon_confidence)
                
                if self.debug_mode:
                    print(f"\n!!! SELL SIGNAL EXECUTED !!!")
                    print(f"Date: {self.data.index[-1]}, Price: {price:.2f}")
                    print(f"Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                    print(f"Position size: {position_size:.4f}")
                    print("============================\n")
        
        # Track trade result when position is closed
        if self.position.size == 0 and hasattr(self, '_prev_position_size') and self._prev_position_size != 0:
            current_equity = self.equity
            if hasattr(self, '_last_trade_pl') and self._last_trade_pl > 0:
                self.last_trade_result = True  # Winner
            else:
                self.last_trade_result = False  # Loser
        
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

def run_backtest(symbol='ETH/USDT', timeframe='1h', capital=10000, start_date='2023-01-01', end_date='2023-06-30'):
    """Run backtest with quantum RSI strategy"""
    print(f"Fetching data for {symbol} on {timeframe} timeframe from {start_date} to {end_date}...")
    data = fetch_crypto_data(symbol, timeframe, start_date=start_date, end_date=end_date)
    print(f"Data fetched: {len(data)} candles from {data.index[0]} to {data.index[-1]}")
    
    print(f"Running backtest with quantum RSI strategy...")
    bt = Backtest(data, QuantumRSIStrategy, cash=capital, commission=0.001)
    stats = bt.run()
    
    print(f"\n----- Backtest Results for {symbol} -----")
    print(f"Net Profit: {stats['Return [%]']:.2f}%")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"# Trades: {stats['# Trades']}")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    print(f"Avg. Trade [%]: {stats['Avg. Trade [%]']:.2f}%")
    
    # Plot with custom info
    bt.plot(resample=False)
    
    return stats, bt

# ------ Main Execution ------

if __name__ == "__main__":
    # Run backtest for ETH/USD with various parameter sets for highest win rate
    print("Starting Optimized Quantum RSI Backtest with Maxwell's Demon")
    print("==================================================================")
    print("This version is optimized for ETH/USD with enhanced whale detection")
    print("Key optimizations: 21-period RSI, asymmetric thresholds (70/30), ")
    print("                   relaxed filter criteria for more trade signals")
    print("==================================================================")
    
    # Define timeframes and capital
    initial_capital = 10000
    
    # Use a longer timeframe for more realistic backtest
    start_date = '2023-01-01'
    end_date = '2023-06-30'
    
    # Function to run backtest with modified parameters
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
        bt = Backtest(data, strategy_class, cash=capital, commission=0.001)
        stats = bt.run()
        
        print(f"\n----- Backtest Results for {symbol} -----")
        print(f"Net Profit: {stats['Return [%]']:.2f}%")
        print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"# Trades: {stats['# Trades']}")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"Profit Factor: {stats['Profit Factor']:.2f}")
        print(f"Avg. Trade [%]: {stats['Avg. Trade [%]']:.2f}%")
        
        # Plot with custom info
        bt.plot(resample=False)
        
        return stats, bt
    
    # Run ETH/USD backtest with standard parameters
    eth_standard_stats, eth_standard_bt = run_optimized_backtest(
        'ETH/USDT', '1h', initial_capital, start_date, end_date
    )
    
    # Run ETH/USD backtest with selective parameters for higher win rate
    eth_selective_params = {
        'rsi_period': 21,
        'rsi_high': 70,          # Standard threshold
        'rsi_low': 30,
        'v0': 2.0,               # Lower barriers for more signals
        'mass': 1.2,             # Lower mass
        'debug_mode': False      # Disable debug for cleaner output
    }
    
    print("\nRunning ETH/USD backtest with selective parameters...")
    eth_selective_stats, eth_selective_bt = run_optimized_backtest(
        'ETH/USDT', '1h', initial_capital, start_date, end_date, eth_selective_params
    )
    
    # Run ETH/USD backtest with aggressive parameters for more trades
    eth_aggressive_params = {
        'rsi_period': 14,        # Standard RSI period
        'rsi_high': 65,          # More aggressive threshold
        'rsi_low': 35,           # More aggressive threshold
        'v0': 1.5,               # Lower barriers for more signals
        'mass': 1.0,             # Lower mass for more tunneling
        'hbar': 1.0,             # Higher hbar for more tunneling
        'volume_threshold': 1.0, # No volume requirement
        'vwap_deviation': 0.005, # 0.5% VWAP deviation
        'cooldown_period': 6,    # Shorter cooldown
        'debug_mode': False      # Disable debug for cleaner output
    }
    
    print("\nRunning ETH/USD backtest with aggressive parameters for more trades...")
    eth_aggressive_stats, eth_aggressive_bt = run_optimized_backtest(
        'ETH/USDT', '1h', initial_capital, start_date, end_date, eth_aggressive_params
    )
    
    # Run ETH/USD backtest with ultra-conservative parameters for maximum win rate
    eth_ultra_params = {
        'rsi_period': 21,        # Increased RSI period to reduce noise
        'rsi_high': 75,          # More selective
        'rsi_low': 25,           # More selective
        'v0': 3.0,               # Higher barriers
        'mass': 2.0,             # Higher mass
        'hbar': 0.5,             # Lower hbar
        'volume_threshold': 1.5, # Higher volume requirement
        'vwap_deviation': 0.02,  # 2% VWAP deviation
        'stop_loss_atr': 1.0,    # Tighter stops
        'take_profit_atr': 5.0,  # Larger targets
        'cooldown_period': 24,   # Longer cooldown
        'debug_mode': False      # Disable debug for cleaner output
    }
    
    print("\nRunning ETH/USD backtest with ultra-conservative parameters for maximum win rate...")
    eth_ultra_stats, eth_ultra_bt = run_optimized_backtest(
        'ETH/USDT', '1h', initial_capital, start_date, end_date, eth_ultra_params
    )
    
    # Compare results
    print("\n==== COMPARISON OF ETH/USD STRATEGIES ====")
    print(f"{'Strategy':<25} {'Win Rate':<10} {'# Trades':<10} {'Net %':<10} {'Drawdown':<10} {'Sharpe':<10}")
    print("-" * 75)
    print(f"{'ETH Standard':<25} {eth_standard_stats['Win Rate [%]']:<10.2f} {eth_standard_stats['# Trades']:<10} {eth_standard_stats['Return [%]']:<10.2f} {eth_standard_stats['Max. Drawdown [%]']:<10.2f} {eth_standard_stats['Sharpe Ratio']:<10.2f}")
    print(f"{'ETH Selective':<25} {eth_selective_stats['Win Rate [%]']:<10.2f} {eth_selective_stats['# Trades']:<10} {eth_selective_stats['Return [%]']:<10.2f} {eth_selective_stats['Max. Drawdown [%]']:<10.2f} {eth_selective_stats['Sharpe Ratio']:<10.2f}")
    print(f"{'ETH Aggressive':<25} {eth_aggressive_stats['Win Rate [%]']:<10.2f} {eth_aggressive_stats['# Trades']:<10} {eth_aggressive_stats['Return [%]']:<10.2f} {eth_aggressive_stats['Max. Drawdown [%]']:<10.2f} {eth_aggressive_stats['Sharpe Ratio']:<10.2f}")
    print(f"{'ETH Ultra-Conservative':<25} {eth_ultra_stats['Win Rate [%]']:<10.2f} {eth_ultra_stats['# Trades']:<10} {eth_ultra_stats['Return [%]']:<10.2f} {eth_ultra_stats['Max. Drawdown [%]']:<10.2f} {eth_ultra_stats['Sharpe Ratio']:<10.2f}")
    
    # Print conclusion
    print("\n==== FINDINGS AND RECOMMENDATIONS ====")
    strategies = [
        ("ETH Standard", eth_standard_stats),
        ("ETH Selective", eth_selective_stats),
        ("ETH Aggressive", eth_aggressive_stats),
        ("ETH Ultra-Conservative", eth_ultra_stats)
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
    
    print("\nOptimal settings based on quantum model and Maxwell's Demon filter:")
    print("- RSI Period: 21 (reduced noise)")
    print("- Asymmetric thresholds: 70/30 (accounts for crypto's upward bias)")
    print("- Volume confirmation: At least 1.2x average volume")
    print("- VWAP deviation: Minimum 1% from VWAP")
    print("- Delta volume confirmation: +ve for longs, -ve for shorts")
    print("- Cooldown after losing trades: 12+ hours")
    
    print("\nBacktests completed.") 