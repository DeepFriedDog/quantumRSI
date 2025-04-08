"""
Quantum RSI Trading Strategy with Maxwell's Demon Filter

This module implements an optimized version of the Quantum RSI strategy
with parameters tuned for higher win rates through enhanced signal filtering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import ccxt
import os
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# Ensure the results directory exists
os.makedirs("results", exist_ok=True)

# Optimized parameters based on quantum model analysis
class QuantumRSIParameters:
    def __init__(self, market_type="crypto"):
        # Core quantum parameters
        self.mass = 1.8 if market_type == "crypto" else 1.5  # Higher for crypto markets (more momentum persistence)
        self.hbar = 1.5  # Reduced Planck constant analog (uncertainty parameter)
        self.v0 = 2.2  # Potential barrier height - higher for range-bound markets
        
        # RSI parameters
        self.rsi_period = 14
        self.rsi_high = 70
        self.rsi_low = 30
        
        # Demon filter parameters
        self.volume_threshold = 1.8  # Volume spike detection threshold
        self.vwap_deviation = 0.01  # Price deviation from VWAP (normalized by ATR)
        self.atr_period = 14
        
        # Risk management
        self.stop_loss_atr = 1.5  # Wider stops to avoid noise
        self.take_profit_atr = 1.2  # Closer take profit for higher win rate
        self.cooldown_period = 8  # Bars to wait after a trade
        self.max_consecutive_losses = 3  # Increase selectivity after losses

def calculate_indicators(df, params):
    """Calculate all indicators needed for the strategy"""
    # Basic indicators using ta library
    rsi_indicator = RSIIndicator(df['close'], window=params.rsi_period)
    df['rsi'] = rsi_indicator.rsi()
    
    atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=params.atr_period)
    df['atr'] = atr_indicator.average_true_range()
    
    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # VWAP calculation (simplified as cumulative for the available data)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['atr']
    
    # Price momentum
    df['price_change'] = df['close'].pct_change()
    df['price_momentum'] = df['price_change'].rolling(window=3).sum()
    
    # Quantum state calculation (RSI mapped to position coordinate)
    df['x'] = (df['rsi'] - 50) / 50  # Maps RSI 0-100 to x coordinate -1 to +1
    
    # Tunneling probability calculation
    df['barrier_height'] = compute_barrier_height(df, params)
    df['tunneling_probability'] = compute_tunneling_probability(df, params)
    
    # Volume delta approximation (if available)
    if 'volume_delta' not in df.columns and all(col in df.columns for col in ['open', 'close', 'volume']):
        df['volume_delta'] = df['volume'] * np.sign(df['close'] - df['open'])
    
    # Rolling delta to detect changes in buying/selling pressure
    if 'volume_delta' in df.columns:
        df['delta_momentum'] = df['volume_delta'].rolling(window=5).sum()
    
    return df

def compute_barrier_height(df, params):
    """
    Dynamically compute potential barrier height based on market conditions.
    Lower barrier = easier trend reversal, higher barrier = stronger trend persistence.
    """
    # Base barrier height from parameters
    barrier = np.ones(len(df)) * params.v0
    
    # Adjust barrier based on market volatility
    if 'atr' in df.columns:
        volatility = df['atr'] / df['close']
        relative_vol = volatility / volatility.rolling(window=50).mean()
        
        # Higher volatility = lower barrier (easier reversal)
        barrier = barrier * (1.0 / np.clip(relative_vol, 0.5, 2.0))
    
    # Adjust barrier based on volume (high volume can break barriers)
    if 'volume_ratio' in df.columns:
        # Very high volume lowers barriers (big players can push through resistance)
        volume_factor = np.clip(2.0 / df['volume_ratio'], 0.5, 1.5)
        barrier = barrier * volume_factor
    
    return barrier

def compute_tunneling_probability(df, params):
    """
    Calculate quantum tunneling probability for trend reversal based on the
    WKB approximation from quantum mechanics.
    """
    # Map RSI to position coordinate x
    x = df['x']
    
    # Energy relative to barrier height (RSI momentum)
    # Energy is higher when RSI approaches middle from extreme
    energy = params.mass * (1.0 - x**2)
    
    # Barrier parameters
    barrier_height = df['barrier_height']
    a = 0.4  # Well position parameter (RSI ~70 and ~30)
    
    # Simplified WKB tunneling probability calculation
    # T ≈ exp(-2κa) where κ = sqrt(2m(V-E))/ℏ
    
    # Classical turning points approximation
    x1 = np.clip(0.25 * x, -0.35, 0.35)  # Simplified approximation of turning points
    barrier_width = 2 * abs(x1)  # Distance between turning points
    
    # Energy difference from barrier top
    energy_deficit = np.maximum(barrier_height - energy, 0)
    
    # Tunneling probability calculation (WKB approximation)
    kappa = np.sqrt(2 * params.mass * energy_deficit) / params.hbar
    tunneling_probability = np.exp(-2 * kappa * barrier_width)
    
    # Tunneling is only relevant near barriers
    tunneling_probability = np.where(
        (x > 0.2) | (x < -0.2),  # Only in regions near wells
        tunneling_probability,
        1.0  # No barrier in middle region
    )
    
    # Clean up any NaN values
    tunneling_probability = np.nan_to_num(tunneling_probability, nan=0.01)
    
    return tunneling_probability

def maxwell_demon_filter(df, index, params):
    """
    The Maxwell's Demon implementation that selectively filters trading signals
    based on information about institutional activity (volume, delta, VWAP).
    
    Returns:
        tuple: (approved, confidence, adjusted_barrier)
    """
    # Default values
    approved = False
    confidence = 0
    adjusted_barrier = 0
    
    # Check if we have enough data
    if index < 20:
        return approved, confidence, adjusted_barrier
    
    # Get current values
    current = df.iloc[index]
    prev = df.iloc[index-1]
    
    # Base probability of trend shift
    base_probability = current['tunneling_probability']
    
    # === Maxwell's Demon Information Processing ===
    
    # 1. Check for volume spikes (indication of institutional activity)
    volume_spike = current['volume_ratio'] > params.volume_threshold
    
    # 2. Check for price-VWAP relationship
    extended_price = abs(current['vwap_distance']) > params.vwap_deviation
    
    # 3. Check for delta volume (if available)
    delta_change = False
    delta_confirmation = False
    
    if 'delta_momentum' in df.columns:
        delta_change = (np.sign(current['delta_momentum']) != 
                       np.sign(df.iloc[index-5:index]['delta_momentum'].mean()))
        
        # Delta confirms potential reversal when it shows exhaustion or reversal
        # For buy: delta turning positive in oversold
        # For sell: delta turning negative in overbought
        if current['rsi'] < params.rsi_low and current['delta_momentum'] > 0:
            delta_confirmation = True  # Bullish confirmation
        elif current['rsi'] > params.rsi_high and current['delta_momentum'] < 0:
            delta_confirmation = True  # Bearish confirmation
    
    # 4. Check for RSI momentum change (reversal indication)
    rsi_change = (np.sign(current['rsi'] - prev['rsi']) != 
                  np.sign(prev['rsi'] - df.iloc[index-2]['rsi']))
    
    # === Apply Demon Logic - Information-Based Barrier Adjustment ===
    
    # Bullish scenario (oversold, looking for long)
    if current['rsi'] < params.rsi_low:
        # Adjust barrier based on information
        adjusted_barrier = params.v0  # Start with base
        
        # Volume spike with price staying low = accumulation = lower barrier (easier to reverse)
        if volume_spike and current['price_change'] < 0.001:
            adjusted_barrier *= 0.7  # Significant reduction
            confidence += 2
        
        # Price far below VWAP with RSI turning = likely oversold
        if extended_price and current['vwap_distance'] < 0 and rsi_change:
            adjusted_barrier *= 0.8
            confidence += 1
        
        # Delta momentum confirming bottom
        if delta_confirmation:
            adjusted_barrier *= 0.7
            confidence += 2
        
        # Approve if confidence is sufficient
        approved = confidence >= 2
    
    # Bearish scenario (overbought, looking for short)
    elif current['rsi'] > params.rsi_high:
        # Adjust barrier based on information
        adjusted_barrier = params.v0  # Start with base
        
        # Volume spike with price not climbing higher = distribution = lower barrier
        if volume_spike and current['price_change'] < 0.001:
            adjusted_barrier *= 0.7
            confidence += 2
        
        # Price far above VWAP with RSI turning = likely overbought
        if extended_price and current['vwap_distance'] > 0 and rsi_change:
            adjusted_barrier *= 0.8
            confidence += 1
        
        # Delta momentum confirming top
        if delta_confirmation:
            adjusted_barrier *= 0.7
            confidence += 2
        
        # Approve if confidence is sufficient
        approved = confidence >= 2
    
    return approved, confidence, adjusted_barrier

def fetch_crypto_data(symbol, timeframe, start_date, end_date=None, exchange_id='binance'):
    """
    Fetch cryptocurrency historical data
    """
    # Initialize exchange
    exchange = getattr(ccxt, exchange_id)()
    
    # Convert dates to timestamps
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    until = None if end_date is None else exchange.parse8601(f"{end_date}T23:59:59Z")
    
    # Fetch data
    all_candles = []
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not candles or len(candles) == 0:
                break
            
            all_candles.extend(candles)
            
            # Update since timestamp for next iteration
            since = candles[-1][0] + 1
            
            # Break if we reached the end date
            if until and since >= until:
                break
                
            # Exchange-specific rate limiting
            if exchange_id == 'binance':
                import time
                time.sleep(1)  # Respect rate limits
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

def run_backtest(df, params, initial_capital=10000, plot=True):
    """
    Run a backtest of the Quantum RSI strategy with Maxwell's Demon filter
    """
    # Calculate indicators
    df = calculate_indicators(df, params)
    
    # Initialize trading metrics
    equity = [initial_capital]
    positions = []
    trades = []
    consecutive_losses = 0
    in_position = False
    entry_price = 0
    position_type = None  # 'long' or 'short'
    entry_index = 0
    cooldown_counter = 0
    
    # Prepare for tracking
    trade_log = []
    
    # Loop through data (start from index 20 to have enough data for indicators)
    for i in range(20, len(df)):
        current_price = df.iloc[i]['close']
        current_date = df.index[i]
        
        # Skip if in cooldown period
        if cooldown_counter > 0:
            cooldown_counter -= 1
            equity.append(equity[-1])
            continue
        
        # Check if we need to close a position
        if in_position:
            # Calculate profit/loss
            if position_type == 'long':
                profit_pct = (current_price - entry_price) / entry_price
                stop_price = entry_price * (1 - params.stop_loss_atr * df.iloc[entry_index]['atr'] / entry_price)
                take_profit = entry_price * (1 + params.take_profit_atr * df.iloc[entry_index]['atr'] / entry_price)
                
                # Check stop loss or take profit
                if current_price <= stop_price or current_price >= take_profit:
                    # Close position
                    trade_result = (current_price - entry_price) / entry_price
                    equity_change = equity[-1] * (1 + trade_result)
                    equity.append(equity_change)
                    
                    # Record trade
                    exit_reason = 'Stop Loss' if current_price <= stop_price else 'Take Profit'
                    trade_data = {
                        'entry_date': df.index[entry_index],
                        'exit_date': current_date,
                        'position': position_type,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': trade_result * 100,
                        'result': exit_reason
                    }
                    trade_log.append(trade_data)
                    trades.append(trade_result)
                    
                    # Update consecutive losses
                    if trade_result < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    
                    # Reset position
                    in_position = False
                    cooldown_counter = params.cooldown_period
                    
                    continue
            
            elif position_type == 'short':
                profit_pct = (entry_price - current_price) / entry_price
                stop_price = entry_price * (1 + params.stop_loss_atr * df.iloc[entry_index]['atr'] / entry_price)
                take_profit = entry_price * (1 - params.take_profit_atr * df.iloc[entry_index]['atr'] / entry_price)
                
                # Check stop loss or take profit
                if current_price >= stop_price or current_price <= take_profit:
                    # Close position
                    trade_result = (entry_price - current_price) / entry_price
                    equity_change = equity[-1] * (1 + trade_result)
                    equity.append(equity_change)
                    
                    # Record trade
                    exit_reason = 'Stop Loss' if current_price >= stop_price else 'Take Profit'
                    trade_data = {
                        'entry_date': df.index[entry_index],
                        'exit_date': current_date,
                        'position': position_type,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': trade_result * 100,
                        'result': exit_reason
                    }
                    trade_log.append(trade_data)
                    trades.append(trade_result)
                    
                    # Update consecutive losses
                    if trade_result < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    
                    # Reset position
                    in_position = False
                    cooldown_counter = params.cooldown_period
                    
                    continue
        
        # Apply Maxwell's Demon filter to check for valid signals
        approved, confidence, adjusted_barrier = maxwell_demon_filter(df, i, params)
        
        # Adjust RSI thresholds based on consecutive losses
        adjusted_rsi_high = params.rsi_high + (consecutive_losses * 2)
        adjusted_rsi_low = params.rsi_low - (consecutive_losses * 2)
        
        # Check for new position if we're not in one
        if not in_position:
            # Calculate the tunneling probability (reversal likelihood)
            tunneling_prob = df.iloc[i]['tunneling_probability']
            
            # Check for bullish signal (oversold to reversal)
            if (df.iloc[i]['rsi'] < adjusted_rsi_low and 
                df.iloc[i]['rsi'] > df.iloc[i-1]['rsi'] and  # RSI turning up
                approved and  # Demon approves
                tunneling_prob > 0.3):  # Sufficient tunneling probability
                
                # Enter long position
                in_position = True
                position_type = 'long'
                entry_price = current_price
                entry_index = i
                
                # Don't update equity yet - wait for exit
                
            # Check for bearish signal (overbought to reversal)
            elif (df.iloc[i]['rsi'] > adjusted_rsi_high and 
                  df.iloc[i]['rsi'] < df.iloc[i-1]['rsi'] and  # RSI turning down
                  approved and  # Demon approves
                  tunneling_prob > 0.3):  # Sufficient tunneling probability
                
                # Enter short position
                in_position = True
                position_type = 'short'
                entry_price = current_price
                entry_index = i
                
                # Don't update equity yet - wait for exit
        
        # If no trade, equity remains the same
        if equity[-1] != equity[-1]:  # Check for NaN
            equity.append(equity[-2])
        elif len(equity) <= i - 19:
            equity.append(equity[-1])
    
    # Close any remaining position at the end
    if in_position:
        current_price = df.iloc[-1]['close']
        if position_type == 'long':
            trade_result = (current_price - entry_price) / entry_price
        else:  # short
            trade_result = (entry_price - current_price) / entry_price
        
        equity_change = equity[-1] * (1 + trade_result)
        equity.append(equity_change)
        
        trade_data = {
            'entry_date': df.index[entry_index],
            'exit_date': df.index[-1],
            'position': position_type,
            'entry_price': entry_price,
            'exit_price': current_price,
            'profit_pct': trade_result * 100,
            'result': 'End of Backtest'
        }
        trade_log.append(trade_data)
        trades.append(trade_result)
    
    # Convert trade log to DataFrame
    trade_df = pd.DataFrame(trade_log)
    
    # Calculate performance metrics
    if len(trades) > 0:
        win_rate = len([t for t in trades if t > 0]) / len(trades) * 100
        profit_factor = sum([t for t in trades if t > 0]) / abs(sum([t for t in trades if t < 0])) if sum([t for t in trades if t < 0]) != 0 else float('inf')
        avg_win = sum([t for t in trades if t > 0]) / len([t for t in trades if t > 0]) if len([t for t in trades if t > 0]) > 0 else 0
        avg_loss = sum([t for t in trades if t < 0]) / len([t for t in trades if t < 0]) if len([t for t in trades if t < 0]) > 0 else 0
        max_drawdown = 0
        peak = equity[0]
        
        for e in equity:
            if e > peak:
                peak = e
            drawdown = (peak - e) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Final equity
        final_equity = equity[-1]
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        print(f"=== Quantum RSI Backtest Results ===")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Win: {avg_win*100:.2f}%")
        print(f"Average Loss: {avg_loss*100:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
        print(f"Total Trades: {len(trades)}")
        print(f"Final Equity: ${final_equity:.2f}")
    else:
        print("No trades executed during the backtest period.")
        win_rate = 0
        profit_factor = 0
        total_return = 0
        max_drawdown = 0
    
    # Plot results if requested
    if plot and len(equity) > 1:
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Price and RSI with trade markers
        plt.subplot(2, 1, 1)
        plt.plot(df.index[20:], df['close'][20:], label='Price')
        
        # Add buy/sell markers
        for trade in trade_log:
            if trade['position'] == 'long':
                plt.scatter(trade['entry_date'], trade['entry_price'], color='green', marker='^', s=100)
                plt.scatter(trade['exit_date'], trade['exit_price'], color='red', marker='v', s=100)
            else:  # short
                plt.scatter(trade['entry_date'], trade['entry_price'], color='red', marker='v', s=100)
                plt.scatter(trade['exit_date'], trade['exit_price'], color='green', marker='^', s=100)
        
        plt.title('Price Chart with Trade Signals')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: Equity Curve - Fix potential dimension mismatch
        plt.subplot(2, 1, 2)
        
        # Ensure x and y have the same length
        if len(equity) > len(df.index) - 20:
            # If equity is longer, trim it to match the available x values
            equity_plot = equity[:len(df.index)-20]
        else:
            # If equity is shorter, use only the available data points
            equity_plot = equity
        
        # Plot using correctly sized arrays
        x_vals = df.index[20:20+len(equity_plot)]
        plt.plot(x_vals, equity_plot, label='Equity Curve')
        
        plt.title('Equity Curve')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"results/quantum_rsi_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        # Save trade log
        if not trade_df.empty:
            trade_df.to_csv(f"results/trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    
    results = {
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'trades': len(trades),
        'equity_curve': equity,
        'trade_log': trade_log
    }
    
    return results

def run_multi_pair_test(pairs, timeframe, start_date, end_date, params=None):
    """Run backtest on multiple pairs and compare results"""
    if params is None:
        params = QuantumRSIParameters()
    
    results = {}
    
    for pair in pairs:
        print(f"Testing {pair}...")
        df = fetch_crypto_data(pair, timeframe, start_date, end_date)
        pair_result = run_backtest(df, params, plot=False)
        results[pair] = pair_result
        
        print(f"{pair} Win Rate: {pair_result['win_rate']:.2f}%")
        print(f"{pair} Total Return: {pair_result['total_return']:.2f}%")
        print(f"{pair} Trades: {pair_result['trades']}")
        print("-------------------")
    
    # Summarize results
    win_rates = [r['win_rate'] for r in results.values()]
    returns = [r['total_return'] for r in results.values()]
    
    print(f"Average Win Rate: {np.mean(win_rates):.2f}%")
    print(f"Average Return: {np.mean(returns):.2f}%")
    
    # Plot comparison
    pairs_list = list(results.keys())
    win_rates = [results[p]['win_rate'] for p in pairs_list]
    returns = [results[p]['total_return'] for p in pairs_list]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(pairs_list, win_rates)
    plt.title('Win Rates by Pair')
    plt.ylabel('Win Rate (%)')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(pairs_list, returns)
    plt.title('Total Returns by Pair')
    plt.ylabel('Return (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"results/multi_pair_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    return results

if __name__ == "__main__":
    # Import numpy patch to fix bool8 issue
    import numpy_patch
    
    # Example usage
    params = QuantumRSIParameters()
    
    # Single pair backtest
    symbol = "ETH/USDT"
    timeframe = "1h"
    start_date = "2023-01-01"
    end_date = "2023-07-01"
    
    print(f"Running backtest on {symbol} from {start_date} to {end_date}...")
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    result = run_backtest(df, params, plot=True)
    
    # Optionally run multi-pair test
    # pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
    # multi_results = run_multi_pair_test(pairs, timeframe, start_date, end_date, params) 