"""
Simplified Quantum RSI Backtesting implementation
"""
import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Import the tunneling_probability function from our main file
from quantum_crypto_backtest import tunneling_probability, calculate_rsi, demon_filter

def calculate_indicators(df, rsi_period=14, volatility_window=20):
    """Calculate all the indicators needed for the strategy"""
    # RSI
    df['rsi'] = calculate_rsi(df['Close'].values, rsi_period)
    
    # Volatility
    df['volatility'] = df['Close'].pct_change().rolling(volatility_window).std()
    
    # Simple moving averages
    df['sma50'] = df['Close'].rolling(50).mean()
    df['sma200'] = df['Close'].rolling(200).mean()
    df['ema21'] = df['Close'].ewm(span=21).mean()
    
    # VWAP calculation (approximate using daily reset)
    df['vwap'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    
    # Volume indicators
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    
    # Calculate delta volume
    df['delta_volume'] = df['Volume'] * df['Close'].pct_change()
    
    # Calculate SMA slope
    df['sma_slope'] = df['sma50'].pct_change(5)
    
    # ATR calculation
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

def fetch_crypto_data(symbol='ETH/USDT', timeframe='1h', limit=5000, start_date=None, end_date=None):
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
    
    # Rename columns for consistency
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df

def run_simple_backtest(data, params, symbol='ETH/USDT', timeframe='1h'):
    """Run a simplified backtest with the Quantum RSI strategy"""
    # Initialize portfolio and trade tracking
    initial_capital = params.get('initial_capital', 10000)
    equity = initial_capital
    position_size = 0
    in_position = False
    position_type = None  # 'long' or 'short'
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trades = []
    equity_curve = [initial_capital]
    
    # Strategy parameters
    rsi_high = params.get('rsi_high', 70)
    rsi_low = params.get('rsi_low', 30)
    v0 = params.get('v0', 2.0)
    a = params.get('a', 1.0)
    mass = params.get('mass', 1.5)
    hbar = params.get('hbar', 1.0)
    stop_loss_atr = params.get('stop_loss_atr', 1.5)     # Wider stop loss (was 1.0)
    take_profit_atr = params.get('take_profit_atr', 1.2)  # Closer take profit (was 2.0)
    debug_mode = params.get('debug_mode', False)
    
    # Add cooldown tracking
    last_trade_exit_bar = 0
    cooldown_period = params.get('cooldown_period', 12)
    
    # Add tracking for consecutive losses
    consecutive_losses = 0
    max_consecutive_losses = params.get('max_consecutive_losses', 3)
    
    # Track successful trade setups to become more selective after losses
    successful_patterns = []
    
    for i in range(250, len(data)):  # Start after indicators are established
        date = data.index[i]
        price = data['Close'].iloc[i]
        rsi = data['rsi'].iloc[i]
        rsi_prev = data['rsi'].iloc[i-1]
        rsi_prev2 = data['rsi'].iloc[i-2]
        rsi_prev3 = data['rsi'].iloc[i-3] if i > 3 else rsi_prev2
        
        # Current ATR for position sizing and stop placement
        current_atr = data['atr'].iloc[i]
        
        # Check if we're in a position and update P&L
        if in_position:
            # Check if we hit stop loss or take profit
            if position_type == 'long':
                if price <= stop_loss:  # Hit stop loss
                    profit_loss = (price / entry_price - 1) * position_size * equity
                    equity += profit_loss
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'type': position_type,
                        'profit_loss': profit_loss,
                        'result': 'Stop Loss'
                    })
                    if debug_mode:
                        print(f"STOP LOSS: Closed LONG at {price:.2f} for P&L: {profit_loss:.2f}")
                    in_position = False
                    last_trade_exit_bar = i
                    
                    # Track consecutive losses
                    if profit_loss < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    
                elif price >= take_profit:  # Hit take profit
                    profit_loss = (price / entry_price - 1) * position_size * equity
                    equity += profit_loss
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'type': position_type,
                        'profit_loss': profit_loss,
                        'result': 'Take Profit'
                    })
                    if debug_mode:
                        print(f"TAKE PROFIT: Closed LONG at {price:.2f} for P&L: {profit_loss:.2f}")
                    in_position = False
                    last_trade_exit_bar = i
                    
                    # Reset consecutive losses on profitable trade
                    consecutive_losses = 0
                    
                    # Save successful pattern
                    entry_rsi = trade_entry_rsi
                    entry_momentum = trade_entry_momentum
                    successful_patterns.append({
                        'rsi': entry_rsi,
                        'momentum': entry_momentum,
                        'type': 'long'
                    })
                    
            elif position_type == 'short':
                if price >= stop_loss:  # Hit stop loss
                    profit_loss = (entry_price / price - 1) * position_size * equity
                    equity += profit_loss
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'type': position_type,
                        'profit_loss': profit_loss,
                        'result': 'Stop Loss'
                    })
                    if debug_mode:
                        print(f"STOP LOSS: Closed SHORT at {price:.2f} for P&L: {profit_loss:.2f}")
                    in_position = False
                    last_trade_exit_bar = i
                    
                    # Track consecutive losses
                    if profit_loss < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    
                elif price <= take_profit:  # Hit take profit
                    profit_loss = (entry_price / price - 1) * position_size * equity
                    equity += profit_loss
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'type': position_type,
                        'profit_loss': profit_loss,
                        'result': 'Take Profit'
                    })
                    if debug_mode:
                        print(f"TAKE PROFIT: Closed SHORT at {price:.2f} for P&L: {profit_loss:.2f}")
                    in_position = False
                    last_trade_exit_bar = i
                    
                    # Reset consecutive losses on profitable trade
                    consecutive_losses = 0
                    
                    # Save successful pattern
                    entry_rsi = trade_entry_rsi
                    entry_momentum = trade_entry_momentum
                    successful_patterns.append({
                        'rsi': entry_rsi,
                        'momentum': entry_momentum,
                        'type': 'short'
                    })
        
        # Only look for new signals if we're not in a position and not in cooldown
        if not in_position and (i - last_trade_exit_bar > cooldown_period):
            # Calculate current momentum and tunneling probability
            rsi_momentum = rsi - rsi_prev
            rsi_momentum_prev = rsi_prev - rsi_prev2
            current_volatility = data['volatility'].iloc[i]
            tunnel_prob = tunneling_probability(rsi, rsi_momentum, v0, a, mass, hbar, current_volatility)
            
            # Check for trend using SMA slope
            sma_slope = data['sma_slope'].iloc[i] if 'sma_slope' in data.columns else 0
            
            # VWAP ratio for filtering
            price_to_vwap = price / data['vwap'].iloc[i] if data['vwap'].iloc[i] > 0 else 1.0
            
            # Volume for filtering
            volume = data['Volume'].iloc[i]
            avg_volume = data['volume_ma'].iloc[i]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            # Check if there's a price reversal pattern forming
            has_reversal_pattern = False
            
            # For BUY: Check if previous candles show a bottoming pattern
            if (data['Close'].iloc[i] > data['Close'].iloc[i-1] and 
                data['Close'].iloc[i-1] <= data['Close'].iloc[i-2] and
                data['Low'].iloc[i-1] < data['Low'].iloc[i-2]):
                has_reversal_pattern = True
                
            # For SELL: Check if previous candles show a topping pattern
            if (data['Close'].iloc[i] < data['Close'].iloc[i-1] and 
                data['Close'].iloc[i-1] >= data['Close'].iloc[i-2] and
                data['High'].iloc[i-1] > data['High'].iloc[i-2]):
                has_reversal_pattern = True
            
            # Initialize signal
            signal = None
            allow_signal = True
            reason = ""
            
            # Adjust thresholds based on consecutive losses
            rsi_threshold_modifier = min(consecutive_losses * 3, 10)
            effective_rsi_high = rsi_high + rsi_threshold_modifier
            effective_rsi_low = rsi_low - rsi_threshold_modifier
            
            # Minimum probability threshold, increased after losses
            min_probability = 0.4 + (consecutive_losses * 0.1)
            
            # BUY signals with improved criteria for higher win rate
            if (
                # Classic oversold reversal - more stringent requirements
                (rsi_prev < effective_rsi_low and rsi >= effective_rsi_low and rsi_momentum > 3) or 
                # Strong momentum from near-oversold area with confirmation
                (rsi < effective_rsi_low + 5 and rsi_momentum > 4 and rsi_momentum_prev < 0 and has_reversal_pattern) or
                # Higher low formation in oversold area
                (rsi < 45 and rsi > rsi_prev and rsi_prev < rsi_prev2 and rsi_prev3 > rsi_prev2 and sma_slope > 0)
            ):
                signal = "BUY"
                # Apply Maxwell's Demon filter
                delta_vol = data['delta_volume'].iloc[i]
                allow_signal, demon_confidence, demon_reason = demon_filter(
                    rsi, rsi_prev, volume, avg_volume, price_to_vwap, signal, delta_vol
                )
                
                # Additional confirmation for higher win rate
                if allow_signal:
                    # Only enter during increasing volume and sufficient tunneling probability
                    if volume_ratio >= 1.5 and tunnel_prob > min_probability and price_to_vwap < 0.99:
                        # Ensure we're not entering right before resistance (previous high)
                        recent_high = max(data['High'].iloc[i-10:i])
                        if (recent_high - price) / price > 0.01:  # At least 1% room to move up
                            # Calculate stop loss and take profit - asymmetric risk-reward
                            sl = price - stop_loss_atr * current_atr
                            tp = price + take_profit_atr * current_atr
                            
                            # Enter long position
                            position_type = 'long'
                            in_position = True
                            entry_price = price
                            stop_loss = sl
                            take_profit = tp
                            entry_date = date
                            position_size = 0.2  # Use 20% of equity for position
                            
                            # Save entry RSI for pattern tracking
                            trade_entry_rsi = rsi
                            trade_entry_momentum = rsi_momentum
                            
                            if debug_mode:
                                print(f"BUY SIGNAL: Entered LONG at {price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}, RSI: {rsi:.1f}")
                                print(f"Reason: RSI {rsi:.1f} with momentum {rsi_momentum:.1f}, Tunneling prob: {tunnel_prob:.2f}")
                    
            # SELL signals with improved criteria for higher win rate
            elif (
                # Classic overbought reversal - more stringent requirements
                (rsi_prev > effective_rsi_high and rsi <= effective_rsi_high and rsi_momentum < -3) or 
                # Strong negative momentum from near-overbought area with confirmation
                (rsi > effective_rsi_high - 5 and rsi_momentum < -4 and rsi_momentum_prev > 0 and has_reversal_pattern) or
                # Lower high formation in overbought area
                (rsi > 55 and rsi < rsi_prev and rsi_prev > rsi_prev2 and rsi_prev3 < rsi_prev2 and sma_slope < 0)
            ):
                signal = "SELL"
                # Apply Maxwell's Demon filter
                delta_vol = data['delta_volume'].iloc[i]
                allow_signal, demon_confidence, demon_reason = demon_filter(
                    rsi, rsi_prev, volume, avg_volume, price_to_vwap, signal, delta_vol
                )
                
                # Additional confirmation for higher win rate
                if allow_signal:
                    # Only enter during increasing volume and sufficient tunneling probability
                    if volume_ratio >= 1.5 and tunnel_prob > min_probability and price_to_vwap > 1.01:
                        # Ensure we're not entering right before support (previous low)
                        recent_low = min(data['Low'].iloc[i-10:i])
                        if (price - recent_low) / price > 0.01:  # At least 1% room to move down
                            # Calculate stop loss and take profit - asymmetric risk-reward
                            sl = price + stop_loss_atr * current_atr
                            tp = price - take_profit_atr * current_atr
                            
                            # Enter short position
                            position_type = 'short'
                            in_position = True
                            entry_price = price
                            stop_loss = sl
                            take_profit = tp
                            entry_date = date
                            position_size = 0.2  # Use 20% of equity for position
                            
                            # Save entry RSI for pattern tracking
                            trade_entry_rsi = rsi
                            trade_entry_momentum = rsi_momentum
                            
                            if debug_mode:
                                print(f"SELL SIGNAL: Entered SHORT at {price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}, RSI: {rsi:.1f}")
                                print(f"Reason: RSI {rsi:.1f} with momentum {rsi_momentum:.1f}, Tunneling prob: {tunnel_prob:.2f}")
        
        # Update equity curve
        equity_curve.append(equity)
    
    # Compile results
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['profit_loss'] > 0)
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    total_profit = sum(t['profit_loss'] for t in trades)
    profit_factor = (
        sum(t['profit_loss'] for t in trades if t['profit_loss'] > 0) / 
        abs(sum(t['profit_loss'] for t in trades if t['profit_loss'] < 0))
    ) if sum(t['profit_loss'] for t in trades if t['profit_loss'] < 0) != 0 else float('inf')
    
    # Calculate max drawdown
    peak = equity_curve[0]
    max_drawdown = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Print results
    print("\n----- Backtest Results -----")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Final Equity: ${equity:.2f}")
    print(f"Net Profit: ${equity - initial_capital:.2f} ({(equity / initial_capital - 1) * 100:.2f}%)")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # Create some plots to visualize results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price with buy/sell markers
    ax1.plot(data.index[250:], data['Close'].iloc[250:], label='ETH/USDT')
    
    for trade in trades:
        if trade['type'] == 'long':
            ax1.scatter(trade['entry_date'], trade['entry_price'], marker='^', color='green', s=100)
            ax1.scatter(trade['exit_date'], trade['exit_price'], marker='v', color='red', s=100)
        else:
            ax1.scatter(trade['entry_date'], trade['entry_price'], marker='v', color='red', s=100)
            ax1.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='green', s=100)
    
    ax1.set_title('Quantum RSI Strategy')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot equity curve - Fix the dimension mismatch
    # Ensure x and y have the same length
    if len(equity_curve) > len(data.index) - 250:
        # If equity_curve is longer, trim it to match the available x values
        equity_curve_plot = equity_curve[:len(data.index)-250]
    else:
        # If equity_curve is shorter, use only the available data points for x
        equity_curve_plot = equity_curve
        
    # Plot using the adjusted arrays
    x_vals = data.index[250:250+len(equity_curve_plot)]
    ax2.plot(x_vals, equity_curve_plot)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Equity')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot to file instead of showing it
    plt.savefig(f"static/results/{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_chart.png")
    
    # Compile stats
    stats = {
        'Total Trades': total_trades,
        'Win Rate [%]': win_rate,
        'Net Profit': equity - initial_capital,
        'Return [%]': (equity / initial_capital - 1) * 100,
        'Max Drawdown [%]': max_drawdown,
        'Profit Factor': profit_factor,
        'Final Equity': equity,
        'Trades': trades,
        'Equity Curve': equity_curve  # Add equity curve to return data
    }
    
    return stats

if __name__ == "__main__":
    # Fetch data
    symbol = 'ETH/USDT'
    timeframe = '1h'
    start_date = '2024-01-01'
    end_date = '2024-07-01'
    
    data = fetch_crypto_data(symbol, timeframe, start_date=start_date, end_date=end_date)
    data = calculate_indicators(data)
    
    # Run backtest with relaxed parameters
    params = {
        'initial_capital': 10000,
        'rsi_period': 14,
        'rsi_high': 65,         # Lower threshold
        'rsi_low': 35,          # Higher threshold
        'v0': 1.5,              # Lower barrier
        'a': 1.0,               # Default value
        'mass': 1.0,            # Lower mass
        'hbar': 1.5,            # Higher hbar
        'stop_loss_atr': 1.0,   # ATR multiplier for stop loss
        'take_profit_atr': 2.0, # ATR multiplier for take profit
        'cooldown_period': 6,   # Reduced cooldown
        'debug_mode': True      # Show detailed logs
    }
    
    stats = run_simple_backtest(data, params)
    
    # Save detailed trade log
    trades_df = pd.DataFrame(stats['Trades'])
    if len(trades_df) > 0:
        trades_df.to_csv('trade_log.csv', index=False)
        print("Trade log saved to trade_log.csv") 