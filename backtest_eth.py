"""
Backtest script for ETH/USDT using the Quantum RSI Strategy with Maxwell's Demon Filter

This script runs a backtest on ETH/USDT data with optimized parameters
for the Quantum RSI strategy, with an emphasis on achieving a high win rate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import ccxt
import time

# Add numpy bool8 patch
np.bool8 = np.bool_ if not hasattr(np, 'bool8') else np.bool8
print("NumPy patched to add np.bool8 alias")

# Import quantum RSI functions
from quantum_rsi_optimized import (
    QuantumRSIParameters,
    calculate_indicators,
    compute_barrier_height,
    compute_tunneling_probability,
    maxwell_demon_filter
)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

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
    
    results = {
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'trades': len(trades),
        'equity_curve': equity,
        'trade_log': trade_log
    }
    
    # Save trade log
    if not trade_df.empty:
        trade_df.to_csv(f"results/trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    
    return results

def compare_parameter_settings():
    """
    Run backtest with different parameter settings and compare results
    """
    # Configure parameter sets
    parameter_sets = {
        "Conservative": QuantumRSIParameters(),
        "Balanced": QuantumRSIParameters(),
        "Aggressive": QuantumRSIParameters()
    }
    
    # Set different parameters for each profile
    # Conservative - Highest win rate, fewer trades
    parameter_sets["Conservative"].rsi_high = 78
    parameter_sets["Conservative"].rsi_low = 22
    parameter_sets["Conservative"].volume_threshold = 2.0
    parameter_sets["Conservative"].take_profit_atr = 1.0
    parameter_sets["Conservative"].stop_loss_atr = 1.8
    
    # Balanced - Default parameters
    
    # Aggressive - More trades, potentially lower win rate
    parameter_sets["Aggressive"].rsi_high = 65
    parameter_sets["Aggressive"].rsi_low = 35
    parameter_sets["Aggressive"].volume_threshold = 1.5
    parameter_sets["Aggressive"].take_profit_atr = 1.0
    parameter_sets["Aggressive"].stop_loss_atr = 1.5
    
    # Fetch data
    symbol = "ETH/USDT"
    timeframe = "1h"
    start_date = "2023-01-01"
    end_date = "2023-07-01"
    
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    # Run backtest for each parameter set
    results = {}
    
    for name, params in parameter_sets.items():
        print(f"\nTesting {name} parameters...")
        result = run_backtest(df, params, plot=False)
        results[name] = result
    
    # Compare results
    print("\n=== Parameter Comparison ===")
    print(f"{'Parameter Set':<15} {'Win Rate':<10} {'Total Return':<15} {'Profit Factor':<15} {'Max Drawdown':<15} {'Trades':<10}")
    print("-" * 75)
    
    for name, result in results.items():
        print(f"{name:<15} {result['win_rate']:<10.2f}% {result['total_return']:<15.2f}% {result['profit_factor']:<15.2f} {result['max_drawdown']*100:<15.2f}% {result['trades']:<10}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Win Rates
    plt.subplot(1, 3, 1)
    win_rates = [result['win_rate'] for result in results.values()]
    plt.bar(results.keys(), win_rates)
    plt.title('Win Rates')
    plt.ylabel('Win Rate (%)')
    
    # Returns
    plt.subplot(1, 3, 2)
    returns = [result['total_return'] for result in results.values()]
    plt.bar(results.keys(), returns)
    plt.title('Total Returns')
    plt.ylabel('Return (%)')
    
    # Trade Counts
    plt.subplot(1, 3, 3)
    trade_counts = [result['trades'] for result in results.values()]
    plt.bar(results.keys(), trade_counts)
    plt.title('Number of Trades')
    plt.ylabel('Trades')
    
    plt.tight_layout()
    plt.savefig(f"results/parameter_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    return results

if __name__ == "__main__":
    print("=== Quantum RSI Backtest with Maxwell's Demon Filter ===")
    
    # Run comparison of different parameter settings
    comparison_results = compare_parameter_settings()
    
    # Identify best parameter set based on win rate
    best_win_rate = 0
    best_set = None
    
    for name, result in comparison_results.items():
        if result['win_rate'] > best_win_rate:
            best_win_rate = result['win_rate']
            best_set = name
    
    print(f"\nBest parameter set for high win rate: {best_set} with {best_win_rate:.2f}% win rate") 