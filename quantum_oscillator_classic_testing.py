import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import json
import os
import random
import pickle
from datetime import datetime, timedelta
import warnings
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

from app import run_quantum_oscillator_classic_strategy, fetch_crypto_data

# Define DOGE, ADA and similar tokens
FOCUS_TOKENS = [
    'DOGE/USDT',  # Meme coin with high volatility
    'ADA/USDT',   # Smart contract platform with strong retail following
    'SHIB/USDT',  # Similar to DOGE (meme coin)
    'XRP/USDT',   # Strong retail sentiment-driven token
    'DOT/USDT',   # Smart contract platform like ADA
    'MATIC/USDT', # Layer-2 solution with similar market dynamics
    'ATOM/USDT',  # Interoperability token with similar volatility profile
    'AVAX/USDT',  # Smart contract platform like ADA
    'ALGO/USDT',  # Smart contract platform with similar positioning
    'NEAR/USDT',  # Similar target demographic to ADA
    'FTM/USDT',   # Smart contract platform with similar volatility
    'ONE/USDT',   # Similar market behavior to ADA during rallies
    'SOL/USDT',   # Smart contract platform with similar market dynamics
    'HBAR/USDT'   # Alternative consensus protocol like ADA
]

# Add a data caching directory constant
DATA_CACHE_DIR = 'data_cache'

def select_focus_tokens(num_tokens=10):
    """
    Select DOGE, ADA and similar tokens.
    Always includes DOGE and ADA, plus randomly selected similar tokens.
    """
    # Always include DOGE and ADA
    selected = ['DOGE/USDT', 'ADA/USDT']
    
    # Select additional tokens to reach the desired count
    remaining_tokens = [t for t in FOCUS_TOKENS if t not in selected]
    additional_needed = min(num_tokens - len(selected), len(remaining_tokens))
    
    if additional_needed > 0:
        selected.extend(random.sample(remaining_tokens, additional_needed))
    
    return selected

def generate_one_year_period(year=2023):
    """
    Generate a one-year testing period
    """
    # Use a full year of data
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    # Make sure we don't go into the future
    current_date = datetime.now()
    if end_date > current_date:
        end_date = current_date
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def get_cached_data(token, timeframe, start_date, end_date):
    """Get data from cache if available, otherwise fetch and cache it."""
    # Create cache directory if it doesn't exist
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    token_safe = token.replace('/', '_')
    cache_file = os.path.join(DATA_CACHE_DIR, f"{token_safe}_{timeframe}_{start_date}_{end_date}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Using cached data for {token} {timeframe} ({len(data)} candles)")
            return data
        except Exception as e:
            print(f"Error loading cached data: {str(e)}")
    
    # If we get here, we need to fetch the data
    print(f"Fetching {token} {timeframe} data...")
    data = fetch_crypto_data(token, timeframe, start_date, end_date)
    
    # Cache the data for future use
    if data is not None and len(data) > 0:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Cached {len(data)} candles to {cache_file}")
    
    return data

def create_summary_table(results):
    """
    Create a summary table of all testing results
    """
    summary_data = []
    
    for token, stats in results.items():
        if stats:
            summary_data.append({
                'Token': token,
                'Total Trades': stats.get('Total Trades', 0),
                'Win Rate [%]': stats.get('Win Rate [%]', 0),
                'Return [%]': stats.get('Return [%]', 0),
                'Max Drawdown [%]': stats.get('Max Drawdown [%]', 0),
                'Profit Factor': stats.get('Profit Factor', 0),
                'Risk/Reward': stats.get('Return [%]', 0) / stats.get('Max Drawdown [%]', 1) if stats.get('Max Drawdown [%]', 0) > 0 and stats.get('Return [%]', 0) > 0 else 0
            })
    
    return pd.DataFrame(summary_data)

def create_comprehensive_visualizations(results, output_file, period_str):
    """Create comprehensive visualizations of the results."""
    # Convert results to a flat DataFrame for easier plotting
    rows = []
    for token, stats in results.items():
        if stats:
            rows.append({
                'token': token,
                'timeframe': '1d',
                'strategy': 'Quantum Oscillator - Classic',
                'total_trades': stats.get('Total Trades', 0),
                'win_rate': stats.get('Win Rate [%]', 0),
                'total_return': stats.get('Return [%]', 0),
                'max_drawdown': stats.get('Max Drawdown [%]', 0),
                'profit_factor': stats.get('Profit Factor', 0),
                'risk_adjusted_return': stats.get('Return [%]', 0) / stats.get('Max Drawdown [%]', 1) if stats.get('Max Drawdown [%]', 0) > 0 and stats.get('Return [%]', 0) > 0 else 0
            })
    
    if not rows:
        print("No valid results to visualize")
        return
    
    df = pd.DataFrame(rows)
    
    # Create a multi-panel figure
    plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.5])
    
    # 1. Bar chart of win rates by token
    ax1 = plt.subplot(gs[0, 0])
    sns.barplot(x='token', y='win_rate', data=df, ax=ax1)
    ax1.set_title(f'Win Rate (%) by Token - {period_str}', fontsize=14)
    ax1.set_xlabel('')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Bar chart of returns by token
    ax2 = plt.subplot(gs[0, 1])
    sns.barplot(x='token', y='total_return', data=df, ax=ax2)
    ax2.set_title(f'Return (%) by Token - {period_str}', fontsize=14)
    ax2.set_xlabel('')
    ax2.set_ylabel('Return (%)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Scatter plot of win rate vs return
    ax3 = plt.subplot(gs[1, 0])
    sns.scatterplot(
        x='win_rate', 
        y='total_return', 
        size='total_trades',
        sizes=(50, 200),
        alpha=0.7,
        data=df, 
        ax=ax3
    )
    ax3.set_title('Win Rate vs Return (size = # of trades)', fontsize=14)
    ax3.set_xlabel('Win Rate (%)')
    ax3.set_ylabel('Return (%)')
    
    # Add token labels to each point
    for i, row in df.iterrows():
        ax3.annotate(row['token'], (row['win_rate'], row['total_return']), 
                    xytext=(5, 5), textcoords='offset points')
    
    # 4. Bar chart of risk-adjusted returns
    ax4 = plt.subplot(gs[1, 1])
    risk_adj_return = df.sort_values('risk_adjusted_return', ascending=False)
    sns.barplot(x='token', y='risk_adjusted_return', data=risk_adj_return, ax=ax4)
    ax4.set_title('Risk-Adjusted Return (Return/Drawdown)', fontsize=14)
    ax4.set_xlabel('')
    ax4.set_ylabel('Return/Drawdown Ratio')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # 5. Bar chart of profit factors
    ax5 = plt.subplot(gs[2, 0])
    profit_factor_df = df.sort_values('profit_factor', ascending=False)
    sns.barplot(x='token', y='profit_factor', data=profit_factor_df, ax=ax5)
    ax5.set_title('Profit Factor by Token', fontsize=14)
    ax5.set_xlabel('')
    ax5.set_ylabel('Profit Factor')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    
    # 6. Bar chart of drawdowns
    ax6 = plt.subplot(gs[2, 1])
    drawdown_df = df.sort_values('max_drawdown')
    sns.barplot(x='token', y='max_drawdown', data=drawdown_df, ax=ax6)
    ax6.set_title('Maximum Drawdown (%) by Token', fontsize=14)
    ax6.set_xlabel('')
    ax6.set_ylabel('Max Drawdown (%)')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
    
    # 7. Summary table
    ax7 = plt.subplot(gs[3, :])
    ax7.axis('off')
    
    # Calculate average metrics
    avg_trades = df['total_trades'].mean()
    avg_win_rate = df['win_rate'].mean()
    avg_return = df['total_return'].mean()
    avg_drawdown = df['max_drawdown'].mean()
    avg_profit_factor = df['profit_factor'].mean()
    profitable_count = len(df[df['total_return'] > 0])
    
    # Format table data
    table_data = [
        ['Quantum Oscillator - Classic', f"{avg_win_rate:.1f}%", f"{avg_return:.1f}%", 
         f"{avg_drawdown:.1f}%", f"{avg_profit_factor:.2f}", 
         f"{profitable_count}/{len(df)} ({profitable_count/len(df)*100:.1f}%)", f"{int(avg_trades)}"]
    ]
    
    # Create header row with color
    header = ['Strategy', 'Avg Win Rate', 'Avg Return', 'Avg Max DD', 'Profit Factor', 'Profitable Tests', 'Total Trades']
    
    # Create the table
    table = ax7.table(
        cellText=table_data,
        colLabels=header,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color code based on return
    if avg_return > 10:
        cell_color = (0.1, 0.6, 0.1)  # Dark green
    elif avg_return > 5:
        cell_color = (0.2, 0.8, 0.2)  # Green
    elif avg_return > 0:
        cell_color = (0.8, 0.8, 0.2)  # Yellow
    else:
        cell_color = (0.8, 0.2, 0.2)  # Red
    
    # Apply color to strategy cell
    table[(1, 0)].set_facecolor(cell_color)
    
    ax7.set_title(f'Strategy Performance Summary - {period_str}', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()

def run_test_for_token(token, start_date, end_date, timeframe='1d'):
    """
    Run test for a single token on specified timeframe
    """
    print(f"Testing {token} on {timeframe} timeframe for period {start_date} to {end_date}")
    
    try:
        # Fetch data with caching
        df = get_cached_data(token, timeframe, start_date, end_date)
        if df is None or len(df) < 20:  # Need minimum data points
            print(f"⚠️ Insufficient data for {token}, skipping")
            return None
        
        print(f"✓ Data fetched: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        # Run strategy
        stats = run_quantum_oscillator_classic_strategy(df, token=token)
        
        # Extract and print key metrics
        print(f"Total Return: {stats.get('Return [%]', 0):.2f}%")
        print(f"Total Trades: {stats.get('Total Trades', 0)}")
        print(f"Win Rate: {stats.get('Win Rate [%]', 0):.2f}%")
        print(f"Profit Factor: {stats.get('Profit Factor', 0):.2f}")
        print(f"Max Drawdown: {stats.get('Max Drawdown [%]', 0):.2f}%")
        print("-" * 50)
        
        return stats
    
    except Exception as e:
        print(f"❌ Error testing {token}: {str(e)}")
        return None

def main():
    # Create output directory
    output_dir = "quantum_oscillator_classic_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a one-year period (for 2023 by default)
    start_date, end_date = generate_one_year_period(2023)
    period_str = f"Period: {start_date} to {end_date}"
    print(f"Test period: {start_date} to {end_date}")
    
    # Select DOGE, ADA and similar tokens
    tokens = select_focus_tokens(num_tokens=10)
    print(f"Selected tokens: {tokens}")
    print("-" * 50)
    
    # Run tests
    results = {}
    for token in tokens:
        stats = run_test_for_token(token, start_date, end_date, timeframe='1d')
        results[token] = stats
    
    # Create summary table
    summary_df = create_summary_table(results)
    
    # Save summary to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_df.to_csv(f"{output_dir}/summary_{timestamp}.csv", index=False)
    
    # Create comprehensive visualizations
    if not summary_df.empty:
        output_file = f"{output_dir}/comprehensive_analysis_{timestamp}.png"
        create_comprehensive_visualizations(results, output_file, period_str)
    else:
        print("No results to visualize.")
    
    # Print summary report
    if not summary_df.empty:
        print("\n===== TESTING SUMMARY =====")
        print("\nAverage metrics:")
        
        # Calculate means for numeric columns
        numeric_cols = ['Total Trades', 'Win Rate [%]', 'Return [%]', 'Max Drawdown [%]', 'Profit Factor', 'Risk/Reward']
        means = summary_df[numeric_cols].mean()
        print(means)
        
        print("\nTop performers by return:")
        top_return = summary_df.sort_values('Return [%]', ascending=False).head(3)
        print(top_return[['Token', 'Return [%]', 'Win Rate [%]', 'Profit Factor']])
        
        print("\nTop performers by win rate:")
        top_winrate = summary_df.sort_values('Win Rate [%]', ascending=False).head(3)
        print(top_winrate[['Token', 'Win Rate [%]', 'Return [%]', 'Profit Factor']])
    else:
        print("\nNo results to summarize.")
    
    # Save results to JSON
    serializable_results = {}
    for token, stats in results.items():
        if stats:
            # Convert numpy types to Python native types for JSON serialization
            serializable_results[token] = {
                'Total Trades': int(stats.get('Total Trades', 0)),
                'Win Rate [%]': float(stats.get('Win Rate [%]', 0)),
                'Return [%]': float(stats.get('Return [%]', 0)),
                'Max Drawdown [%]': float(stats.get('Max Drawdown [%]', 0)),
                'Profit Factor': float(stats.get('Profit Factor', 0)),
                'Risk/Reward': float(stats.get('Return [%]', 0) / stats.get('Max Drawdown [%]', 1)) if stats.get('Max Drawdown [%]', 0) > 0 and stats.get('Return [%]', 0) > 0 else 0
            }
    
    with open(f"{output_dir}/results_{timestamp}.json", 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/ directory")

if __name__ == "__main__":
    main() 