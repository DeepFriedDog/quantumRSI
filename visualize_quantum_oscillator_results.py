import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Import visualization functions from token_strategy_analysis.py
from token_strategy_analysis import (
    create_visualizations,
    create_token_comparison,
    extract_key_metrics
)

def load_results(results_file):
    """Load results from JSON file."""
    print(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def main():
    """Main function to visualize quantum oscillator testing results."""
    # Create output directory if it doesn't exist
    output_dir = "quantum_oscillator_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the most recent results file (non-interim)
    results_dir = "improved_quantum_oscillator_strategy"
    result_files = sorted([
        f for f in os.listdir(results_dir) 
        if f.startswith("analysis_results") and not f.endswith("interim.json")
    ], reverse=True)
    
    if not result_files:
        print("No result files found!")
        return
    
    # Use the most recent result file
    results_file = os.path.join(results_dir, result_files[0])
    print(f"Using results file: {results_file}")
    
    # Load the results
    results = load_results(results_file)
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Generate comprehensive visualizations
    output_file = os.path.join(output_dir, f"comprehensive_analysis_{timestamp}.png")
    create_visualizations(results, output_file)
    
    # Generate token comparison visualization
    token_output_file = os.path.join(output_dir, f"token_comparison_{timestamp}.png")
    
    # Convert results to DataFrame for token comparison
    rows = []
    for token, timeframe_data in results.items():
        for timeframe, strategy_data in timeframe_data.items():
            if isinstance(strategy_data, dict) and 'error' not in strategy_data:
                for strategy, metrics in strategy_data.items():
                    rows.append({
                        'token': token,
                        'timeframe': timeframe,
                        'strategy': strategy,
                        'total_trades': metrics.get('total_trades', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'total_return': metrics.get('total_return', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'profit_factor': metrics.get('profit_factor', 0),
                        'risk_adjusted_return': metrics.get('risk_adjusted_return', 0)
                    })
    
    if rows:
        df = pd.DataFrame(rows)
        create_token_comparison(df, token_output_file)
    
    # Generate market cap analysis
    market_cap_analysis(results, output_dir, timestamp)
    
    print(f"Visualizations generated in {output_dir} directory")

def market_cap_analysis(results, output_dir, timestamp):
    """Create analysis based on market cap categories."""
    # Define market cap categories
    high_market_cap = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT', 'XRP/USDT']
    medium_market_cap = ['LINK/USDT', 'DOT/USDT', 'AVAX/USDT', 'ATOM/USDT', 'NEAR/USDT', 'MATIC/USDT', 'UNI/USDT', 'AAVE/USDT']
    small_market_cap = ['ALGO/USDT', 'RNDR/USDT', 'FTM/USDT', 'ONE/USDT', 'ZIL/USDT', 'ROSE/USDT', 'CHZ/USDT', 'ENJ/USDT', 'OCEAN/USDT', 'API3/USDT', 'KSM/USDT', 'ZRX/USDT']
    
    # Convert results to DataFrame
    rows = []
    for token, timeframe_data in results.items():
        # Determine market cap category
        if token in high_market_cap:
            market_cap = 'High Market Cap'
        elif token in medium_market_cap:
            market_cap = 'Medium Market Cap'
        elif token in small_market_cap:
            market_cap = 'Small Market Cap'
        else:
            market_cap = 'Unknown'
            
        for timeframe, strategy_data in timeframe_data.items():
            if isinstance(strategy_data, dict) and 'error' not in strategy_data:
                for strategy, metrics in strategy_data.items():
                    rows.append({
                        'Token': token,
                        'Market Cap': market_cap,
                        'Timeframe': timeframe,
                        'Strategy': strategy,
                        'Total Trades': metrics.get('total_trades', 0),
                        'Win Rate [%]': metrics.get('win_rate', 0),
                        'Return [%]': metrics.get('total_return', 0),
                        'Max Drawdown [%]': metrics.get('max_drawdown', 0),
                        'Profit Factor': metrics.get('profit_factor', 0),
                        'Risk/Reward': metrics.get('risk_adjusted_return', 0)
                    })
    
    if not rows:
        print("No valid data for market cap analysis")
        return
        
    df = pd.DataFrame(rows)
    
    # Create market cap summary
    numeric_columns = ['Total Trades', 'Win Rate [%]', 'Return [%]', 'Max Drawdown [%]', 'Profit Factor']
    market_cap_summary = df.groupby(['Market Cap', 'Strategy'])[numeric_columns].mean().reset_index()
    
    # Create summary table visualization
    plt.figure(figsize=(14, 10))
    
    # Create a table for each market cap category
    for i, market_cap in enumerate(['High Market Cap', 'Medium Market Cap', 'Small Market Cap']):
        if market_cap not in market_cap_summary['Market Cap'].values:
            continue
            
        data = market_cap_summary[market_cap_summary['Market Cap'] == market_cap].sort_values('Return [%]', ascending=False)
        
        # Format the data for the table
        table_data = []
        for _, row in data.iterrows():
            table_data.append([
                row['Strategy'],
                f"{row['Win Rate [%]']:.1f}%",
                f"{row['Return [%]']:.1f}%",
                f"{row['Max Drawdown [%]']:.1f}%",
                f"{row['Profit Factor']:.2f}",
                f"{row['Total Trades']:.0f}"
            ])
        
        ax = plt.subplot(3, 1, i+1)
        table = ax.table(
            cellText=table_data,
            colLabels=['Strategy', 'Win Rate', 'Return', 'Max Drawdown', 'Profit Factor', 'Total Trades'],
            loc='center',
            cellLoc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        
        # Color code the cells based on return
        for j in range(len(table_data)):
            return_val = float(table_data[j][2].replace('%', ''))
            if return_val > 10:
                cell_color = (0.1, 0.6, 0.1)  # Dark green
            elif return_val > 5:
                cell_color = (0.2, 0.8, 0.2)  # Green
            elif return_val > 0:
                cell_color = (0.8, 0.8, 0.2)  # Yellow
            else:
                cell_color = (0.8, 0.2, 0.2)  # Red
            
            # Apply the color to the strategy name cell
            table[(j+1, 0)].set_facecolor(cell_color)
        
        ax.set_title(f'{market_cap} Performance Summary', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"market_cap_analysis_{timestamp}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create timeframe performance by market cap
    plt.figure(figsize=(15, 10))
    
    # 1. Heatmap of win rates by market cap and timeframe
    ax1 = plt.subplot(2, 2, 1)
    win_rate_pivot = df.pivot_table(
        values='Win Rate [%]', 
        index='Market Cap', 
        columns='Timeframe', 
        aggfunc='mean'
    )
    
    # Sort columns by timeframe
    if '15m' in win_rate_pivot.columns and '1h' in win_rate_pivot.columns and '4h' in win_rate_pivot.columns and '1d' in win_rate_pivot.columns:
        win_rate_pivot = win_rate_pivot[['15m', '1h', '4h', '1d']]
    
    import seaborn as sns
    sns.heatmap(win_rate_pivot, annot=True, cmap='RdYlGn', fmt='.1f', ax=ax1)
    ax1.set_title('Average Win Rate (%) by Market Cap and Timeframe', fontsize=12)
    
    # 2. Heatmap of returns by market cap and timeframe
    ax2 = plt.subplot(2, 2, 2)
    return_pivot = df.pivot_table(
        values='Return [%]', 
        index='Market Cap', 
        columns='Timeframe', 
        aggfunc='mean'
    )
    
    # Sort columns by timeframe
    if '15m' in return_pivot.columns and '1h' in return_pivot.columns and '4h' in return_pivot.columns and '1d' in return_pivot.columns:
        return_pivot = return_pivot[['15m', '1h', '4h', '1d']]
    
    sns.heatmap(return_pivot, annot=True, cmap='RdYlGn', fmt='.1f', ax=ax2)
    ax2.set_title('Average Return (%) by Market Cap and Timeframe', fontsize=12)
    
    # 3. Box plot of returns by market cap
    ax3 = plt.subplot(2, 2, 3)
    sns.boxplot(x='Market Cap', y='Return [%]', data=df, ax=ax3)
    ax3.set_title('Distribution of Returns (%) by Market Cap', fontsize=12)
    ax3.set_ylabel('Return (%)')
    
    # 4. Box plot of win rates by market cap
    ax4 = plt.subplot(2, 2, 4)
    sns.boxplot(x='Market Cap', y='Win Rate [%]', data=df, ax=ax4)
    ax4.set_title('Distribution of Win Rates (%) by Market Cap', fontsize=12)
    ax4.set_ylabel('Win Rate (%)')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"market_cap_timeframe_analysis_{timestamp}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 