import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

def load_all_results(results_dir="improved_quantum_oscillator_strategy"):
    """Load all non-interim results from the directory."""
    result_files = sorted([
        f for f in os.listdir(results_dir) 
        if f.startswith("analysis_results") and not f.endswith("interim.json")
    ], reverse=True)
    
    if not result_files:
        print("No result files found!")
        return None
    
    # Load all result files and combine them
    all_results = {}
    for filename in result_files:
        results_file = os.path.join(results_dir, filename)
        print(f"Loading results from {results_file}")
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
                
            # Merge results from this file
            for token, timeframe_data in results.items():
                if token not in all_results:
                    all_results[token] = {}
                
                for timeframe, strategy_data in timeframe_data.items():
                    if timeframe not in all_results[token]:
                        all_results[token][timeframe] = {}
                    
                    # Only add if not already present (prioritize more recent files)
                    for strategy, metrics in strategy_data.items():
                        if strategy not in all_results[token][timeframe]:
                            all_results[token][timeframe][strategy] = metrics
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    return all_results

def results_to_dataframe(results):
    """Convert results dictionary to a pandas DataFrame."""
    if not results:
        return None
        
    rows = []
    for token, timeframe_data in results.items():
        for timeframe, strategy_data in timeframe_data.items():
            if isinstance(strategy_data, dict) and 'error' not in strategy_data:
                for strategy, metrics in strategy_data.items():
                    # Set strategy_type based on strategy name
                    if "Classic" in strategy or "Original" in strategy:
                        strategy_type = "Classic"
                    elif "Conservative" in strategy:
                        strategy_type = "Conservative"
                    elif "Aggressive" in strategy:
                        strategy_type = "Aggressive"
                    elif "Balanced" in strategy:
                        strategy_type = "Balanced"
                    else:
                        strategy_type = "Other"
                    
                    # Determine model type from strategy name
                    if "Quantum" not in strategy:
                        model = "Basic"
                    elif "Oscillator" in strategy:
                        model = "Quantum Oscillator"
                    elif "Stock" in strategy:
                        model = "Quantum Stock"
                    elif "Anharmonic" in strategy:
                        model = "Quantum Anharmonic"
                    elif "Prob" in strategy:
                        model = "Quantum Prob"
                    elif "Tunneling" in strategy:
                        model = "Quantum Tunneling"
                    else:
                        model = "Other Quantum"
                    
                    # Extract if it's improved version
                    improved = "Improved" in strategy
                    
                    # Add row
                    rows.append({
                        'Token': token,
                        'Timeframe': timeframe,
                        'Strategy': strategy,
                        'Strategy Type': strategy_type,
                        'Model': model,
                        'Improved': improved,
                        'Total Trades': metrics.get('total_trades', 0),
                        'Win Rate [%]': metrics.get('win_rate', 0),
                        'Return [%]': metrics.get('total_return', 0),
                        'Max Drawdown [%]': metrics.get('max_drawdown', 0),
                        'Profit Factor': metrics.get('profit_factor', 0),
                        'Risk/Reward': metrics.get('risk_adjusted_return', 0)
                    })
    
    return pd.DataFrame(rows) if rows else None

def create_heatmap_visualizations(df, output_dir):
    """Create heatmap visualizations for strategy and timeframe performance."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Filter for strategies with enough data points
    strategy_counts = df.groupby('Strategy').size()
    valid_strategies = strategy_counts[strategy_counts >= 5].index
    df_filtered = df[df['Strategy'].isin(valid_strategies)]
    
    # Create multi-panel figure for strategy performance
    plt.figure(figsize=(20, 8))
    
    # 1. Heatmap of win rates by strategy and timeframe
    ax1 = plt.subplot(1, 2, 1)
    win_rate_pivot = df_filtered.pivot_table(
        values='Win Rate [%]', 
        index='Strategy', 
        columns='Timeframe', 
        aggfunc='mean'
    )
    
    # Sort timeframes
    if set(['15m', '1h', '4h', '1d']).issubset(win_rate_pivot.columns):
        win_rate_pivot = win_rate_pivot[['15m', '1h', '4h', '1d']]
    
    # Create custom colormap for win rates (centered around 50%)
    green = plt.cm.RdYlGn(0.7)
    yellow = plt.cm.RdYlGn(0.5)
    red = plt.cm.RdYlGn(0.2)
    win_rate_cmap = LinearSegmentedColormap.from_list(
        'winrate_cmap', 
        [(0, red), (0.5, yellow), (1, green)], 
        N=100
    )
    
    sns.heatmap(
        win_rate_pivot, 
        annot=True, 
        cmap=win_rate_cmap, 
        fmt='.1f', 
        ax=ax1,
        vmin=20,
        vmax=60
    )
    ax1.set_title('Average Win Rate (%) by Strategy and Timeframe', fontsize=14)
    
    # 2. Heatmap of returns by strategy and timeframe
    ax2 = plt.subplot(1, 2, 2)
    return_pivot = df_filtered.pivot_table(
        values='Return [%]', 
        index='Strategy', 
        columns='Timeframe', 
        aggfunc='mean'
    )
    
    # Sort timeframes
    if set(['15m', '1h', '4h', '1d']).issubset(return_pivot.columns):
        return_pivot = return_pivot[['15m', '1h', '4h', '1d']]
    
    # Create custom colormap for returns (centered around 0%)
    return_cmap = LinearSegmentedColormap.from_list(
        'return_cmap', 
        [(0, plt.cm.RdYlGn(0.2)), (0.5, plt.cm.RdYlGn(0.5)), (1, plt.cm.RdYlGn(0.8))], 
        N=100
    )
    
    sns.heatmap(
        return_pivot, 
        annot=True, 
        cmap=return_cmap, 
        fmt='.1f', 
        ax=ax2,
        vmin=-12,
        vmax=20,
        center=0
    )
    ax2.set_title('Average Return (%) by Strategy and Timeframe', fontsize=14)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"strategy_timeframe_heatmaps_{timestamp}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def create_model_comparison_plots(df, output_dir):
    """Create box plots and bar charts showing model performance comparisons."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create multi-panel figure
    plt.figure(figsize=(20, 16))
    
    # 1. Box plot of win rates by model and strategy type
    ax1 = plt.subplot(2, 2, 1)
    sns.boxplot(
        x='Model', 
        y='Win Rate [%]', 
        hue='Strategy Type',
        data=df,
        palette='viridis',
        ax=ax1
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title('Win Rate (%) by Model and Strategy Type', fontsize=14)
    ax1.legend(title='Strategy Type', loc='upper right')
    
    # 2. Box plot of returns by model and strategy type
    ax2 = plt.subplot(2, 2, 2)
    sns.boxplot(
        x='Model', 
        y='Return [%]', 
        hue='Strategy Type',
        data=df,
        palette='viridis',
        ax=ax2
    )
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_title('Return (%) by Model and Strategy Type', fontsize=14)
    ax2.legend(title='Strategy Type', loc='upper right')
    
    # 3. Bar plot of average return by model
    ax3 = plt.subplot(2, 2, 3)
    model_returns = df.groupby('Model')['Return [%]'].mean().sort_values(ascending=False)
    sns.barplot(
        x=model_returns.index,
        y=model_returns.values,
        palette='viridis',
        ax=ax3
    )
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.set_title('Average Return (%) by Model', fontsize=14)
    ax3.set_ylabel('Return (%)')
    
    # Add value labels to bars
    for i, v in enumerate(model_returns.values):
        ax3.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha='center')
    
    # 4. Bar plot of average win rate by model
    ax4 = plt.subplot(2, 2, 4)
    model_win_rates = df.groupby('Model')['Win Rate [%]'].mean().sort_values(ascending=False)
    sns.barplot(
        x=model_win_rates.index,
        y=model_win_rates.values,
        palette='viridis',
        ax=ax4
    )
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    ax4.set_title('Average Win Rate (%) by Model', fontsize=14)
    ax4.set_ylabel('Win Rate (%)')
    
    # Add value labels to bars
    for i, v in enumerate(model_win_rates.values):
        ax4.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"model_comparison_plots_{timestamp}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def create_timeframe_comparison_plots(df, output_dir):
    """Create timeframe comparison visualizations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    plt.figure(figsize=(15, 10))
    
    # 1. Box plot of win rates by timeframe
    ax1 = plt.subplot(2, 2, 1)
    sns.boxplot(
        x='Timeframe', 
        y='Win Rate [%]', 
        data=df,
        palette='viridis',
        order=['15m', '1h', '4h', '1d'] if set(['15m', '1h', '4h', '1d']).issubset(df['Timeframe'].unique()) else None,
        ax=ax1
    )
    ax1.set_title('Win Rate (%) by Timeframe', fontsize=14)
    
    # 2. Box plot of returns by timeframe
    ax2 = plt.subplot(2, 2, 2)
    sns.boxplot(
        x='Timeframe', 
        y='Return [%]', 
        data=df,
        palette='viridis',
        order=['15m', '1h', '4h', '1d'] if set(['15m', '1h', '4h', '1d']).issubset(df['Timeframe'].unique()) else None,
        ax=ax2
    )
    ax2.set_title('Return (%) by Timeframe', fontsize=14)
    
    # 3. Bar plot of average return by timeframe
    ax3 = plt.subplot(2, 2, 3)
    
    # Define the order if all timeframes exist
    timeframe_order = None
    if set(['15m', '1h', '4h', '1d']).issubset(df['Timeframe'].unique()):
        timeframe_order = ['15m', '1h', '4h', '1d']
    
    timeframe_returns = df.groupby('Timeframe')['Return [%]'].mean()
    if timeframe_order:
        timeframe_returns = timeframe_returns.reindex(timeframe_order)
    
    sns.barplot(
        x=timeframe_returns.index,
        y=timeframe_returns.values,
        palette='viridis',
        ax=ax3
    )
    ax3.set_title('Average Return (%) by Timeframe', fontsize=14)
    ax3.set_ylabel('Return (%)')
    
    # Add value labels to bars
    for i, v in enumerate(timeframe_returns.values):
        ax3.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha='center')
    
    # 4. Bar plot of average win rate by timeframe
    ax4 = plt.subplot(2, 2, 4)
    timeframe_win_rates = df.groupby('Timeframe')['Win Rate [%]'].mean()
    if timeframe_order:
        timeframe_win_rates = timeframe_win_rates.reindex(timeframe_order)
    
    sns.barplot(
        x=timeframe_win_rates.index,
        y=timeframe_win_rates.values,
        palette='viridis',
        ax=ax4
    )
    ax4.set_title('Average Win Rate (%) by Timeframe', fontsize=14)
    ax4.set_ylabel('Win Rate (%)')
    
    # Add value labels to bars
    for i, v in enumerate(timeframe_win_rates.values):
        ax4.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"timeframe_comparison_plots_{timestamp}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def create_scatter_plot(df, output_dir):
    """Create scatter plot of win rate vs return."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot of win rate vs return with strategy
    sns.scatterplot(
        x='Win Rate [%]', 
        y='Return [%]', 
        hue='Model',
        size='Total Trades',
        sizes=(20, 200),
        alpha=0.7,
        data=df
    )
    
    plt.title('Win Rate vs Return (size = # of trades)', fontsize=14)
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Return (%)')
    plt.legend(title='Model', loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add reference lines
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=50, color='r', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"win_rate_return_scatter_{timestamp}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_table(df, output_dir):
    """Create a comprehensive summary table of all strategies."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Calculate summary stats by strategy
    strategy_summary = df.groupby('Strategy').agg({
        'Total Trades': 'sum',
        'Win Rate [%]': 'mean',
        'Return [%]': 'mean',
        'Max Drawdown [%]': 'mean',
        'Profit Factor': 'mean',
        'Risk/Reward': 'mean'
    }).sort_values('Return [%]', ascending=False)
    
    # Add count of profitable tests
    profitable_tests = df[df['Return [%]'] > 0].groupby('Strategy').size()
    strategy_summary['Profitable Tests'] = profitable_tests
    
    total_tests = df.groupby('Strategy').size()
    strategy_summary['Total Tests'] = total_tests
    
    strategy_summary['Profitable %'] = (profitable_tests / total_tests * 100).round(1)
    
    # Create figure for the table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Format the table data
    table_data = []
    for idx, row in strategy_summary.iterrows():
        table_data.append([
            idx,
            f"{row['Win Rate [%]']:.1f}%",
            f"{row['Return [%]']:.1f}%",
            f"{row['Max Drawdown [%]']:.1f}%",
            f"{row['Profit Factor']:.2f}",
            f"{row['Risk/Reward']:.2f}",
            f"{row['Profitable Tests']}/{row['Total Tests']} ({row['Profitable %']}%)",
            f"{row['Total Trades']}"
        ])
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=['Strategy', 'Avg Win Rate', 'Avg Return', 'Avg Max DD', 'Profit Factor', 'Risk/Reward', 'Profitable Tests', 'Total Trades'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color code the cells based on return
    for i in range(len(table_data)):
        # Get return value
        return_str = table_data[i][2].replace('%', '')
        return_val = float(return_str)
        
        # Choose color based on return
        if return_val > 10:
            cell_color = (0.1, 0.6, 0.1)  # Dark green
        elif return_val > 5:
            cell_color = (0.2, 0.8, 0.2)  # Green
        elif return_val > 0:
            cell_color = (0.8, 0.8, 0.2)  # Yellow
        else:
            cell_color = (0.8, 0.2, 0.2)  # Red
        
        # Apply the color to the strategy name cell
        table[(i+1, 0)].set_facecolor(cell_color)
    
    plt.title('Strategy Performance Summary', fontsize=16)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"strategy_performance_summary_{timestamp}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save as CSV for further analysis
    strategy_summary.to_csv(os.path.join(output_dir, f"strategy_performance_summary_{timestamp}.csv"))

def create_distribution_plots(df, output_dir):
    """Create distribution plots of returns and win rates."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    plt.figure(figsize=(20, 10))
    
    # 1. Distribution of returns by strategy
    ax1 = plt.subplot(1, 2, 1)
    
    # Filter for strategies with reasonable amount of data
    strategy_counts = df.groupby('Strategy').size()
    valid_strategies = strategy_counts[strategy_counts >= 5].index
    filtered_df = df[df['Strategy'].isin(valid_strategies)]
    
    sns.boxplot(
        x='Strategy', 
        y='Return [%]', 
        data=filtered_df,
        ax=ax1
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title('Distribution of Returns (%) by Strategy', fontsize=14)
    ax1.set_ylabel('Return (%)')
    
    # 2. Distribution of win rates by strategy
    ax2 = plt.subplot(1, 2, 2)
    sns.boxplot(
        x='Strategy', 
        y='Win Rate [%]', 
        data=filtered_df,
        ax=ax2
    )
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_title('Distribution of Win Rates (%) by Strategy', fontsize=14)
    ax2.set_ylabel('Win Rate (%)')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"strategy_distributions_{timestamp}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def create_risk_adjusted_return_plot(df, output_dir):
    """Create risk-adjusted return visualization."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    plt.figure(figsize=(12, 8))
    
    # Average risk-adjusted return by strategy
    risk_adj_return = df.groupby('Strategy')['Risk/Reward'].mean().sort_values(ascending=False)
    
    # Calculate standard errors for error bars
    risk_adj_std_err = df.groupby('Strategy')['Risk/Reward'].sem()
    
    # Create bar chart with error bars
    ax = sns.barplot(
        x=risk_adj_return.index,
        y=risk_adj_return.values,
        yerr=risk_adj_std_err[risk_adj_return.index],
        palette='viridis'
    )
    
    # Customize appearance
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Average Risk-Adjusted Return (Return/Drawdown)', fontsize=14)
    ax.set_ylabel('Return/Drawdown Ratio')
    
    # Add data labels
    for i, v in enumerate(risk_adj_return.values):
        ax.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"risk_adjusted_return_{timestamp}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main function."""
    # Create output directory
    output_dir = "quantum_oscillator_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    results = load_all_results()
    if not results:
        print("No results to analyze!")
        return
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    if df is None or df.empty:
        print("Failed to convert results to DataFrame!")
        return
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print(f"Total number of tests: {len(df)}")
    print(f"Number of unique tokens: {df['Token'].nunique()}")
    print(f"Number of unique strategies: {df['Strategy'].nunique()}")
    print(f"Average win rate: {df['Win Rate [%]'].mean():.2f}%")
    print(f"Average return: {df['Return [%]'].mean():.2f}%")
    
    # Create all visualizations
    print("\nCreating visualizations...")
    create_heatmap_visualizations(df, output_dir)
    create_model_comparison_plots(df, output_dir)
    create_timeframe_comparison_plots(df, output_dir)
    create_scatter_plot(df, output_dir)
    create_summary_table(df, output_dir)
    create_distribution_plots(df, output_dir)
    create_risk_adjusted_return_plot(df, output_dir)
    
    print(f"\nAnalysis completed! All visualizations saved to {output_dir} directory.")

if __name__ == "__main__":
    main() 