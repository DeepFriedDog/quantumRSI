#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantum Stock Analysis Summary

This script reads the merged results JSON file and prints a textual summary
of the most important findings from the backtesting results.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

def load_latest_results(folder_path='improved_Quantum_stock_strategy'):
    """Load the latest merged results JSON file."""
    # Find the most recent merged results file
    json_files = [f for f in os.listdir(folder_path) 
                 if f.startswith('merged_results_') and f.endswith('.json')]
    
    if not json_files:
        # If no merged file exists, try to find the most recent interim file
        json_files = [f for f in os.listdir(folder_path) 
                     if f.startswith('analysis_results_') and f.endswith('_interim.json')]
        
    if not json_files:
        print("No results files found!")
        return None
    
    # Sort by timestamp (newer first)
    json_files.sort(reverse=True)
    latest_file = os.path.join(folder_path, json_files[0])
    
    print(f"Loading results from {latest_file}...")
    
    try:
        with open(latest_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None

def create_analysis_df(results):
    """Convert results to a DataFrame for analysis."""
    if not results:
        return None
        
    rows = []
    for token, timeframe_data in results.items():
        for timeframe, strategy_data in timeframe_data.items():
            if isinstance(strategy_data, dict) and 'error' not in strategy_data:
                for strategy, metrics in strategy_data.items():
                    # Skip entries with errors
                    if 'error' in metrics or not isinstance(metrics, dict):
                        continue
                    
                    # Fix any string values that should be numeric
                    for key in ['win_rate', 'total_return', 'max_drawdown', 'profit_factor']:
                        if key in metrics and isinstance(metrics[key], str):
                            try:
                                # Strip percentage signs if present
                                metrics[key] = float(metrics[key].replace('%', ''))
                            except:
                                pass
                    
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
    
    if not rows:
        print("No valid results for analysis")
        return None
    
    return pd.DataFrame(rows)

def summarize_strategy_performance(df):
    """Print a summary of strategy performance."""
    if df is None or df.empty:
        print("No data available for summarizing strategy performance.")
        return
    
    print("\n" + "="*80)
    print("QUANTUM STOCK STRATEGY PERFORMANCE SUMMARY")
    print("="*80)
    
    # Overall performance by strategy
    strategy_summary = df.groupby('strategy').agg({
        'total_trades': 'sum',
        'win_rate': 'mean',
        'total_return': 'mean',
        'max_drawdown': 'mean',
        'profit_factor': 'mean',
        'risk_adjusted_return': 'mean'
    }).sort_values('total_return', ascending=False)
    
    print("\n1. OVERALL STRATEGY PERFORMANCE (sorted by average return):")
    print("-"*80)
    
    for i, (strategy, row) in enumerate(strategy_summary.iterrows(), 1):
        print(f"{i}. {strategy}")
        print(f"   Win Rate: {row['win_rate']:.1f}%")
        print(f"   Average Return: {row['total_return']:.1f}%")
        print(f"   Average Max Drawdown: {row['max_drawdown']:.1f}%")
        print(f"   Average Profit Factor: {row['profit_factor']:.2f}")
        print(f"   Risk-Adjusted Return: {row['risk_adjusted_return']:.2f}")
        print(f"   Total Trades: {int(row['total_trades'])}")
        print()
    
    # Performance by timeframe
    print("\n2. PERFORMANCE BY TIMEFRAME:")
    print("-"*80)
    
    timeframe_pivot = pd.pivot_table(
        df, 
        values=['win_rate', 'total_return'], 
        index='strategy',
        columns='timeframe', 
        aggfunc='mean'
    )
    
    for timeframe in sorted(df['timeframe'].unique()):
        print(f"\nTimeframe: {timeframe}")
        
        # Sort strategies by return for this timeframe
        tf_results = df[df['timeframe'] == timeframe].groupby('strategy').agg({
            'win_rate': 'mean',
            'total_return': 'mean'
        }).sort_values('total_return', ascending=False)
        
        for i, (strategy, row) in enumerate(tf_results.iterrows(), 1):
            print(f"{i}. {strategy}: {row['total_return']:.1f}% return, {row['win_rate']:.1f}% win rate")
    
    # Best overall combinations
    print("\n3. TOP 10 INDIVIDUAL STRATEGY-TOKEN-TIMEFRAME COMBINATIONS:")
    print("-"*80)
    
    top_combinations = df.sort_values('total_return', ascending=False).head(10)
    
    for i, row in enumerate(top_combinations.itertuples(), 1):
        print(f"{i}. {row.strategy} on {row.token} ({row.timeframe})")
        print(f"   Return: {row.total_return:.1f}%, Win Rate: {row.win_rate:.1f}%, Trades: {row.total_trades}")
    
    # Win rate vs return analysis
    high_win_rate = df[df['win_rate'] > 60].sort_values('win_rate', ascending=False).head(5)
    high_return = df[df['total_return'] > 0].sort_values('total_return', ascending=False).head(5)
    
    print("\n4. HIGHEST WIN RATES (all combinations):")
    print("-"*80)
    
    for i, row in enumerate(high_win_rate.itertuples(), 1):
        print(f"{i}. {row.strategy} on {row.token} ({row.timeframe}): {row.win_rate:.1f}% win rate, {row.total_return:.1f}% return")
    
    print("\n5. HIGHEST RETURNS (all combinations):")
    print("-"*80)
    
    for i, row in enumerate(high_return.itertuples(), 1):
        print(f"{i}. {row.strategy} on {row.token} ({row.timeframe}): {row.total_return:.1f}% return, {row.win_rate:.1f}% win rate")
    
    # Quantum Stock vs Quantum Oscillator comparison
    stock_data = df[df['strategy'].str.contains('Stock')]
    
    print("\n6. QUANTUM STOCK MODEL PERFORMANCE:")
    print("-"*80)
    
    print(f"Average Win Rate: {stock_data['win_rate'].mean():.1f}%")
    print(f"Average Return: {stock_data['total_return'].mean():.1f}%")
    print(f"Average Profit Factor: {stock_data['profit_factor'].mean():.2f}")
    print(f"Best Strategy: {strategy_summary.index[0]} with {strategy_summary.iloc[0]['total_return']:.1f}% return")
    
    print("\n7. ANALYSIS OF IMPROVED MODELS:")
    print("-"*80)
    
    # Compare original vs improved
    original = df[~df['strategy'].str.contains('Improved|Paper')]
    improved = df[df['strategy'].str.contains('Improved')]
    
    print(f"Original Models - Avg Return: {original['total_return'].mean():.1f}%, Win Rate: {original['win_rate'].mean():.1f}%")
    print(f"Improved Models - Avg Return: {improved['total_return'].mean():.1f}%, Win Rate: {improved['win_rate'].mean():.1f}%")
    
    # Conservative vs Balanced vs Aggressive
    conservative = df[df['strategy'].str.contains('Conservative')]
    balanced = df[df['strategy'].str.contains('Balanced')]
    aggressive = df[df['strategy'].str.contains('Aggressive')]
    paper_model = df[df['strategy'].str.contains('Paper')]
    
    print(f"\nConservative Strategies: {conservative['total_return'].mean():.1f}% return, {conservative['win_rate'].mean():.1f}% win rate")
    print(f"Balanced Strategies: {balanced['total_return'].mean():.1f}% return, {balanced['win_rate'].mean():.1f}% win rate")
    print(f"Aggressive Strategies: {aggressive['total_return'].mean():.1f}% return, {aggressive['win_rate'].mean():.1f}% win rate")
    
    if not paper_model.empty:
        print(f"Paper Model Strategy: {paper_model['total_return'].mean():.1f}% return, {paper_model['win_rate'].mean():.1f}% win rate")
    
    print("\n8. KEY FINDINGS & RECOMMENDATIONS:")
    print("-"*80)
    
    # Calculate combined win rate and return metrics
    combined_score = df['win_rate'] * df['total_return']
    best_combined = df.loc[combined_score.idxmax()]
    
    print("Best Combined Win Rate & Return Performance:")
    print(f"Strategy: {best_combined['strategy']}")
    print(f"Token: {best_combined['token']}")
    print(f"Timeframe: {best_combined['timeframe']}")
    print(f"Win Rate: {best_combined['win_rate']:.1f}%")
    print(f"Return: {best_combined['total_return']:.1f}%")
    
    # Recommendations
    best_strat = strategy_summary.index[0]
    best_tf = timeframe_pivot['total_return'].mean(axis=0).idxmax()
    
    print("\nRECOMMENDATIONS:")
    print(f"1. Best Overall Strategy: {best_strat}")
    print(f"2. Best Timeframe: {best_tf}")
    
    # Identify potential for hybrid approach
    high_win_rate_strat = df.groupby('strategy')['win_rate'].mean().idxmax()
    high_return_strat = df.groupby('strategy')['total_return'].mean().idxmax()
    
    if high_win_rate_strat != high_return_strat:
        print(f"3. Potential for Hybrid Approach: Combine {high_win_rate_strat} (high win rate) with {high_return_strat} (high return)")

if __name__ == "__main__":
    results = load_latest_results()
    
    if results:
        df = create_analysis_df(results)
        summarize_strategy_performance(df)
    else:
        print("No results available for analysis.") 