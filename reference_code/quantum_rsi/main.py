"""
Quantum RSI - Main Module

This is the main entry point for the Quantum RSI trading system.
It provides a command-line interface to run the strategy in live or backtest mode.
"""

import argparse
import datetime
import logging
import os
import sys
import time
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import components
from quantum_rsi.core.quantum_model import (
    QuantumWellPotential, 
    SchrodingerSolver, 
    QuantumTunneling, 
    MaxwellsDemon
)
from quantum_rsi.data.dex_data import DEXDataProvider, check_web3_availability
from quantum_rsi.trading.trader import QuantumRSITrader
from quantum_rsi.backtesting.backtester import QuantumRSIBacktester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_rsi.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('quantum_rsi.main')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Quantum RSI Trading System')
    
    # Mode selection
    parser.add_argument('--mode', choices=['live', 'backtest', 'analysis', 'visualize'], default='analysis',
                      help='Operation mode: live trading, backtest, analysis (read-only), or visualize results')
    
    # Common parameters
    parser.add_argument('--pairs', type=str, default='ETH/USDT,ETH/USDC',
                      help='Comma-separated list of trading pairs')
    parser.add_argument('--dex', type=str, default='uniswap',
                      help='DEX provider to use (uniswap, dydx, 1inch)')
    parser.add_argument('--risk', type=float, default=0.01,
                      help='Risk per trade as fraction of wallet balance (default: 0.01 = 1%)')
    parser.add_argument('--volume-threshold', type=float, default=2.0,
                      help='Volume threshold for Maxwell\'s Demon filter (default: 2.0)')
    parser.add_argument('--vwap-threshold', type=float, default=0.01,
                      help='VWAP deviation threshold for Maxwell\'s Demon filter (default: 0.01)')
    
    # Quantum model parameters
    parser.add_argument('--v0', type=float, default=1.0,
                      help='Quantum potential strength (default: 1.0)')
    parser.add_argument('--a', type=float, default=0.4,
                      help='Well position parameter (default: 0.4)')
    parser.add_argument('--m', type=float, default=1.0,
                      help='Effective mass parameter (default: 1.0)')
    parser.add_argument('--hbar', type=float, default=1.0,
                      help='Reduced Planck constant (default: 1.0)')
    parser.add_argument('--evolution-steps', type=int, default=10,
                      help='Quantum state evolution steps (default: 10)')
    parser.add_argument('--no-demon', action='store_true',
                      help='Disable Maxwell\'s Demon filtering')
    
    # Backtesting parameters
    parser.add_argument('--start-date', type=str, default=None,
                      help='Start date for backtest (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                      help='End date for backtest (format: YYYY-MM-DD)')
    parser.add_argument('--initial-balance', type=float, default=1.0,
                      help='Initial account balance in ETH for backtest (default: 1.0 ETH)')
    parser.add_argument('--save-results', action='store_true',
                      help='Save backtest results to file')
    parser.add_argument('--results-file', type=str, default='quantum_rsi_results.json',
                      help='Filename for saving/loading results')
    parser.add_argument('--show-trades', action='store_true',
                      help='Show individual trades on the visualization')
    
    # Live trading parameters
    parser.add_argument('--check-interval', type=int, default=3600,
                      help='Check interval in seconds for live trading (default: 3600 = 1 hour)')
    
    # New parameters
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()


def configure_logging(verbose=False):
    """Configure logging settings."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure new handlers
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set 3rd party loggers to a higher level to reduce noise
    if not verbose:
        logging.getLogger('web3').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('websockets').setLevel(logging.WARNING)
    
    return logging.getLogger('quantum_rsi.main')


def run_live_trading(args):
    """Run the strategy in live trading mode."""
    logger.info("Starting Quantum RSI in LIVE mode")
    
    # Check if Web3 is available
    if not check_web3_availability():
        logger.error("Web3 provider is not available. Please check your .env configuration.")
        return False
    
    # Prepare pairs list
    pairs = [p.strip() for p in args.pairs.split(',')]
    
    # Prepare quantum parameters
    quantum_params = {
        "v0": args.v0,
        "a": args.a,
        "m": args.m,
        "hbar": args.hbar,
        "evolution_steps": args.evolution_steps
    }
    
    # Create the trader
    trader = QuantumRSITrader(
        pairs=pairs,
        timeframe=os.getenv("TIMEFRAME", "1h"),
        risk_per_trade=args.risk,
        volume_threshold=args.volume_threshold,
        vwap_threshold=args.vwap_threshold,
        dex_provider=args.dex,
        quantum_params=quantum_params
    )
    
    # Run the trading loop
    try:
        trader.run_trading_loop(interval=args.check_interval)
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    except Exception as e:
        logger.error(f"Error in trading loop: {e}")
        return False
    
    return True


def run_analysis_mode(args):
    """Run the strategy in analysis mode (read-only, no trading)."""
    logger.info("Starting Quantum RSI in ANALYSIS mode (read-only)")
    
    # Check if Web3 is available
    if not check_web3_availability():
        logger.error("Web3 provider is not available. Please check your .env configuration.")
        return False
    
    # Prepare pairs list
    pairs = [p.strip() for p in args.pairs.split(',')]
    
    # Prepare quantum parameters
    quantum_params = {
        "v0": args.v0,
        "a": args.a,
        "m": args.m,
        "hbar": args.hbar,
        "evolution_steps": args.evolution_steps
    }
    
    # Create the trader (will automatically be in read-only mode if no private key is set)
    trader = QuantumRSITrader(
        pairs=pairs,
        timeframe=os.getenv("TIMEFRAME", "1h"),
        risk_per_trade=args.risk,
        volume_threshold=args.volume_threshold,
        vwap_threshold=args.vwap_threshold,
        dex_provider=args.dex,
        quantum_params=quantum_params
    )
    
    # Run the trading cycle once to get analysis
    try:
        trader.run_trading_cycle()
        
        # Wait for user to exit
        logger.info("Analysis complete. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Analysis mode stopped by user")
    except Exception as e:
        logger.error(f"Error in analysis mode: {e}")
        return False
    
    return True


def run_backtest(args):
    """Run the strategy in backtest mode."""
    logger.info("Starting Quantum RSI in BACKTEST mode")
    
    # Prepare pairs list
    pairs = [p.strip() for p in args.pairs.split(',')]
    
    # Parse dates
    start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    
    # Prepare quantum parameters
    quantum_params = {
        "v0": args.v0,
        "a": args.a,
        "m": args.m,
        "hbar": args.hbar,
        "evolution_steps": args.evolution_steps
    }
    
    # Create the backtester
    backtester = QuantumRSIBacktester(
        pairs=pairs,
        dex_provider=args.dex,
        start_date=start_date,
        end_date=end_date,
        initial_balance=args.initial_balance,
        risk_per_trade=args.risk,
        volume_threshold=args.volume_threshold,
        vwap_threshold=args.vwap_threshold,
        quantum_params=quantum_params,
        include_demon=not args.no_demon
    )
    
    try:
        # Load data and run backtest
        backtester.load_historical_data()
        backtester.run_backtest()
        
        # Print and display results
        backtester.print_results()
        backtester.plot_results(show_trades=args.show_trades)
        
        # Save results if requested
        if args.save_results:
            backtester.save_results(args.results_file)
            logger.info(f"Backtest results saved to {args.results_file}")
    
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        return False
    
    return True


def visualize_results(args):
    """Visualize previously saved backtest results."""
    logger.info("Visualizing saved backtest results")
    
    # Check if results file exists
    if not os.path.exists(args.results_file):
        logger.error(f"Results file not found: {args.results_file}")
        return False
    
    try:
        # Load results from file
        with open(args.results_file, 'r') as f:
            results = json.load(f)
        
        # Print summary
        print("\n=== Quantum RSI Backtest Results ===")
        config = results.get("backtest_config", {})
        metrics = results.get("metrics", {})
        
        print(f"Period: {config.get('start_date')} to {config.get('end_date')}")
        print(f"Pairs: {', '.join(config.get('pairs', []))}")
        print(f"DEX: {config.get('dex_provider', 'uniswap')}")
        print(f"Initial Balance: {config.get('initial_balance', 0):.2f} ETH")
        print(f"Final Equity: {metrics.get('final_equity', 0):.2f} ETH")
        print(f"Net Profit: {metrics.get('net_profit', 0):.2f} ETH ({metrics.get('profit_percentage', 0):.2f}%)")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print("=====================================")
        
        # Extract equity curve
        equity_curve = results.get("equity_curve", [])
        if equity_curve:
            import pandas as pd
            import matplotlib.pyplot as plt
            
            # Convert to DataFrame
            equity_df = pd.DataFrame(equity_curve)
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df.set_index('date', inplace=True)
            
            # Plot equity curve
            plt.figure(figsize=(12, 6))
            equity_df['equity'].plot(linewidth=2)
            plt.title('Quantum RSI Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity (ETH)')
            plt.grid(True, alpha=0.3)
            plt.show()
    
    except Exception as e:
        logger.error(f"Error visualizing results: {e}")
        return False
    
    return True


def display_welcome():
    """Display welcome message and logo."""
    print("\n" + "=" * 80)
    print("""
             ██████  ██    ██  █████  ███    ██ ████████ ██    ██ ███    ███     ██████  ███████ ██ 
            ██    ██ ██    ██ ██   ██ ████   ██    ██    ██    ██ ████  ████     ██   ██ ██      ██ 
            ██    ██ ██    ██ ███████ ██ ██  ██    ██    ██    ██ ██ ████ ██     ██████  ███████ ██ 
            ██ ▄▄ ██ ██    ██ ██   ██ ██  ██ ██    ██    ██    ██ ██  ██  ██     ██   ██      ██ ██ 
             ██████   ██████  ██   ██ ██   ████    ██     ██████  ██      ██     ██   ██ ███████ ██ 
                ▀▀
    """)
    print(f"Quantum RSI Trading System v{__import__('quantum_rsi').__version__}")
    print("A Schrödinger equation approach to trend reversals with Maxwell's Demon filtering")
    print("=" * 80)
    print("CryptoCompare data integration for reliable market data")
    print("=" * 80 + "\n")


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging
    logger = configure_logging(args.verbose)
    
    # Print banner
    display_welcome()
    
    # Debug output
    print(f"\n[DEBUG] Running in {args.mode.upper()} mode with these arguments:")
    for arg, value in vars(args).items():
        print(f"  - {arg}: {value}")
    print("")
    
    # Parse trading pairs
    pairs = [p.strip() for p in args.pairs.split(',')]
    
    # Convert dates if provided
    start_date = None
    if args.start_date:
        start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
        
    end_date = None  
    if args.end_date:
        end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Initialize system based on mode
    logger.info(f"Starting Quantum RSI in {args.mode.upper()} mode")
    print(f"Starting Quantum RSI in {args.mode.upper()} mode")
    print(f"Data source: CryptoCompare (primary) and {args.dex} (secondary)")
    
    if args.mode == 'backtest':
        print(f"Running backtest for {pairs} from {start_date} to {end_date}")
        # Run in backtest mode
        backtester = QuantumRSIBacktester(
            pairs=pairs,
            dex_provider=args.dex,
            initial_balance=args.initial_balance,
            risk_per_trade=args.risk
        )
        
        # Run the backtest
        results = backtester.run_backtest(
            start_date=start_date,
            end_date=end_date,
            show_trades=args.show_trades,
            save_results=args.save_results
        )
        
        # Display results summary
        backtester.display_results()
        
    elif args.mode == 'live':
        # Run in live trading mode
        trader = QuantumRSITrader(
            pairs=pairs,
            risk_per_trade=args.risk,
            dex_provider=args.dex
        )
        
        try:
            trader.start()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            trader.stop()
            
    elif args.mode == 'analysis':
        # Run in analysis mode
        analyzer = QuantumRSIAnalyzer(
            pairs=pairs,
            dex_provider=args.dex
        )
        
        analyzer.run_analysis()
        
    elif args.mode == 'simulation':
        # Run in simulation mode
        simulator = QuantumRSISimulator(
            pairs=pairs,
            dex_provider=args.dex,
            initial_balance=args.initial_balance,
            risk_per_trade=args.risk
        )
        
        simulator.run_simulation(
            start_date=start_date,
            duration_days=30 if not end_date else (end_date - start_date).days,
            show_trades=args.show_trades
        )
    
    print("Quantum RSI session completed.")
    logger.info("Quantum RSI session completed.")


if __name__ == "__main__":
    sys.exit(main()) 