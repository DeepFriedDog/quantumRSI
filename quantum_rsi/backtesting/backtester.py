"""
Quantum RSI - Backtesting Module

This module implements backtesting functionality for the quantum RSI model,
allowing validation against historical data and performance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import datetime
from typing import Dict, List, Tuple, Optional, Any
import os
import logging
import pickle
import json
from tqdm import tqdm
import time

# Import core components
from quantum_rsi.core.quantum_model import (
    QuantumWellPotential, 
    SchrodingerSolver, 
    QuantumTunneling,
    MaxwellsDemon
)
from quantum_rsi.data.dex_data import DEXDataProvider

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quantum_rsi.backtesting')


class Trade:
    """Class representing a backtest trade."""
    
    def __init__(self, 
                pair: str,
                direction: str,  # "BUY" or "SELL"
                entry_time: datetime.datetime,
                entry_price: float,
                stop_loss: float,
                take_profit: float,
                position_size: float = 0.01,
                quantum_state: Dict = None,
                signal_reason: str = ""):
        """
        Initialize a trade.
        
        Args:
            pair: Trading pair (e.g., "ETH/USDT")
            direction: Trade direction ("BUY" or "SELL")
            entry_time: Entry time
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size: Position size in ETH
            quantum_state: Quantum state at entry
            signal_reason: Reason for the signal
        """
        self.pair = pair
        self.direction = direction
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.quantum_state = quantum_state or {}
        self.signal_reason = signal_reason
        
        # Will be filled when trade is closed
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.pnl_percent = 0.0
        self.status = "OPEN"
        self.exit_reason = ""
        
    def close(self, 
             exit_time: datetime.datetime,
             exit_price: float,
             exit_reason: str = ""):
        """
        Close the trade.
        
        Args:
            exit_time: Exit time
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.status = "CLOSED"
        
        # Calculate P&L
        if self.direction == "BUY":
            # For long trades, profit is position_size * (exit_price - entry_price) / entry_price
            self.pnl_percent = (exit_price - self.entry_price) / self.entry_price * 100
            self.pnl = self.position_size * (exit_price - self.entry_price)
        else:  # SELL
            # For short trades, profit is position_size * (entry_price - exit_price) / entry_price
            self.pnl_percent = (self.entry_price - exit_price) / self.entry_price * 100
            self.pnl = self.position_size * (self.entry_price - exit_price)
            
    def to_dict(self) -> Dict:
        """Convert trade to dictionary for storage/analysis."""
        return {
            "pair": self.pair,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "status": self.status,
            "exit_reason": self.exit_reason,
            "signal_reason": self.signal_reason,
            "quantum_state": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in self.quantum_state.items()
            } if self.quantum_state else {}
        }
        
    @staticmethod
    def from_dict(data: Dict) -> 'Trade':
        """Create trade from dictionary."""
        trade = Trade(
            pair=data["pair"],
            direction=data["direction"],
            entry_time=data["entry_time"],
            entry_price=data["entry_price"],
            stop_loss=data["stop_loss"],
            take_profit=data["take_profit"],
            position_size=data["position_size"],
            quantum_state=data.get("quantum_state", {}),
            signal_reason=data.get("signal_reason", "")
        )
        
        if data["status"] == "CLOSED":
            trade.close(
                exit_time=data["exit_time"],
                exit_price=data["exit_price"],
                exit_reason=data.get("exit_reason", "")
            )
            trade.pnl = data["pnl"]
            trade.pnl_percent = data["pnl_percent"]
            
        return trade


class QuantumRSIBacktester:
    """
    Backtester for the Quantum RSI trading strategy.
    """
    
    def __init__(self, 
                pairs: List[str] = None,
                start_date: datetime.datetime = None,
                end_date: datetime.datetime = None,
                timeframe: str = None,
                initial_balance: float = 1.0,
                risk_per_trade: float = 0.01,
                volume_threshold: float = 2.0,
                vwap_threshold: float = 0.01,
                quantum_params: Dict = None,
                include_demon: bool = True,
                dex_provider: str = "uniswap",
                data_provider: DEXDataProvider = None):
        """
        Initialize the backtester.
        
        Args:
            pairs: List of trading pairs to backtest (e.g., ["ETH/USDT", "ETH/USDC"])
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Timeframe string (e.g., "1h", "1d")
            initial_balance: Initial account balance in ETH
            risk_per_trade: Risk per trade as fraction of balance
            volume_threshold: Threshold for volume spike detection
            vwap_threshold: Threshold for VWAP divergence
            quantum_params: Parameters for quantum model
            include_demon: Whether to include Maxwell's Demon filter
            dex_provider: DEX provider to use (uniswap, dydx, 1inch)
            data_provider: Optional existing data provider
        """
        self.pairs = pairs or ["ETH/USDT", "ETH/USDC"]
        self.start_date = start_date or datetime.datetime.now() - datetime.timedelta(days=365)
        self.end_date = end_date or datetime.datetime.now()
        self.timeframe = timeframe or "1h"
        self.dex_provider = dex_provider
        
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.volume_threshold = volume_threshold
        self.vwap_threshold = vwap_threshold
        
        # Quantum model parameters
        self.quantum_params = quantum_params or {
            "v0": 1.0,
            "a": 0.4,
            "m": 1.0,
            "hbar": 1.0,
            "evolution_steps": 10
        }
        
        self.include_demon = include_demon
        
        # Initialize data provider
        self.data_provider = data_provider or DEXDataProvider(
            pairs=self.pairs,
            timeframe=self.timeframe,
            provider=self.dex_provider
        )
        
        # Storage for backtesting
        self.historical_data = {}
        self.quantum_models = {}
        self.trades = []
        self.equity_curve = []
        self.balance = self.initial_balance
        self.metrics = {}
        
        # For storing signal history
        self.signals = []
        
    def load_historical_data(self, use_cache=True, cache_dir="./data_cache") -> None:
        """
        Load historical data for all trading pairs.
        
        Args:
            use_cache: Whether to use cached data if available
            cache_dir: Directory to store cached data
        """
        logger.info(f"Loading historical data...")
        
        # Create cache directory if it doesn't exist
        if use_cache:
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create cache directory: {e}")
                use_cache = False
        
        for pair in self.pairs:
            # Create cache filename
            cache_file = None
            if use_cache:
                start_str = self.start_date.strftime('%Y%m%d')
                end_str = self.end_date.strftime('%Y%m%d')
                # Replace / with _ in pair for filename
                safe_pair_name = pair.replace('/', '_')
                cache_file = os.path.join(cache_dir, f"{safe_pair_name}_{start_str}_{end_str}.pkl")
                
                # Try to load from cache
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'rb') as f:
                            self.historical_data[pair] = pickle.load(f)
                        logger.info(f"Loaded cached data for {pair}")
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to load cached data for {pair}: {e}")
            
            # Fetch data from provider if not loaded from cache
            logger.info(f"Fetching historical data for {pair} from {self.start_date} to {self.end_date}")
            
            # Get data using CryptoCompare as primary source
            df = self.data_provider.get_historical_data(
                pair=pair,
                bars=10000,  # Large number to ensure we get enough data
                start_time=self.start_date,
                data_source="cryptocompare"  # Specify CryptoCompare as the source
            )
            
            if df.empty:
                logger.error(f"Failed to get historical data for {pair}")
                continue
                
            # Log debug info about the dataframe
            logger.info(f"Got raw dataframe with {len(df)} rows and {df.columns.size} columns")
            logger.info(f"Columns: {', '.join(df.columns.tolist())}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
                
            # Filter by date range
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            logger.info(f"After date filtering: {len(df)} rows remain")
            
            # Calculate indicators
            df_with_indicators = self.calculate_indicators(df)
            
            # Log more debug info after indicators
            indicator_cols = [col for col in df_with_indicators.columns if col not in df.columns]
            logger.info(f"Added {len(indicator_cols)} indicators: {', '.join(indicator_cols)}")
            logger.info(f"Final dataframe shape: {df_with_indicators.shape}")
            
            # Check for NaN values in key columns
            for col in ['rsi', 'close', 'volume', 'atr']:
                nan_count = df_with_indicators[col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Column '{col}' has {nan_count} NaN values")
            
            # Check if dataframe is empty after processing
            if df_with_indicators.empty:
                logger.error(f"Dataframe for {pair} is empty after processing indicators")
                continue
                
            # Store data
            self.historical_data[pair] = df_with_indicators
            
            # Get timeframe from the data if not already set
            if self.timeframe is None and len(df) > 1:
                # Infer timeframe from the data
                time_diff = (df.index[1] - df.index[0]).total_seconds()
                if time_diff == 3600:
                    self.timeframe = "1h"  # 1 hour
                elif time_diff == 3600 * 4:
                    self.timeframe = "4h"  # 4 hours
                elif time_diff == 86400:
                    self.timeframe = "1d"  # Daily
                    
            # Cache the data
            if use_cache:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(df_with_indicators, f)
                    logger.info(f"Cached data for {pair}")
                except Exception as e:
                    logger.warning(f"Failed to cache data for {pair}: {e}")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators needed for the quantum RSI model.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate moving averages
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        df['sma200'] = df['close'].rolling(window=200).mean()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Calculate VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Calculate RSI momentum
        df['rsi_momentum'] = df['rsi'].diff(3)
        
        # Calculate Volume Ratio (compared to moving average)
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        
        # Price to VWAP ratio
        df['price_to_vwap'] = df['close'] / df['vwap']
        
        # Volume delta ratio (change in volume)
        df['volume_delta'] = df['volume'].diff()
        df['delta_ratio'] = df['volume_delta'] / df['volume'].shift()
        
        return df
    
    def _initialize_quantum_models(self) -> None:
        """Initialize quantum model components for all symbols."""
        for pair in self.pairs:
            # Create potential well model
            potential = QuantumWellPotential(
                v0=self.quantum_params["v0"],
                a=self.quantum_params["a"]
            )
            
            # Create Schrödinger solver
            solver = SchrodingerSolver(
                potential=potential,
                m=self.quantum_params["m"],
                hbar=self.quantum_params["hbar"]
            )
            
            # Create tunneling calculator
            tunneling = QuantumTunneling(
                potential=potential,
                m=self.quantum_params["m"],
                hbar=self.quantum_params["hbar"]
            )
            
            # Create Maxwell's Demon
            demon = MaxwellsDemon(
                potential=potential,
                volume_threshold=self.volume_threshold,
                vwap_threshold=self.vwap_threshold
            )
            
            # Store components
            self.quantum_models[pair] = {
                "potential": potential,
                "solver": solver,
                "tunneling": tunneling,
                "demon": demon,
                "current_state": None  # Will store current wavefunction state
            }
    
    def update_quantum_state(self, pair: str, row: pd.Series) -> Dict:
        """
        Update quantum state based on market data row.
        
        Args:
            pair: Trading pair
            row: DataFrame row with market data
            
        Returns:
            Dictionary with quantum state
        """
        try:
            # Check if the quantum model exists for this pair
            if pair not in self.quantum_models:
                logger.warning(f"No quantum model for {pair}, initializing now...")
                # Initialize the quantum model components for this pair
                self._initialize_quantum_models()
                if pair not in self.quantum_models:
                    # If still not initialized, create a default state
                    return {
                        "normalized_rsi": 0.0,
                        "barrier_adjustment": 0.0,
                        "allow_signal": True,
                        "demon_reason": "Default state, no quantum model",
                        "tunneling_probability": 0.0,
                        "well_probabilities": {"left": 0.5, "right": 0.5},
                        "error": "No quantum model for pair"
                    }
            
            model = self.quantum_models[pair]
            
            # Get normalized RSI value (-1 to 1)
            rsi_value = row.get('rsi', 50.0)
            normalized_rsi = (rsi_value - 50) / 50
            
            # Initialize wavefunction at current RSI position if not already initialized
            if model["current_state"] is None:
                model["solver"].initialize_wavefunction(center=normalized_rsi)
                model["current_state"] = "initialized"
            
            # Process market information with Maxwell's Demon
            if self.include_demon:
                try:
                    demon_result = model["demon"].process_information(
                        current_rsi=rsi_value,
                        volume_ratio=row.get('volume_ratio', 1.0),
                        price_to_vwap_ratio=row.get('price_to_vwap', 1.0),
                        delta_volume=row.get('delta_ratio', 0.0)
                    )
                except Exception as e:
                    logger.warning(f"Error in Maxwell's Demon processing: {str(e)}")
                    demon_result = {
                        "barrier_adjustment": 0.0,
                        "allow_signal": True,
                        "confidence": "Low",
                        "reason": f"Error in demon processing: {str(e)}"
                    }
            else:
                # Skip demon filtering if not included
                demon_result = {
                    "barrier_adjustment": 0.0,
                    "allow_signal": True,
                    "confidence": "Medium",
                    "reason": "Maxwell's Demon not included in backtest"
                }
            
            # Evolve quantum state with the adjusted potential
            try:
                probability_density = model["solver"].evolve(steps=self.quantum_params["evolution_steps"])
            except Exception as e:
                logger.warning(f"Error in quantum evolution: {str(e)}")
                probability_density = None
            
            # Calculate well probabilities
            try:
                well_probs = model["solver"].calculate_well_probabilities()
            except Exception as e:
                logger.warning(f"Error calculating well probabilities: {str(e)}")
                well_probs = {"left": 0.5, "right": 0.5}
            
            # Estimate tunneling probability based on current "energy"
            # We'll use RSI momentum as a proxy for energy against the trend
            try:
                rsi_momentum = abs(row.get('rsi_momentum', 0.0))
                energy = min(1.0, rsi_momentum / 10.0)  # Scale to [0, 1]
                
                tunneling_prob = model["tunneling"].calculate_wkb_tunneling(energy)
            except Exception as e:
                logger.warning(f"Error calculating tunneling probability: {str(e)}")
                tunneling_prob = 0.0
            
            # Record the updated state
            current_state = {
                "normalized_rsi": normalized_rsi,
                "barrier_adjustment": demon_result.get("barrier_adjustment", 0.0),
                "allow_signal": demon_result.get("allow_signal", True),
                "demon_reason": demon_result.get("reason", "Unknown"),
                "tunneling_probability": tunneling_prob,
                "well_probabilities": well_probs
            }
            
            return current_state
            
        except Exception as e:
            logger.error(f"Error updating quantum state: {str(e)}")
            # Return a default state if we encounter any errors
            return {
                "normalized_rsi": 0.0,
                "barrier_adjustment": 0.0,
                "allow_signal": False,
                "demon_reason": f"Error: {str(e)}",
                "tunneling_probability": 0.0,
                "well_probabilities": {"left": 0.5, "right": 0.5},
                "error": f"Exception: {str(e)}"
            }
    
    def generate_trading_signal(self, pair: str, row: pd.Series, quantum_state: Dict) -> Dict:
        """
        Generate trading signal based on quantum model and market data.
        
        Args:
            pair: Trading pair
            row: DataFrame row with market data
            quantum_state: Current quantum state
            
        Returns:
            Dictionary with signal details
        """
        # Get RSI
        rsi = row.get('rsi', 50.0)
        
        # Default signal (no action)
        signal = {
            "time": row.name,  # Index of the row is the timestamp
            "action": "NONE",
            "confidence": 0.0,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "reason": "No signal generated",
            "quantum_state": quantum_state
        }
        
        # Early exit if demon blocked signal
        if not quantum_state.get("allow_signal", True):
            signal["reason"] = f"Maxwell's Demon blocked signal: {quantum_state.get('demon_reason', 'N/A')}"
            return signal
            
        # Get tunneling probability (probability of trend reversal)
        tunneling_prob = quantum_state.get("tunneling_probability", 0.0)
        
        # Get rsi momentum
        rsi_momentum = row.get('rsi_momentum', 0.0)
        
        # RSI crossing oversold -> bullish reversal
        if rsi < 30 and rsi_momentum > 0:
            # Check if significant tunneling probability
            if tunneling_prob > 0.2:  # Adjust threshold as needed
                signal["action"] = "BUY"
                signal["confidence"] = tunneling_prob
                signal["reason"] = (
                    f"Bullish reversal signal: RSI oversold ({rsi:.2f}) with positive momentum. "
                    f"Tunneling probability: {tunneling_prob:.4f}. "
                    f"Demon analysis: {quantum_state.get('demon_reason', 'N/A')}"
                )
                
                # Calculate entry, stop loss and take profit
                entry_price = row['close']
                atr = row.get('atr', row['high'] - row['low'])
                
                signal["entry_price"] = entry_price
                signal["stop_loss"] = entry_price - self.quantum_params["a"] * atr
                signal["take_profit"] = entry_price + 2 * self.quantum_params["a"] * atr  # 2:1 reward:risk
                
        # RSI crossing overbought -> bearish reversal
        elif rsi > 70 and rsi_momentum < 0:
            # Check if significant tunneling probability
            if tunneling_prob > 0.2:  # Adjust threshold as needed
                signal["action"] = "SELL"
                signal["confidence"] = tunneling_prob
                signal["reason"] = (
                    f"Bearish reversal signal: RSI overbought ({rsi:.2f}) with negative momentum. "
                    f"Tunneling probability: {tunneling_prob:.4f}. "
                    f"Demon analysis: {quantum_state.get('demon_reason', 'N/A')}"
                )
                
                # Calculate entry, stop loss and take profit
                entry_price = row['close']
                atr = row.get('atr', row['high'] - row['low'])
                
                signal["entry_price"] = entry_price
                signal["stop_loss"] = entry_price + self.quantum_params["a"] * atr
                signal["take_profit"] = entry_price - 2 * self.quantum_params["a"] * atr  # 2:1 reward:risk
        else:
            # No trading conditions met
            signal["action"] = "HOLD"
            signal["reason"] = (
                f"No trading signal. RSI: {rsi:.2f}, Momentum: {rsi_momentum:.2f}, "
                f"Tunneling probability: {tunneling_prob:.4f}"
            )
        
        return signal
    
    def run_backtest(self, 
                  start_date: Optional[datetime.datetime] = None,
                  end_date: Optional[datetime.datetime] = None,
                  show_trades: bool = False,
                  save_results: bool = False) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            show_trades: Whether to print trade details
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with backtest metrics
        """
        # Set dates if provided
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
            
        # Initialize equity curve
        self.equity_curve = [{"date": self.start_date, "equity": self.initial_balance}]
        self.balance = self.initial_balance
        
        # Load historical data
        if not self.historical_data:
            logger.info("No historical data loaded. Loading now...")
            self.load_historical_data()
            
        # Check if we have any historical data
        if not self.historical_data:
            logger.error("No historical data available. Cannot run backtest.")
            return {}
            
        # Check that we have valid data for at least one pair
        valid_data = False
        for pair, df in self.historical_data.items():
            if df is not None and not df.empty and 'rsi' in df.columns:
                valid_data = True
                logger.info(f"Valid data found for {pair}: {df.shape} rows, RSI range: {df['rsi'].min():.2f}-{df['rsi'].max():.2f}")
                break
                
        if not valid_data:
            logger.error("No valid data with RSI indicators found for any pair. Cannot run backtest.")
            return {}
            
        # Initialize quantum models
        if not self.quantum_models:
            logger.info("Initializing quantum models...")
            self._initialize_quantum_models()
            
        logger.info("Starting backtest...")
        
        # Track open positions
        open_positions = {}
        
        # Signal counters for debugging
        signal_counters = {
            "total": 0,
            "BUY": 0,
            "SELL": 0,
            "HOLD": 0,
            "NONE": 0,
            "blocked_by_demon": 0
        }
        
        # Process each pair
        for pair in self.pairs:
            # Skip if no data or empty dataframe
            if pair not in self.historical_data:
                logger.warning(f"No data for {pair}, skipping")
                continue
                
            df = self.historical_data[pair]
            
            # Skip if dataframe is empty or doesn't have the required columns
            if df is None or df.empty:
                logger.warning(f"Empty dataframe for {pair}, skipping")
                continue
                
            # Check if RSI column exists
            if 'rsi' not in df.columns:
                logger.warning(f"No RSI data for {pair}, skipping")
                continue
                
            # Check if there are any non-NaN RSI values
            if df['rsi'].isna().all():
                logger.warning(f"All RSI values are NaN for {pair}, skipping")
                continue
                
            # Log details about the data we're processing
            logger.info(f"Running backtest for {pair}...")
            logger.info(f"Data shape: {df.shape}")
            
            # Log data stats
            data_stats = {
                "periods": len(df),
                "start": df.index.min(),
                "end": df.index.max(),
                "min_rsi": df["rsi"].min(),
                "max_rsi": df["rsi"].max(),
                "mean_rsi": df["rsi"].mean(),
                "min_price": df["close"].min(),
                "max_price": df["close"].max(),
                "nan_rsi": df["rsi"].isna().sum()
            }
            
            logger.info(f"Data stats: {data_stats}")
            
            # Calculate timestamp for progress logging
            total_rows = len(df)
            last_log = time.time()
            
            # Reset pair-specific signal counters
            pair_signals = {"BUY": 0, "SELL": 0, "HOLD": 0, "NONE": 0, "blocked": 0}
            
            # Iterate over all data points
            for idx, (timestamp, row) in enumerate(df.iterrows()):
                # Log progress every 5 seconds
                if time.time() - last_log > 5:
                    progress = 100 * idx / total_rows
                    logger.info(f"Processing {pair}: {progress:.2f}% complete")
                    last_log = time.time()
                
                # Skip if RSI is NaN
                if pd.isna(row.get('rsi')):
                    continue
                    
                # Update quantum state
                quantum_state = self.update_quantum_state(pair, row)
                
                # Get trading signal
                signal = self.generate_trading_signal(pair, row, quantum_state)
                
                # Count signals
                signal_counters["total"] += 1
                action = signal.get("action", "NONE")
                
                if action in ["BUY", "SELL", "HOLD", "NONE"]:
                    signal_counters[action] += 1
                    pair_signals[action] += 1
                
                if not quantum_state.get("allow_signal", True):
                    signal_counters["blocked_by_demon"] += 1
                    pair_signals["blocked"] += 1
                
                # Record the signal
                self.signals.append({
                    "timestamp": timestamp,
                    "pair": pair,
                    "price": row['close'],
                    "quantum_state": quantum_state,
                    "signal": signal
                })
                
                # Check for stop loss or take profit on existing positions
                if pair in open_positions:
                    trade = open_positions[pair]
                    
                    # Check stop loss
                    if trade.direction == "BUY" and row['low'] <= trade.stop_loss:
                        trade.close(timestamp, trade.stop_loss, "Stop Loss")
                        self.trades.append(trade)
                        del open_positions[pair]
                        
                        # Update balance
                        self.balance += trade.pnl
                        self.equity_curve.append({
                            "date": timestamp,
                            "equity": self.balance
                        })
                        
                        if show_trades:
                            logger.info(f"STOPPED OUT: {trade.direction} {pair} at {trade.exit_price:.2f}, PnL: {trade.pnl:.2f}")
                            
                    elif trade.direction == "SELL" and row['high'] >= trade.stop_loss:
                        trade.close(timestamp, trade.stop_loss, "Stop Loss")
                        self.trades.append(trade)
                        del open_positions[pair]
                        
                        # Update balance
                        self.balance += trade.pnl
                        self.equity_curve.append({
                            "date": timestamp,
                            "equity": self.balance
                        })
                        
                        if show_trades:
                            logger.info(f"STOPPED OUT: {trade.direction} {pair} at {trade.exit_price:.2f}, PnL: {trade.pnl:.2f}")
                            
                    # Check take profit
                    elif trade.direction == "BUY" and row['high'] >= trade.take_profit:
                        trade.close(timestamp, trade.take_profit, "Take Profit")
                        self.trades.append(trade)
                        del open_positions[pair]
                        
                        # Update balance
                        self.balance += trade.pnl
                        self.equity_curve.append({
                            "date": timestamp,
                            "equity": self.balance
                        })
                        
                        if show_trades:
                            logger.info(f"TAKE PROFIT: {trade.direction} {pair} at {trade.exit_price:.2f}, PnL: {trade.pnl:.2f}")
                            
                    elif trade.direction == "SELL" and row['low'] <= trade.take_profit:
                        trade.close(timestamp, trade.take_profit, "Take Profit")
                        self.trades.append(trade)
                        del open_positions[pair]
                        
                        # Update balance
                        self.balance += trade.pnl
                        self.equity_curve.append({
                            "date": timestamp,
                            "equity": self.balance
                        })
                        
                        if show_trades:
                            logger.info(f"TAKE PROFIT: {trade.direction} {pair} at {trade.exit_price:.2f}, PnL: {trade.pnl:.2f}")
                            
                # Execute new trades based on signal
                if signal["action"] in ["BUY", "SELL"] and pair not in open_positions:
                    # Calculate position size
                    risk_amount = self.balance * self.risk_per_trade
                    
                    # Calculate stop loss
                    if signal["action"] == "BUY":
                        stop_loss = signal.get("stop_loss") or row['close'] * 0.95  # Default 5% stop
                        take_profit = signal.get("take_profit") or row['close'] * 1.15  # Default 15% take profit
                        
                        # Adjust position size based on stop loss distance
                        stop_distance = (row['close'] - stop_loss) / row['close']
                        position_size = risk_amount / (stop_distance * row['close'])
                        
                    else:  # SELL
                        stop_loss = signal.get("stop_loss") or row['close'] * 1.05  # Default 5% stop
                        take_profit = signal.get("take_profit") or row['close'] * 0.85  # Default 15% take profit
                        
                        # Adjust position size based on stop loss distance
                        stop_distance = (stop_loss - row['close']) / row['close']
                        position_size = risk_amount / (stop_distance * row['close'])
                    
                    # Cap position size at 20% of balance
                    max_position = self.balance * 0.2 / row['close']
                    position_size = min(position_size, max_position)
                    
                    # Create trade
                    trade = Trade(
                        pair=pair,
                        direction=signal["action"],
                        entry_time=timestamp,
                        entry_price=row['close'],
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size=position_size,
                        quantum_state=quantum_state,
                        signal_reason=signal.get("reason", "")
                    )
                    
                    # Add to open positions
                    open_positions[pair] = trade
                    
                    if show_trades:
                        logger.info(f"NEW TRADE: {trade.direction} {pair} at {trade.entry_price:.2f}, "
                                    f"SL: {trade.stop_loss:.2f}, TP: {trade.take_profit:.2f}")
            
            # Log signal statistics for this pair
            logger.info(f"Signal stats for {pair}: {pair_signals}")
        
        # Close any remaining open positions at the last available price
        for pair, trade in list(open_positions.items()):
            if pair in self.historical_data and not self.historical_data[pair].empty:
                last_price = self.historical_data[pair].iloc[-1]['close']
                trade.close(self.historical_data[pair].index[-1], last_price, "End of Test")
                self.trades.append(trade)
                
                # Update balance
                self.balance += trade.pnl
                
                if show_trades:
                    logger.info(f"CLOSED AT END: {trade.direction} {pair} at {trade.exit_price:.2f}, PnL: {trade.pnl:.2f}")
        
        # Final equity
        self.equity_curve.append({
            "date": self.end_date,
            "equity": self.balance
        })
        
        # Log overall signal statistics
        logger.info(f"Overall signal statistics: {signal_counters}")
        logger.info(f"Total trades executed: {len(self.trades)}")
        
        # Calculate metrics
        self._calculate_metrics()
        
        # Print results
        self.display_results()
        
        if show_trades:
            self.plot_results(show_trades=True)
            
        if save_results:
            self.save_results()
            
        logger.info("Backtest completed.")
        
        return self.metrics
    
    def _calculate_metrics(self) -> None:
        """Calculate performance metrics for the backtest."""
        # Extract closed trades
        closed_trades = [t for t in self.trades if t.status == "CLOSED"]
        
        if not closed_trades:
            logger.warning("No closed trades to calculate metrics")
            self.metrics = {
                "total_trades": 0,
                "net_profit": 0,
                "win_rate": 0,
                "average_win": 0,
                "average_loss": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }
            return
            
        # Basic metrics
        total_trades = len(closed_trades)
        profitable_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        win_count = len(profitable_trades)
        loss_count = len(losing_trades)
        
        total_profit = sum(t.pnl for t in profitable_trades)
        total_loss = sum(t.pnl for t in losing_trades)
        net_profit = total_profit + total_loss
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        average_win = total_profit / win_count if win_count > 0 else 0
        average_loss = total_loss / loss_count if loss_count > 0 else 0
        profit_factor = abs(total_profit / total_loss) if total_loss < 0 else float('inf')
        
        # Calculate drawdown
        equity_series = pd.DataFrame(self.equity_curve).set_index('date')['equity']
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100  # As percentage
        max_drawdown = abs(drawdown.min())
        
        # Calculate Sharpe ratio (simplified)
        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Store metrics
        self.metrics = {
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "profit_percentage": (net_profit / self.initial_balance) * 100,
            "average_win": average_win,
            "average_loss": average_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "final_equity": self.equity_curve[-1]["equity"] if self.equity_curve else self.initial_balance
        }
        
    def display_results(self) -> None:
        """Display the backtest results."""
        if not self.metrics:
            logger.warning("No metrics calculated. Run backtest first.")
            return
            
        print("\n=== Quantum RSI Backtest Results ===")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Pairs: {', '.join(self.pairs)}")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Equity: ${self.metrics.get('final_equity', 0):.2f}")
        print(f"Net Profit: ${self.metrics.get('net_profit', 0):.2f} ({self.metrics.get('profit_percentage', 0):.2f}%)")
        print(f"Total Trades: {self.metrics.get('total_trades', 0)}")
        print(f"Win Rate: {self.metrics.get('win_rate', 0):.2f}%")
        print(f"Profit Factor: {self.metrics.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown: {self.metrics.get('max_drawdown', 0):.2f}%")
        print(f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
        print("=====================================")
    
    def plot_results(self, show_trades: bool = True, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot backtest results.
        
        Args:
            show_trades: Whether to show trade markers
            figsize: Figure size
        """
        if not self.trades or not self.equity_curve:
            logger.warning("No trades or equity data to plot")
            return
            
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 1, height_ratios=[3, 1, 1])
        
        # Equity curve
        ax1 = fig.add_subplot(gs[0])
        equity_df['equity'].plot(ax=ax1, linewidth=2)
        ax1.set_title('Quantum RSI Backtest Results')
        ax1.set_ylabel('Account Equity ($)')
        ax1.grid(True, alpha=0.3)
        
        # Add initial balance line
        ax1.axhline(self.initial_balance, color='r', linestyle='--', alpha=0.5, label='Initial Balance')
        
        # Mark trades if requested
        if show_trades and self.historical_data and self.trades:
            # Get price data for main pair
            pair = self.pairs[0] if self.pairs else None
            if pair and pair in self.historical_data:
                price_df = self.historical_data[pair]
                
                # Plot successful trades
                for trade in self.trades:
                    if trade.pair == pair and trade.status == "CLOSED":
                        # Entry point
                        entry_date = trade.entry_time
                        entry_price = trade.entry_price
                        
                        # Exit point
                        exit_date = trade.exit_time
                        exit_price = trade.exit_price
                        
                        # Marker color based on profit/loss
                        color = 'green' if trade.pnl > 0 else 'red'
                        
                        # Plot entry and exit points
                        if trade.direction == "BUY":
                            ax1.scatter(entry_date, entry_price, color=color, marker='^', s=100, alpha=0.7)
                            ax1.scatter(exit_date, exit_price, color=color, marker='v', s=100, alpha=0.7)
                        else:  # SELL
                            ax1.scatter(entry_date, entry_price, color=color, marker='v', s=100, alpha=0.7)
                            ax1.scatter(exit_date, exit_price, color=color, marker='^', s=100, alpha=0.7)
        
        # Drawdown subplot
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        running_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - running_max) / running_max * 100
        drawdown.plot(ax=ax2, color='red', linewidth=1)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('')
        ax2.grid(True, alpha=0.3)
        
        # Monthly returns subplot
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        monthly_returns = equity_df['equity'].resample('M').last().pct_change() * 100
        monthly_returns.plot(kind='bar', ax=ax3)
        ax3.set_ylabel('Monthly Return (%)')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.tight_layout()
        plt.show()
        
    def save_results(self, filename: str = "quantum_rsi_backtest_results.json") -> None:
        """
        Save backtest results to file.
        
        Args:
            filename: Output filename
        """
        # Prepare data
        results = {
            "backtest_config": {
                "pairs": self.pairs,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "initial_balance": self.initial_balance,
                "risk_per_trade": self.risk_per_trade,
                "quantum_params": self.quantum_params,
                "include_demon": self.include_demon,
                "dex_provider": self.dex_provider
            },
            "metrics": self.metrics,
            "trades": [trade.to_dict() for trade in self.trades],
            "equity_curve": self.equity_curve
        }
        
        # Convert datetime objects and ensure serializable
        def datetime_handler(x):
            if isinstance(x, datetime.datetime):
                return x.isoformat()
            raise TypeError(f"Object of type {type(x)} is not JSON serializable")
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results, f, default=datetime_handler, indent=2)
            
        logger.info(f"Backtest results saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Set up backtest
    backtester = QuantumRSIBacktester(
        pairs=["ETH/USDT", "ETH/USDC"],
        start_date=datetime.datetime(2021, 1, 1),
        end_date=datetime.datetime(2022, 1, 1),
        initial_balance=1.0,
        risk_per_trade=0.02,
        volume_threshold=2.0,
        vwap_threshold=0.01,
        quantum_params={
            "v0": 1.0,
            "a": 0.4,
            "m": 1.0,
            "hbar": 1.0,
            "evolution_steps": 10
        },
        include_demon=True,
        dex_provider="uniswap"
    )
    
    # Load data and run backtest
    backtester.load_historical_data()
    backtester.run_backtest()
    
    # Print and plot results
    backtester.display_results()
    backtester.plot_results()
    
    # Save results
    backtester.save_results() 