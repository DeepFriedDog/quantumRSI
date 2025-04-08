#!/usr/bin/env python
# Quantum RSI Live Trading Strategy with Maxwell's Demon
# For Bitcoin and Ethereum markets
# Important: Run with a demo account first

import numpy as np
import pandas as pd
import time
import ccxt
from datetime import datetime, timedelta
import logging
import os
import json
from threading import Thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_rsi_live.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('quantum_rsi_live')

# ------ Quantum Mechanics and RSI Utility Functions ------

def calculate_rsi(prices, period=14):
    """Calculate RSI values for price series"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def normalize_rsi(rsi_value):
    """Map RSI (0-100) to normalized position (-1 to 1)"""
    return (rsi_value - 50) / 50

def calculate_potential(x, v0=1.0, a=0.4):
    """Calculate potential in double-well"""
    return v0 * (x**2 - a**2)**2

def tunneling_probability(rsi, rsi_momentum, v0=1.5, a=0.5, mass=1.0, hbar=1.0):
    """Calculate tunneling probability based on WKB approximation.
    This represents likelihood of trend reversal."""
    normalized_rsi = normalize_rsi(rsi)
    
    # Energy proxy based on RSI momentum
    energy = min(abs(rsi_momentum) / 10.0, 0.95)
    
    # Higher barrier when far from center
    barrier_height = v0 * a**4  # height at x=0
    
    # Classical turning points estimation
    if abs(normalized_rsi) < a:
        # Inside the central barrier
        return 0.8  # High probability once started crossing
    else:
        # In one of the wells
        # Calculate approximate barrier width
        barrier_width = 2 * a  # Simplification
        
        # WKB formula approximation
        k = np.sqrt(2 * mass * (barrier_height - energy)) / hbar
        tunneling_prob = np.exp(-2 * k * barrier_width)
        
        # Scale for better trading signals (raw tunneling prob is typically tiny)
        tunneling_prob = min(0.9, tunneling_prob * 10000)
        
        # Adjust based on distance from center
        if abs(normalized_rsi) > 0.8:  # In extreme territory
            tunneling_prob *= 0.5  # Reduce probability (trend persistence)
            
        return tunneling_prob

# ------ Maxwell's Demon Logic ------

def demon_filter(rsi, rsi_prev, volume, avg_volume, price_to_vwap, direction, on_chain_metrics=None):
    """Maxwell's Demon filter for detecting whale activity
    Returns: (allow_signal, confidence, reason)"""
    
    demon_confidence = 0.5  # Default neutral confidence
    allow_signal = True
    reason = ""
    
    # Volume spike check (whale activity)
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
    if volume_ratio > 2.0:
        demon_confidence += 0.2
        reason += "Volume spike detected (2x average). "
    
    # VWAP divergence (price far from value)
    if direction == "BUY" and price_to_vwap < 0.97:
        demon_confidence += 0.15
        reason += "Price significantly below VWAP. "
    elif direction == "SELL" and price_to_vwap > 1.03:
        demon_confidence += 0.15
        reason += "Price significantly above VWAP. "
    
    # RSI momentum check
    rsi_change = rsi - rsi_prev
    if (direction == "BUY" and rsi_change > 3) or (direction == "SELL" and rsi_change < -3):
        demon_confidence += 0.15
        reason += f"Strong RSI momentum ({rsi_change:.1f}). "
    
    # On-chain metrics analysis (if available)
    if on_chain_metrics:
        # Check for exchange inflows/outflows (whale activity)
        if direction == "BUY" and on_chain_metrics.get('exchange_outflow', 0) > on_chain_metrics.get('exchange_inflow', 0):
            demon_confidence += 0.15
            reason += "Net exchange outflows detected (accumulation). "
        elif direction == "SELL" and on_chain_metrics.get('exchange_inflow', 0) > on_chain_metrics.get('exchange_outflow', 0):
            demon_confidence += 0.15
            reason += "Net exchange inflows detected (distribution). "
    
    # Decide if signal should be allowed
    if demon_confidence < 0.6:  # Higher threshold for live trading
        allow_signal = False
        reason += "Low confidence whale activity. "
    
    return allow_signal, demon_confidence, reason

# ------ Live Trading Class ------

class QuantumRSILiveTrader:
    def __init__(self, 
                 symbols=['BTC/USDT', 'ETH/USDT'],
                 timeframe='1h',
                 api_key=None,
                 api_secret=None,
                 exchange_id='binance',
                 use_testnet=True,
                 initial_balance=10000,
                 max_position_size=0.03,  # 3% of portfolio per trade
                 config=None):
        """Initialize the Quantum RSI Live Trader"""
        
        # Trading parameters
        self.symbols = symbols
        self.timeframe = timeframe
        self.use_testnet = use_testnet
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        
        # Load configuration if provided, otherwise use defaults
        self.config = config or {
            # Strategy parameters
            'rsi_period': 14,        
            'rsi_high': 65,          # Lower than standard for crypto markets
            'rsi_low': 35,           # Higher than standard for crypto markets
            
            # Quantum model parameters (optimized for crypto)
            'v0': 1.5,               # Deeper wells for crypto trends
            'a': 0.5,                # Wider separation for larger price swings
            'mass': 1.0,             # Effective mass (trend inertia)
            'hbar': 1.0,             # Reduced Planck's constant analog
            
            # Demon parameters
            'volume_threshold': 2.0,  # Volume spike detection
            'vwap_period': 20,        # Period for VWAP calculation
            
            # Risk management
            'stop_loss_atr': 2.0,     # Stop loss multiplier based on ATR
            'take_profit_atr': 4.0,   # Take profit multiplier based on ATR
            'atr_period': 14,         # ATR period
            'cooldown_bars': 6,       # Bars to wait after a trade
            
            # Order parameters
            'order_type': 'market',   # 'market' or 'limit'
            'limit_offset': 0.001,    # 0.1% below/above for limit orders
        }
        
        # Initialize exchange
        self.connect_exchange(exchange_id, api_key, api_secret)
        
        # Store state
        self.positions = {}  # Current positions by symbol
        self.historical_data = {}  # Historical data by symbol
        self.indicators = {}  # Calculated indicators by symbol
        self.recent_signals = {}  # Recent signals by symbol
        self.last_trade_time = {}  # Last trade time by symbol
        self.on_chain_metrics = {}  # On-chain metrics if available
        
        # Statistics
        self.trade_history = []
        self.performance = {
            'starting_balance': initial_balance,
            'current_balance': initial_balance,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0,
            'total_loss': 0,
        }
        
        logger.info(f"Quantum RSI Live Trader initialized with {len(symbols)} symbols")
        
    def connect_exchange(self, exchange_id, api_key, api_secret):
        """Connect to exchange"""
        try:
            exchange_class = getattr(ccxt, exchange_id)
            
            exchange_config = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            }
            
            # Use testnet if specified
            if self.use_testnet:
                if exchange_id == 'binance':
                    exchange_config.update({
                        'options': {
                            'defaultType': 'future',
                            'testnet': True,
                        }
                    })
                logger.info("Using TESTNET for paper trading")
                
            self.exchange = exchange_class(exchange_config)
            
            # Load markets
            self.exchange.load_markets()
            logger.info(f"Connected to {exchange_id} exchange")
            
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {str(e)}")
            raise
    
    def fetch_historical_data(self, symbol, limit=500):
        """Fetch historical data for a symbol"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def calculate_indicators(self, df):
        """Calculate trading indicators from price data"""
        indicators = {}
        
        # Verify we have required columns and sufficient data
        if 'close' not in df.columns or len(df) < 50:
            logger.warning(f"Insufficient data for indicator calculation")
            return indicators
        
        # Calculate RSI
        indicators['rsi'] = calculate_rsi(df['close'], self.config['rsi_period'])
        
        # Calculate VWAP
        indicators['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Calculate volume averages
        indicators['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Calculate ATR (Average True Range) for position sizing
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = tr.rolling(window=self.config['atr_period']).mean()
        
        # Calculate RSI momentum (change over last 3 periods)
        indicators['rsi_momentum'] = indicators['rsi'] - indicators['rsi'].shift(3)
        
        return indicators
    
    def get_on_chain_metrics(self, symbol):
        """Get on-chain metrics for whale detection
        This is a placeholder - in a real system, connect to on-chain data APIs"""
        # In a real implementation, connect to APIs like Glassnode, IntoTheBlock, etc.
        # For demo purposes, return some default values
        coin = symbol.split('/')[0]  # Extract BTC, ETH, etc.
        
        # Placeholder metrics
        metrics = {
            'exchange_inflow': np.random.random() * 100,    # BTC flowing into exchanges
            'exchange_outflow': np.random.random() * 100,   # BTC flowing out of exchanges
            'large_transactions': np.random.random() * 10,  # Number of >$1M transactions
            'whale_wallet_activity': np.random.random(),    # Whale wallet activity indicator
        }
        
        logger.debug(f"Retrieved on-chain metrics for {symbol}")
        return metrics
    
    def check_for_signals(self, symbol):
        """Check for trading signals based on current data and indicators"""
        # Check if we have the data
        if symbol not in self.historical_data or symbol not in self.indicators:
            logger.warning(f"No data or indicators for {symbol}")
            return None
        
        df = self.historical_data[symbol]
        ind = self.indicators[symbol]
        
        # Get current values
        current_price = df['close'].iloc[-1]
        rsi = ind['rsi'].iloc[-1]
        rsi_prev = ind['rsi'].iloc[-2] if len(ind['rsi']) > 1 else None
        volume = df['volume'].iloc[-1]
        avg_volume = ind['volume_ma'].iloc[-1] if not pd.isna(ind['volume_ma'].iloc[-1]) else volume
        vwap = ind['vwap'].iloc[-1] if not pd.isna(ind['vwap'].iloc[-1]) else current_price
        rsi_momentum = ind['rsi_momentum'].iloc[-1] if not pd.isna(ind['rsi_momentum'].iloc[-1]) else 0
        current_atr = ind['atr'].iloc[-1] if not pd.isna(ind['atr'].iloc[-1]) else (df['high'].iloc[-1] - df['low'].iloc[-1])
        price_to_vwap = current_price / vwap if vwap > 0 else 1.0
        
        signal = None
        reason = ""
        confidence = 0.0
        
        # Get on-chain metrics for enhanced whale detection
        on_chain = self.get_on_chain_metrics(symbol)
        
        # Calculate quantum tunneling probability
        tunnel_prob = tunneling_probability(
            rsi, 
            rsi_momentum, 
            self.config['v0'], 
            self.config['a'], 
            self.config['mass'], 
            self.config['hbar']
        )
        
        # Check for cooldown period
        last_trade = self.last_trade_time.get(symbol, 0)
        current_bar = len(df)
        if current_bar - last_trade < self.config['cooldown_bars']:
            logger.debug(f"In cooldown period for {symbol}, skipping signal check")
            return None
        
        # Check for existing position
        if symbol in self.positions:
            logger.debug(f"Position already exists for {symbol}, skipping new entry signal")
            return None
        
        # Check for quantum tunneling signals (trend reversals)
        
        # Oversold to bullish (RSI crossing up from below threshold)
        if (rsi_prev and rsi_prev < self.config['rsi_low'] and rsi >= self.config['rsi_low']) or \
           (rsi < self.config['rsi_low'] and rsi_momentum > 3):  # Strong momentum from oversold
            
            signal = "BUY"
            confidence = min(0.9, tunnel_prob + 0.2)  # Base confidence
            reason = f"RSI: {rsi:.1f} crossing up from oversold, momentum: {rsi_momentum:.1f}"
            
            # Apply Maxwell's Demon filter with on-chain data
            allow_signal, demon_confidence, demon_reason = demon_filter(
                rsi, rsi_prev, volume, avg_volume, price_to_vwap, signal, on_chain
            )
            
            if not allow_signal:
                logger.info(f"Signal {signal} for {symbol} vetoed by demon: {demon_reason}")
                return None
                
            # Calculate order parameters
            quantity = self.calculate_position_size(symbol, current_price, current_atr)
            stop_loss = current_price - current_atr * self.config['stop_loss_atr']
            take_profit = current_price + current_atr * self.config['take_profit_atr']
            
            signal_data = {
                'symbol': symbol,
                'signal': signal,
                'price': current_price,
                'confidence': demon_confidence,
                'reason': reason + " " + demon_reason,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now(),
                'metrics': {
                    'rsi': rsi,
                    'rsi_momentum': rsi_momentum,
                    'tunnel_prob': tunnel_prob,
                    'volume_ratio': volume / avg_volume if avg_volume > 0 else 1.0,
                    'price_to_vwap': price_to_vwap,
                    'atr': current_atr
                }
            }
            
            return signal_data
            
        # Overbought to bearish (RSI crossing down from above threshold)
        elif (rsi_prev and rsi_prev > self.config['rsi_high'] and rsi <= self.config['rsi_high']) or \
             (rsi > self.config['rsi_high'] and rsi_momentum < -3):  # Strong momentum from overbought
            
            signal = "SELL"
            confidence = min(0.9, tunnel_prob + 0.2)  # Base confidence
            reason = f"RSI: {rsi:.1f} crossing down from overbought, momentum: {rsi_momentum:.1f}"
            
            # Apply Maxwell's Demon filter with on-chain data
            allow_signal, demon_confidence, demon_reason = demon_filter(
                rsi, rsi_prev, volume, avg_volume, price_to_vwap, signal, on_chain
            )
            
            if not allow_signal:
                logger.info(f"Signal {signal} for {symbol} vetoed by demon: {demon_reason}")
                return None
                
            # Calculate order parameters
            quantity = self.calculate_position_size(symbol, current_price, current_atr)
            stop_loss = current_price + current_atr * self.config['stop_loss_atr']
            take_profit = current_price - current_atr * self.config['take_profit_atr']
            
            signal_data = {
                'symbol': symbol,
                'signal': signal,
                'price': current_price,
                'confidence': demon_confidence,
                'reason': reason + " " + demon_reason,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now(),
                'metrics': {
                    'rsi': rsi,
                    'rsi_momentum': rsi_momentum,
                    'tunnel_prob': tunnel_prob,
                    'volume_ratio': volume / avg_volume if avg_volume > 0 else 1.0,
                    'price_to_vwap': price_to_vwap,
                    'atr': current_atr
                }
            }
            
            return signal_data
            
        return None
    
    def calculate_position_size(self, symbol, price, atr):
        """Calculate position size based on risk management"""
        # Get account balance
        try:
            if self.use_testnet:
                account_balance = self.performance['current_balance']
            else:
                balances = self.exchange.fetch_balance()
                account_balance = balances['total']['USDT'] if 'USDT' in balances['total'] else self.initial_balance
            
            # Risk per trade (1% of account)
            risk_amount = account_balance * 0.01
            
            # Calculate stop loss distance
            stop_loss_pct = (self.config['stop_loss_atr'] * atr) / price
            
            # Calculate position size
            position_value = risk_amount / stop_loss_pct
            
            # Limit max position size
            max_position = account_balance * self.max_position_size
            position_value = min(position_value, max_position)
            
            # Convert to quantity
            quantity = position_value / price
            
            # Round to appropriate precision for the symbol
            if symbol in self.exchange.markets:
                precision = self.exchange.markets[symbol]['precision']['amount']
                quantity = round(quantity, precision)
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.01  # Default small position size on error
    
    def execute_trade(self, signal_data):
        """Execute trade based on signal data"""
        if not signal_data:
            return False
        
        symbol = signal_data['symbol']
        signal = signal_data['signal']
        price = signal_data['price']
        quantity = signal_data['quantity']
        stop_loss = signal_data['stop_loss']
        take_profit = signal_data['take_profit']
        
        logger.info(f"Executing {signal} trade for {symbol} at {price:.2f}, "
                   f"quantity: {quantity:.6f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
        
        try:
            # Create order
            if signal == "BUY":
                if self.config['order_type'] == 'market':
                    order = self.exchange.create_market_buy_order(symbol, quantity)
                else:
                    limit_price = price * (1 - self.config['limit_offset'])
                    order = self.exchange.create_limit_buy_order(symbol, quantity, limit_price)
            else:  # SELL
                if self.config['order_type'] == 'market':
                    order = self.exchange.create_market_sell_order(symbol, quantity)
                else:
                    limit_price = price * (1 + self.config['limit_offset'])
                    order = self.exchange.create_limit_sell_order(symbol, quantity, limit_price)
            
            # Place stop loss and take profit orders
            if self.exchange.has['createOrder']:
                if signal == "BUY":
                    sl_order = self.exchange.create_order(
                        symbol, 'stop', 'sell', quantity, None, {'stopPrice': stop_loss}
                    )
                    tp_order = self.exchange.create_order(
                        symbol, 'limit', 'sell', quantity, take_profit
                    )
                else:  # SELL
                    sl_order = self.exchange.create_order(
                        symbol, 'stop', 'buy', quantity, None, {'stopPrice': stop_loss}
                    )
                    tp_order = self.exchange.create_order(
                        symbol, 'limit', 'buy', quantity, take_profit
                    )
            
            # Save position
            self.positions[symbol] = {
                'symbol': symbol,
                'type': signal,
                'entry_price': price,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'order_id': order['id'] if 'id' in order else None,
                'sl_order_id': sl_order['id'] if 'id' in sl_order else None,
                'tp_order_id': tp_order['id'] if 'id' in tp_order else None,
            }
            
            # Update last trade time
            self.last_trade_time[symbol] = len(self.historical_data[symbol])
            
            # Update statistics
            self.performance['trades'] += 1
            
            logger.info(f"Successfully executed {signal} trade for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False
    
    def check_open_positions(self):
        """Check status of open positions and update"""
        if not self.positions:
            return
        
        for symbol, position in list(self.positions.items()):
            try:
                # Get current market price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Calculate current profit/loss
                if position['type'] == "BUY":
                    pnl_percent = (current_price - position['entry_price']) / position['entry_price'] * 100
                else:  # SELL
                    pnl_percent = (position['entry_price'] - current_price) / position['entry_price'] * 100
                
                position['current_price'] = current_price
                position['current_pnl'] = pnl_percent
                
                # Check if stop loss or take profit has been hit
                # Note: In a live system, you'd check order status instead
                if position['type'] == "BUY":
                    stop_hit = current_price <= position['stop_loss']
                    take_hit = current_price >= position['take_profit']
                else:  # SELL
                    stop_hit = current_price >= position['stop_loss']
                    take_hit = current_price <= position['take_profit']
                
                # If stop or take profit hit, close position
                if stop_hit or take_hit:
                    reason = "Stop Loss" if stop_hit else "Take Profit"
                    logger.info(f"Position {symbol} closed: {reason} at {current_price:.2f}, "
                               f"PnL: {pnl_percent:.2f}%")
                    
                    # Update statistics
                    self.trade_history.append({
                        'symbol': symbol,
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'pnl_percent': pnl_percent,
                        'entry_time': position['entry_time'],
                        'exit_time': datetime.now(),
                        'exit_reason': reason
                    })
                    
                    # Update performance
                    if pnl_percent > 0:
                        self.performance['wins'] += 1
                        self.performance['total_profit'] += pnl_percent
                    else:
                        self.performance['losses'] += 1
                        self.performance['total_loss'] += abs(pnl_percent)
                    
                    # Update balance
                    trade_pnl = position['quantity'] * position['entry_price'] * pnl_percent / 100
                    self.performance['current_balance'] += trade_pnl
                    
                    # Remove position
                    del self.positions[symbol]
                
            except Exception as e:
                logger.error(f"Error checking position for {symbol}: {str(e)}")
    
    def update_data(self):
        """Update market data for all symbols"""
        for symbol in self.symbols:
            # Fetch latest data
            df = self.fetch_historical_data(symbol)
            
            if df.empty:
                logger.warning(f"No data retrieved for {symbol}")
                continue
            
            # Convert column names to lowercase for consistency
            df.columns = df.columns.str.lower()
            
            # Store updated data
            self.historical_data[symbol] = df
            
            # Update indicators
            self.indicators[symbol] = self.calculate_indicators(df)
            
            logger.debug(f"Updated data and indicators for {symbol}")
    
    def print_status(self):
        """Print current trading status"""
        logger.info("\n----- Quantum RSI Live Trader Status -----")
        logger.info(f"Current balance: ${self.performance['current_balance']:.2f}")
        logger.info(f"Total trades: {self.performance['trades']}")
        logger.info(f"Win rate: {(self.performance['wins'] / self.performance['trades'] * 100):.2f}% "
                   f"({self.performance['wins']}/{self.performance['trades']})") if self.performance['trades'] > 0 else logger.info("Win rate: N/A")
        logger.info(f"Profit factor: {(self.performance['total_profit'] / self.performance['total_loss']):.2f}") if self.performance['total_loss'] > 0 else logger.info("Profit factor: N/A")
        logger.info(f"Open positions: {len(self.positions)}")
        
        for symbol, position in self.positions.items():
            logger.info(f"  {symbol}: {position['type']} @ {position['entry_price']:.2f}, "
                       f"Current: {position['current_price']:.2f} ({position['current_pnl']:.2f}%)")
        
        logger.info("-------------------------------------------")
    
    def save_state(self, filepath="quantum_rsi_state.json"):
        """Save current state to file"""
        state = {
            'performance': self.performance,
            'trade_history': self.trade_history,
            'positions': {k: {kk: str(vv) if isinstance(vv, datetime) else vv 
                             for kk, vv in v.items()} 
                         for k, v in self.positions.items()},
            'last_trade_time': self.last_trade_time,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath="quantum_rsi_state.json"):
        """Load state from file"""
        if not os.path.exists(filepath):
            logger.warning(f"State file {filepath} not found")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.performance = state['performance']
            self.trade_history = state['trade_history']
            self.last_trade_time = state['last_trade_time']
            
            # Convert positions back to appropriate format
            self.positions = {}
            for symbol, pos in state['positions'].items():
                pos_copy = pos.copy()
                if 'entry_time' in pos_copy:
                    pos_copy['entry_time'] = datetime.fromisoformat(pos_copy['entry_time'])
                self.positions[symbol] = pos_copy
            
            logger.info(f"State loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False
    
    def run(self, interval=60):
        """Run the live trading loop"""
        logger.info(f"Starting Quantum RSI Live Trader with {len(self.symbols)} symbols")
        logger.info(f"Using {'TESTNET' if self.use_testnet else 'LIVE'} mode")
        
        # Initial data fetch
        self.update_data()
        
        try:
            # Main trading loop
            while True:
                # Update market data
                self.update_data()
                
                # Check open positions
                self.check_open_positions()
                
                # Check for new signals
                for symbol in self.symbols:
                    signal = self.check_for_signals(symbol)
                    if signal:
                        logger.info(f"Signal detected: {signal['signal']} {symbol} @ {signal['price']:.2f}")
                        success = self.execute_trade(signal)
                        if success:
                            logger.info(f"Trade executed successfully")
                        else:
                            logger.warning(f"Failed to execute trade")
                
                # Print status
                self.print_status()
                
                # Save state
                self.save_state()
                
                # Wait for next iteration
                logger.info(f"Waiting {interval} seconds until next update...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
        finally:
            # Final state save
            self.save_state()
            logger.info("Trading stopped, final state saved")

# ------ Main Execution ------

if __name__ == "__main__":
    # Configuration setup - you can customize your parameters here
    config = {
        # Strategy parameters
        'rsi_period': 14,
        'rsi_high': 65,          # Lower than standard for crypto markets
        'rsi_low': 35,           # Higher than standard for crypto markets
        
        # Quantum model parameters (optimized for crypto)
        'v0': 1.5,               # Deeper wells for crypto trends
        'a': 0.5,                # Wider separation for larger price swings
        'mass': 1.0,             # Effective mass (trend inertia)
        'hbar': 1.0,             # Reduced Planck's constant analog
        
        # Demon parameters
        'volume_threshold': 2.0,  # Volume spike detection
        'vwap_period': 20,        # Period for VWAP calculation
        
        # Risk management
        'stop_loss_atr': 2.0,     # Stop loss multiplier based on ATR
        'take_profit_atr': 4.0,   # Take profit multiplier based on ATR
        'atr_period': 14,         # ATR period
        'cooldown_bars': 6,       # Bars to wait after a trade
        
        # Order parameters
        'order_type': 'market',   # 'market' or 'limit'
        'limit_offset': 0.001,    # 0.1% below/above for limit orders
    }
    
    # You would replace these with your actual API keys
    API_KEY = "your_api_key"
    API_SECRET = "your_api_secret"
    
    # Create and run the trader
    trader = QuantumRSILiveTrader(
        symbols=['BTC/USDT', 'ETH/USDT'],
        timeframe='1h',
        api_key=API_KEY,
        api_secret=API_SECRET,
        exchange_id='binance',
        use_testnet=True,  # Use testnet for paper trading
        initial_balance=10000,
        max_position_size=0.03,  # 3% max position size
        config=config
    )
    
    # Run with 5-minute check interval
    trader.run(interval=300) 