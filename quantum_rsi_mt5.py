"""
Quantum RSI Trading Strategy for MetaTrader 5

This module implements an optimized version of the Quantum RSI strategy
with Maxwell's Demon filter for live trading via MetaTrader 5.
"""

import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os
import MetaTrader5 as mt5

# Ensure results directory exists
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configure logging
import logging
logging.basicConfig(
    filename=f"logs/quantum_rsi_mt5_{datetime.now().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        
        # Trading parameters
        self.lot_size = 0.01  # Base lot size
        self.max_risk_pct = 1.0  # Maximum risk per trade (% of account)

class QuantumRSIMT5Trader:
    def __init__(self, symbol, timeframe=mt5.TIMEFRAME_H1, params=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params if params else QuantumRSIParameters()
        self.consecutive_losses = 0
        self.cooldown_counter = 0
        self.active_position = False
        self.last_trade_result = None
        self.active_positions = {}
        
        # MT5 timeframe conversion
        self.mt5_timeframes = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1
        }
        
        # Initialize MT5 connection
        self.connect()
    
    def connect(self):
        """Connect to MetaTrader 5"""
        if not mt5.initialize():
            logging.error(f"MT5 initialization failed: {mt5.last_error()}")
            raise Exception(f"MT5 initialization failed: {mt5.last_error()}")
        
        logging.info("MT5 successfully initialized")
        
        # Check if symbol is available
        if self.symbol not in mt5.symbols_get():
            logging.error(f"Symbol {self.symbol} not found in MT5")
            raise Exception(f"Symbol {self.symbol} not found in MT5")
    
    def fetch_data(self, bars=500):
        """Fetch historical price data from MT5"""
        # Calculate from_date based on the number of bars
        from_date = datetime.now() - timedelta(days=bars)
        
        # Get rates
        rates = mt5.copy_rates_from(self.symbol, self.timeframe, from_date, bars)
        
        if rates is None:
            logging.error(f"Failed to get data for {self.symbol}: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rename columns to match our strategy's requirements
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'spread': 'spread',
            'real_volume': 'real_volume'
        }, inplace=True)
        
        return df
    
    def calculate_indicators(self, df):
        """Calculate all indicators needed for the strategy"""
        # Basic indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.params.rsi_period)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.params.atr_period)
        
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
        df['barrier_height'] = self.compute_barrier_height(df)
        df['tunneling_probability'] = self.compute_tunneling_probability(df)
        
        # Volume delta approximation
        df['volume_delta'] = df['volume'] * np.sign(df['close'] - df['open'])
        df['delta_momentum'] = df['volume_delta'].rolling(window=5).sum()
        
        return df
    
    def compute_barrier_height(self, df):
        """
        Dynamically compute potential barrier height based on market conditions.
        Lower barrier = easier trend reversal, higher barrier = stronger trend persistence.
        """
        # Base barrier height from parameters
        barrier = np.ones(len(df)) * self.params.v0
        
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
    
    def compute_tunneling_probability(self, df):
        """
        Calculate quantum tunneling probability for trend reversal based on the
        WKB approximation from quantum mechanics.
        """
        # Map RSI to position coordinate x
        x = df['x']
        
        # Energy relative to barrier height (RSI momentum)
        # Energy is higher when RSI approaches middle from extreme
        energy = self.params.mass * (1.0 - x**2)
        
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
        kappa = np.sqrt(2 * self.params.mass * energy_deficit) / self.params.hbar
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
    
    def maxwell_demon_filter(self, df, index):
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
        volume_spike = current['volume_ratio'] > self.params.volume_threshold
        
        # 2. Check for price-VWAP relationship
        extended_price = abs(current['vwap_distance']) > self.params.vwap_deviation
        
        # 3. Check for delta volume
        delta_change = (np.sign(current['delta_momentum']) != 
                       np.sign(df.iloc[index-5:index]['delta_momentum'].mean()))
        
        # Delta confirms potential reversal when it shows exhaustion or reversal
        # For buy: delta turning positive in oversold
        # For sell: delta turning negative in overbought
        delta_confirmation = False
        if current['rsi'] < self.params.rsi_low and current['delta_momentum'] > 0:
            delta_confirmation = True  # Bullish confirmation
        elif current['rsi'] > self.params.rsi_high and current['delta_momentum'] < 0:
            delta_confirmation = True  # Bearish confirmation
        
        # 4. Check for RSI momentum change (reversal indication)
        rsi_change = (np.sign(current['rsi'] - prev['rsi']) != 
                      np.sign(prev['rsi'] - df.iloc[index-2]['rsi']))
        
        # === Apply Demon Logic - Information-Based Barrier Adjustment ===
        
        # Bullish scenario (oversold, looking for long)
        if current['rsi'] < self.params.rsi_low:
            # Adjust barrier based on information
            adjusted_barrier = self.params.v0  # Start with base
            
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
        elif current['rsi'] > self.params.rsi_high:
            # Adjust barrier based on information
            adjusted_barrier = self.params.v0  # Start with base
            
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
    
    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate position size based on risk parameters"""
        account_info = mt5.account_info()
        if not account_info:
            logging.error("Failed to get account info")
            return self.params.lot_size
        
        # Get account balance/equity
        account_equity = account_info.equity
        
        # Calculate risk amount in account currency
        risk_amount = account_equity * (self.params.max_risk_pct / 100)
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            return self.params.lot_size
        
        # Get symbol info
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            logging.error(f"Failed to get symbol info for {self.symbol}")
            return self.params.lot_size
        
        # Calculate position size based on risk
        contract_size = symbol_info.trade_contract_size
        price_per_pip = contract_size / 10000  # For forex
        
        # Adjust for non-forex assets
        if 'USD' not in self.symbol:
            price_per_pip = contract_size * entry_price / 10000
        
        # Calculate lot size based on risk
        lot_size = risk_amount / (risk_per_unit * price_per_pip * 10000)
        
        # Normalize to symbol minimum lot
        min_lot = symbol_info.volume_min
        lot_step = symbol_info.volume_step
        lot_size = round(lot_size / lot_step) * lot_step
        
        # Ensure minimum and maximum lot sizes
        lot_size = max(min_lot, min(lot_size, symbol_info.volume_max))
        
        logging.info(f"Calculated position size: {lot_size} lots (risk: {risk_amount:.2f}, risk per unit: {risk_per_unit:.5f})")
        
        return lot_size
    
    def place_order(self, order_type, entry_price, stop_loss, take_profit):
        """Place an order in MT5"""
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            logging.error(f"Failed to get symbol info for {self.symbol}")
            return False
        
        # Prepare order
        if order_type == "BUY":
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.symbol).ask
        else:  # "SELL"
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(self.symbol).bid
        
        # Calculate lot size based on risk
        lot_size = self.calculate_position_size(entry_price, stop_loss)
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": trade_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,
            "magic": 123456,  # Magic number for identifying trades
            "comment": f"Quantum RSI {order_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Execute trade
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed, retcode={result.retcode}, {result.comment}")
            return False
        
        logging.info(f"Order {order_type} successful: ticket={result.order}")
        
        # Store position info
        self.active_position = True
        self.active_positions[result.order] = {
            "ticket": result.order,
            "type": order_type,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "lot_size": lot_size,
            "entry_time": datetime.now()
        }
        
        return True
    
    def check_positions(self):
        """Check current positions and update status"""
        positions = mt5.positions_get(symbol=self.symbol)
        
        # If no positions but we think we have active ones
        if not positions and self.active_position:
            self.update_closed_positions()
            return
        
        # Update position info
        for position in positions:
            position_id = position.ticket
            
            # Check if it's our position
            if position_id in self.active_positions:
                # Update with current data
                if position.profit > 0:
                    logging.info(f"Position {position_id} in profit: {position.profit:.2f}")
                elif position.profit < 0:
                    logging.info(f"Position {position_id} in loss: {position.profit:.2f}")
    
    def update_closed_positions(self):
        """Update status after positions are closed"""
        # Get recent trade history
        from_date = datetime.now() - timedelta(days=1)
        history = mt5.history_deals_get(from_date, datetime.now())
        
        if not history:
            logging.warning("No trade history found, but positions seem to be closed")
            self.active_position = False
            self.active_positions = {}
            return
        
        # Process deals to find our closed positions
        for deal in history:
            if deal.position_id in self.active_positions:
                position = self.active_positions[deal.position_id]
                
                # Check if it's a close deal
                if deal.entry == 1:  # Exit position
                    profit = deal.profit
                    
                    # Record trade result
                    self.last_trade_result = {
                        "ticket": deal.position_id,
                        "type": "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL",
                        "profit": profit,
                        "entry_price": position["entry_price"],
                        "exit_price": deal.price,
                        "entry_time": position["entry_time"],
                        "exit_time": datetime.fromtimestamp(deal.time),
                        "profit_pct": profit / (position["entry_price"] * position["lot_size"] * 100)
                    }
                    
                    # Update consecutive losses
                    if profit < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0
                    
                    # Set cooldown period
                    self.cooldown_counter = self.params.cooldown_period
                    
                    # Log trade result
                    logging.info(f"Position {deal.position_id} closed: {profit:.2f}, consecutive losses: {self.consecutive_losses}")
                    
                    # Remove from active positions
                    self.active_positions.pop(deal.position_id, None)
        
        # Reset if no active positions
        if not self.active_positions:
            self.active_position = False
    
    def check_for_signals(self):
        """Check for new trading signals"""
        # Skip if in cooldown or have active positions
        if self.cooldown_counter > 0 or self.active_position:
            self.cooldown_counter = max(0, self.cooldown_counter - 1)
            return
        
        # Get latest data
        df = self.fetch_data()
        if df is None or len(df) < 50:
            logging.error("Not enough data for signal generation")
            return
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get latest bar
        current_index = len(df) - 1
        current = df.iloc[current_index]
        prev = df.iloc[current_index - 1]
        
        # Apply Maxwell's Demon filter
        approved, confidence, adjusted_barrier = self.maxwell_demon_filter(df, current_index)
        
        # Adjust RSI thresholds based on consecutive losses
        adjusted_rsi_high = self.params.rsi_high + (self.consecutive_losses * 2)
        adjusted_rsi_low = self.params.rsi_low - (self.consecutive_losses * 2)
        
        # Calculate tunneling probability
        tunneling_prob = current['tunneling_probability']
        current_price = current['close']
        
        # Check for buy signal
        if (current['rsi'] < adjusted_rsi_low and 
            current['rsi'] > prev['rsi'] and  # RSI turning up
            approved and 
            tunneling_prob > 0.3):  # Sufficient tunneling probability
            
            # Calculate stop loss and take profit
            stop_loss = current_price * (1 - self.params.stop_loss_atr * current['atr'] / current_price)
            take_profit = current_price * (1 + self.params.take_profit_atr * current['atr'] / current_price)
            
            # Place buy order
            logging.info(f"BUY signal: RSI={current['rsi']:.2f}, TP={confidence}, Prob={tunneling_prob:.2f}")
            self.place_order("BUY", current_price, stop_loss, take_profit)
        
        # Check for sell signal
        elif (current['rsi'] > adjusted_rsi_high and 
              current['rsi'] < prev['rsi'] and  # RSI turning down
              approved and 
              tunneling_prob > 0.3):  # Sufficient tunneling probability
            
            # Calculate stop loss and take profit
            stop_loss = current_price * (1 + self.params.stop_loss_atr * current['atr'] / current_price)
            take_profit = current_price * (1 - self.params.take_profit_atr * current['atr'] / current_price)
            
            # Place sell order
            logging.info(f"SELL signal: RSI={current['rsi']:.2f}, TP={confidence}, Prob={tunneling_prob:.2f}")
            self.place_order("SELL", current_price, stop_loss, take_profit)
    
    def run_trading_loop(self, interval_seconds=60):
        """Main trading loop"""
        logging.info(f"Starting trading loop for {self.symbol} on {self.timeframe} timeframe")
        try:
            while True:
                # Check any open positions
                self.check_positions()
                
                # Look for new signals
                self.check_for_signals()
                
                # Wait for next check
                logging.info(f"Sleeping for {interval_seconds} seconds...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logging.info("Trading loop stopped by user")
        except Exception as e:
            logging.error(f"Error in trading loop: {e}")
        finally:
            mt5.shutdown()
            logging.info("MT5 connection closed")

def main():
    # Import numpy patch for bool8 issue
    import numpy_patch
    
    # Symbol to trade
    symbol = "EURUSD"
    
    # Set conservative parameters
    params = QuantumRSIParameters(market_type="forex")
    params.rsi_high = 75
    params.rsi_low = 25
    params.volume_threshold = 2.0
    params.lot_size = 0.01
    params.max_risk_pct = 0.5  # Start with low risk
    
    # Create trader instance
    trader = QuantumRSIMT5Trader(symbol, mt5.TIMEFRAME_H1, params)
    
    # Run trading loop
    print(f"Starting Quantum RSI trader for {symbol}")
    trader.run_trading_loop()

if __name__ == "__main__":
    main() 