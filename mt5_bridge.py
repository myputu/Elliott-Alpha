"""
MetaTrader 5 Bridge for Python Trading System
This module provides a bridge between the Python Elliott Wave trading system and MetaTrader 5.
It enables real-time data acquisition and automated trade execution.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from database import OHLCDatabase
from elliott_wave_analyzer import ElliottWaveAnalyzer
from trading_strategy import ElliottWaveTradingStrategy
from ai_models import AlphaGoModel, SelfPlayModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5Bridge:
    """
    Bridge class to connect Python trading system with MetaTrader 5.
    Handles data acquisition, trade execution, and real-time monitoring.
    """
    
    def __init__(self, symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M1, magic_number=123456):
        """
        Initialize MT5 Bridge.
        
        Args:
            symbol (str): Trading symbol (default: XAUUSD)
            timeframe: MT5 timeframe constant (default: M1)
            magic_number (int): Magic number for trade identification
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.magic_number = magic_number
        self.is_connected = False
        self.db = OHLCDatabase()
        
        # Trading system components
        self.analyzer = None
        self.strategy = None
        self.alphago_model = AlphaGoModel()
        self.self_play_model = SelfPlayModel()
        
        # Trade management
        self.active_trades = {}
        self.trade_history = []
        
    def connect(self, login=None, password=None, server=None):
        """
        Connect to MetaTrader 5 terminal.
        
        Args:
            login (int): MT5 account login (optional if already logged in)
            password (str): MT5 account password (optional)
            server (str): MT5 server name (optional)
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Initialize MT5 connection
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login if credentials provided
            if login and password and server:
                if not mt5.login(login, password, server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
            
            # Check connection
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False
            
            self.is_connected = True
            logger.info(f"Connected to MT5. Account: {account_info.login}, Balance: {account_info.balance}")
            
            # Check if symbol is available
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found")
                return False
            
            # Enable symbol for trading
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Failed to select symbol {self.symbol}")
                return False
            
            logger.info(f"Symbol {self.symbol} selected for trading")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from MetaTrader 5."""
        if self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            logger.info("Disconnected from MT5")
    
    def get_market_data(self, count=1000):
        """
        Get historical market data from MT5.
        
        Args:
            count (int): Number of bars to retrieve
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        if not self.is_connected:
            logger.error("Not connected to MT5")
            return None
        
        try:
            # Get rates
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)
            
            if rates is None:
                logger.error(f"Failed to get rates: {mt5.last_error()}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns to match our system
            df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)
            
            logger.info(f"Retrieved {len(df)} bars for {self.symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return None
    
    def get_current_price(self):
        """
        Get current bid/ask prices.
        
        Returns:
            dict: Current price information
        """
        if not self.is_connected:
            return None
        
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return None
            
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'time': datetime.fromtimestamp(tick.time)
            }
            
        except Exception as e:
            logger.error(f"Error getting current price: {str(e)}")
            return None
    
    def place_order(self, order_type, volume, price=None, sl=None, tp=None, comment="Elliott Wave Trade"):
        """
        Place a trading order.
        
        Args:
            order_type (str): 'buy' or 'sell'
            volume (float): Trade volume
            price (float): Entry price (None for market order)
            sl (float): Stop loss price
            tp (float): Take profit price
            comment (str): Order comment
            
        Returns:
            dict: Order result
        """
        if not self.is_connected:
            logger.error("Not connected to MT5")
            return None
        
        try:
            # Get current price for market orders
            current_price = self.get_current_price()
            if current_price is None:
                logger.error("Failed to get current price")
                return None
            
            # Determine order type and price
            if order_type.lower() == 'buy':
                trade_type = mt5.ORDER_TYPE_BUY
                execution_price = current_price['ask'] if price is None else price
            else:
                trade_type = mt5.ORDER_TYPE_SELL
                execution_price = current_price['bid'] if price is None else price
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": trade_type,
                "price": execution_price,
                "magic": self.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit if provided
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return None
            
            logger.info(f"Order placed successfully: {result.order}")
            
            # Store trade information
            trade_info = {
                'ticket': result.order,
                'type': order_type,
                'volume': volume,
                'price': result.price,
                'sl': sl,
                'tp': tp,
                'time': datetime.now(),
                'comment': comment
            }
            
            self.active_trades[result.order] = trade_info
            return trade_info
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    def close_position(self, ticket):
        """
        Close a position by ticket number.
        
        Args:
            ticket (int): Position ticket number
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                logger.error(f"Position {ticket} not found")
                return False
            
            position = position[0]
            
            # Determine close order type
            if position.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(self.symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(self.symbol).ask
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "magic": self.magic_number,
                "comment": "Close by Elliott Wave System",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Close order failed: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"Position {ticket} closed successfully")
            
            # Move to trade history
            if ticket in self.active_trades:
                trade_info = self.active_trades[ticket]
                trade_info['close_time'] = datetime.now()
                trade_info['close_price'] = result.price
                self.trade_history.append(trade_info)
                del self.active_trades[ticket]
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return False
    
    def get_positions(self):
        """
        Get all open positions.
        
        Returns:
            list: List of open positions
        """
        if not self.is_connected:
            return []
        
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return []
            
            return [
                {
                    'ticket': pos.ticket,
                    'type': 'buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
                    'volume': pos.volume,
                    'price': pos.price_open,
                    'current_price': pos.price_current,
                    'profit': pos.profit,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'time': datetime.fromtimestamp(pos.time)
                }
                for pos in positions
            ]
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def run_trading_system(self, update_interval=60):
        """
        Run the Elliott Wave trading system with real-time data.
        
        Args:
            update_interval (int): Update interval in seconds
        """
        if not self.is_connected:
            logger.error("Not connected to MT5")
            return
        
        logger.info("Starting Elliott Wave trading system...")
        
        try:
            while True:
                # Get latest market data
                market_data = self.get_market_data(count=1000)
                if market_data is None:
                    time.sleep(update_interval)
                    continue
                
                # Store data in database
                self.db.insert_ohlc_data(self.symbol, market_data)
                
                # Initialize analyzer and strategy with latest data
                self.analyzer = ElliottWaveAnalyzer(market_data)
                self.strategy = ElliottWaveTradingStrategy(self.analyzer)
                
                # Run Elliott Wave analysis
                self.analyzer.run_analysis()
                
                # Generate trading signals
                signals = self.strategy.generate_signals(self.analyzer.waves)
                
                # Process signals
                for signal in signals:
                    self.process_signal(signal)
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Wait for next update
                logger.info(f"Waiting {update_interval} seconds for next update...")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            logger.info("Trading system stopped by user")
        except Exception as e:
            logger.error(f"Trading system error: {str(e)}")
        finally:
            self.disconnect()
    
    def process_signal(self, signal):
        """
        Process a trading signal and place orders if conditions are met.
        
        Args:
            signal (dict): Trading signal from strategy
        """
        try:
            # Get current price
            current_price = self.get_current_price()
            if current_price is None:
                return
            
            # Calculate position size (2% risk per trade)
            account_info = mt5.account_info()
            if account_info is None:
                return
            
            risk_amount = account_info.balance * 0.02
            price_risk = abs(signal['price'] - signal['stop_loss'])
            
            if price_risk > 0:
                # Calculate volume (simplified, should consider symbol specifications)
                volume = min(risk_amount / price_risk / 100000, 1.0)  # Max 1 lot
                volume = max(volume, 0.01)  # Min 0.01 lot
                
                # Place order
                order_result = self.place_order(
                    order_type=signal['type'],
                    volume=volume,
                    price=signal['price'],
                    sl=signal['stop_loss'],
                    tp=signal['take_profit'],
                    comment=f"Elliott Wave - {signal.get('pattern', 'Unknown')}"
                )
                
                if order_result:
                    logger.info(f"Signal processed: {signal['type']} {volume} lots at {signal['price']}")
                
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
    
    def monitor_positions(self):
        """Monitor existing positions and apply trade management rules."""
        try:
            positions = self.get_positions()
            
            for position in positions:
                # Example: Trail stop loss for profitable trades
                if position['profit'] > 0:
                    current_price = self.get_current_price()
                    if current_price is None:
                        continue
                    
                    # Simple trailing stop logic (can be enhanced)
                    if position['type'] == 'buy':
                        new_sl = current_price['bid'] - 0.001  # 10 pips trail
                        if new_sl > position['sl']:
                            self.modify_position(position['ticket'], sl=new_sl)
                    else:
                        new_sl = current_price['ask'] + 0.001  # 10 pips trail
                        if new_sl < position['sl']:
                            self.modify_position(position['ticket'], sl=new_sl)
                            
        except Exception as e:
            logger.error(f"Error monitoring positions: {str(e)}")
    
    def modify_position(self, ticket, sl=None, tp=None):
        """
        Modify position stop loss or take profit.
        
        Args:
            ticket (int): Position ticket
            sl (float): New stop loss
            tp (float): New take profit
            
        Returns:
            bool: True if successful
        """
        if not self.is_connected:
            return False
        
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": self.symbol,
                "position": ticket,
                "magic": self.magic_number,
            }
            
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Modify failed: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"Position {ticket} modified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying position: {str(e)}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize MT5 Bridge
    bridge = MT5Bridge(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M1)
    
    # Connect to MT5 (you'll need to provide actual credentials)
    if bridge.connect():
        print("Connected to MT5 successfully")
        
        # Get some market data
        data = bridge.get_market_data(count=100)
        if data is not None:
            print(f"Retrieved {len(data)} bars")
            print(data.head())
        
        # Get current price
        price = bridge.get_current_price()
        if price:
            print(f"Current price: Bid={price['bid']}, Ask={price['ask']}")
        
        # Get positions
        positions = bridge.get_positions()
        print(f"Open positions: {len(positions)}")
        
        # Disconnect
        bridge.disconnect()
    else:
        print("Failed to connect to MT5")

