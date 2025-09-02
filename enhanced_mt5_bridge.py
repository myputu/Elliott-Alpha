"""
Enhanced MetaTrader 5 Bridge with Asynchronous Communication
This module provides high-performance bridge between Python Elliott Wave trading system 
and MetaTrader 5 with asynchronous tick data handling and buffering capabilities.

Author: Manus AI
Date: 28 Agustus 2025
Version: 2.0 (Enhanced)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
from dataclasses import dataclass
import queue

# Import existing components (maintaining compatibility)
from database import OHLCDatabase
from elliott_wave_analyzer import ElliottWaveAnalyzer
from trading_strategy import ElliottWaveTradingStrategy
from ai_models import AlphaGoModel, SelfPlayModel

# Import new async buffer system
from async_tick_buffer import AsyncTickBuffer, TickData, OHLCData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    """Enhanced trade signal with priority and timing information."""
    symbol: str
    action: str  # 'buy', 'sell', 'close'
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    priority: str = 'normal'  # 'critical', 'high', 'normal', 'low'
    timestamp: datetime = None
    pattern: str = ""
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent system overload.
    Monitors error rates and temporarily disables operations if threshold exceeded.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class EnhancedMT5Bridge:
    """
    Enhanced MT5 Bridge with asynchronous communication and high-performance buffering.
    Maintains backward compatibility with existing MT5Bridge interface.
    """
    
    def __init__(self, symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M1, magic_number=123456,
                 enable_async_buffer=True, buffer_config=None):
        """
        Initialize Enhanced MT5 Bridge.
        
        Args:
            symbol (str): Trading symbol
            timeframe: MT5 timeframe constant
            magic_number (int): Magic number for trade identification
            enable_async_buffer (bool): Enable asynchronous buffer system
            buffer_config (dict): Configuration for async buffer
        """
        # Core MT5 Bridge properties (maintaining compatibility)
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
        
        # Enhanced features
        self.enable_async_buffer = enable_async_buffer
        self.async_buffer = None
        self.circuit_breaker = CircuitBreaker()
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="MT5Bridge")
        self._shutdown_event = threading.Event()
        self._worker_threads = {}
        
        # Performance monitoring
        self._performance_metrics = {
            'tick_processing_rate': 0.0,
            'order_execution_time': 0.0,
            'data_latency': 0.0,
            'error_rate': 0.0,
            'uptime': datetime.now()
        }
        
        # Signal processing queue
        self._signal_queue = queue.PriorityQueue(maxsize=1000)
        
        # Initialize async buffer if enabled
        if self.enable_async_buffer:
            buffer_config = buffer_config or {}
            self.async_buffer = AsyncTickBuffer(
                tick_buffer_size=buffer_config.get('tick_buffer_size', 50000),
                ohlc_buffer_size=buffer_config.get('ohlc_buffer_size', 10000),
                queue_size=buffer_config.get('queue_size', 5000),
                max_workers=buffer_config.get('max_workers', 4)
            )
            
            # Register callbacks for async buffer
            self.async_buffer.register_tick_callback(self._on_tick_received)
            self.async_buffer.register_ohlc_callback(self._on_ohlc_received)
            
            logger.info("Enhanced MT5 Bridge initialized with async buffer")
        else:
            logger.info("Enhanced MT5 Bridge initialized without async buffer")
    
    def connect(self, login=None, password=None, server=None):
        """
        Connect to MetaTrader 5 terminal with enhanced error handling.
        Maintains compatibility with original connect method.
        """
        try:
            return self.circuit_breaker.call(self._connect_internal, login, password, server)
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _connect_internal(self, login=None, password=None, server=None):
        """Internal connection method with circuit breaker protection."""
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
            
            # Start enhanced workers
            self._start_enhanced_workers()
            
            logger.info(f"Enhanced MT5 Bridge connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False
    
    def _start_enhanced_workers(self):
        """Start enhanced background workers for high-performance operation."""
        if not self.is_connected:
            return
        
        # Real-time tick data collector
        self._worker_threads['tick_collector'] = threading.Thread(
            target=self._tick_collection_worker,
            name="TickCollector",
            daemon=True
        )
        self._worker_threads['tick_collector'].start()
        
        # Signal processing worker
        self._worker_threads['signal_processor'] = threading.Thread(
            target=self._signal_processing_worker,
            name="SignalProcessor",
            daemon=True
        )
        self._worker_threads['signal_processor'].start()
        
        # Performance monitoring worker
        self._worker_threads['performance_monitor'] = threading.Thread(
            target=self._performance_monitoring_worker,
            name="PerformanceMonitor",
            daemon=True
        )
        self._worker_threads['performance_monitor'].start()
        
        logger.info("Enhanced workers started")
    
    def _tick_collection_worker(self):
        """High-frequency tick data collection worker."""
        last_tick_time = None
        tick_count = 0
        start_time = time.time()
        
        while not self._shutdown_event.is_set() and self.is_connected:
            try:
                # Get current tick
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is None:
                    time.sleep(0.001)  # 1ms sleep on no data
                    continue
                
                current_tick_time = tick.time
                
                # Only process new ticks
                if last_tick_time is None or current_tick_time > last_tick_time:
                    last_tick_time = current_tick_time
                    tick_count += 1
                    
                    # Add to async buffer if enabled
                    if self.async_buffer:
                        self.async_buffer.add_tick_data(
                            symbol=self.symbol,
                            bid=tick.bid,
                            ask=tick.ask,
                            timestamp=datetime.fromtimestamp(current_tick_time),
                            volume=getattr(tick, 'volume', 0),
                            priority='high'
                        )
                    
                    # Update performance metrics
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        self._performance_metrics['tick_processing_rate'] = tick_count / elapsed_time
                
                # Adaptive sleep based on market activity
                time.sleep(0.001)  # 1ms base interval for high-frequency collection
                
            except Exception as e:
                logger.error(f"Tick collection error: {e}")
                self._performance_metrics['error_rate'] += 1
                time.sleep(0.01)  # 10ms pause on error
    
    def _signal_processing_worker(self):
        """Process trading signals from priority queue."""
        while not self._shutdown_event.is_set():
            try:
                # Get signal from priority queue (blocking with timeout)
                try:
                    priority, signal = self._signal_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process signal with circuit breaker protection
                start_time = time.time()
                try:
                    self.circuit_breaker.call(self._execute_signal, signal)
                    execution_time = time.time() - start_time
                    
                    # Update performance metrics
                    alpha = 0.1  # Exponential moving average factor
                    current_avg = self._performance_metrics['order_execution_time']
                    self._performance_metrics['order_execution_time'] = (
                        alpha * execution_time + (1 - alpha) * current_avg
                    )
                    
                except Exception as e:
                    logger.error(f"Signal execution failed: {e}")
                    self._performance_metrics['error_rate'] += 1
                
                # Mark task as done
                self._signal_queue.task_done()
                
            except Exception as e:
                logger.error(f"Signal processing worker error: {e}")
                time.sleep(0.1)
    
    def _performance_monitoring_worker(self):
        """Monitor and log performance metrics."""
        while not self._shutdown_event.is_set():
            try:
                # Calculate uptime
                uptime = datetime.now() - self._performance_metrics['uptime']
                
                # Log performance metrics every 5 minutes
                if int(time.time()) % 300 == 0:
                    logger.info("Enhanced MT5 Bridge Performance Metrics:")
                    logger.info(f"  Uptime: {uptime}")
                    logger.info(f"  Tick processing rate: {self._performance_metrics['tick_processing_rate']:.2f} ticks/sec")
                    logger.info(f"  Avg order execution time: {self._performance_metrics['order_execution_time']:.4f}s")
                    logger.info(f"  Data latency: {self._performance_metrics['data_latency']:.4f}s")
                    logger.info(f"  Error rate: {self._performance_metrics['error_rate']}")
                    logger.info(f"  Circuit breaker state: {self.circuit_breaker.state}")
                    
                    # Get async buffer stats if enabled
                    if self.async_buffer:
                        buffer_stats = self.async_buffer.get_performance_stats()
                        logger.info(f"  Buffer performance: {buffer_stats['buffer_performance']}")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(60)  # Longer pause on error
    
    def _on_tick_received(self, tick_data: TickData):
        """Callback for processing received tick data."""
        try:
            # Calculate data latency
            latency = (datetime.now() - tick_data.timestamp).total_seconds()
            
            # Update latency metric (exponential moving average)
            alpha = 0.1
            current_latency = self._performance_metrics['data_latency']
            self._performance_metrics['data_latency'] = (
                alpha * latency + (1 - alpha) * current_latency
            )
            
            # Trigger real-time analysis if conditions met
            if self._should_trigger_analysis(tick_data):
                self._trigger_realtime_analysis()
                
        except Exception as e:
            logger.error(f"Tick processing callback error: {e}")
    
    def _on_ohlc_received(self, ohlc_data: OHLCData):
        """Callback for processing received OHLC data."""
        try:
            # Store in database asynchronously
            self.executor.submit(self._store_ohlc_async, ohlc_data)
            
        except Exception as e:
            logger.error(f"OHLC processing callback error: {e}")
    
    def _should_trigger_analysis(self, tick_data: TickData) -> bool:
        """Determine if real-time analysis should be triggered."""
        # Trigger analysis on significant price movements
        if hasattr(self, '_last_analysis_price'):
            price_change = abs(tick_data.ask - self._last_analysis_price) / self._last_analysis_price
            return price_change > 0.001  # 0.1% price change threshold
        else:
            self._last_analysis_price = tick_data.ask
            return True
    
    def _trigger_realtime_analysis(self):
        """Trigger real-time Elliott Wave analysis."""
        if not self.async_buffer:
            return
        
        try:
            # Get recent tick data for analysis
            recent_ticks = self.async_buffer.get_latest_ticks(count=1000)
            if len(recent_ticks) < 100:  # Need minimum data for analysis
                return
            
            # Convert ticks to OHLC for analysis (simplified)
            ohlc_data = self._convert_ticks_to_ohlc(recent_ticks)
            
            # Run analysis in thread pool to avoid blocking
            future = self.executor.submit(self._run_analysis_async, ohlc_data)
            
            # Process result when available
            future.add_done_callback(self._on_analysis_complete)
            
        except Exception as e:
            logger.error(f"Real-time analysis trigger error: {e}")
    
    def _convert_ticks_to_ohlc(self, ticks: List[TickData], timeframe_seconds: int = 60) -> pd.DataFrame:
        """Convert tick data to OHLC format for analysis."""
        if not ticks:
            return pd.DataFrame()
        
        # Create DataFrame from ticks
        df = pd.DataFrame([
            {
                'timestamp': tick.timestamp,
                'price': (tick.bid + tick.ask) / 2,  # Mid price
                'volume': tick.volume
            }
            for tick in ticks
        ])
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Resample to OHLC
        ohlc = df['price'].resample(f'{timeframe_seconds}S').ohlc()
        volume = df['volume'].resample(f'{timeframe_seconds}S').sum()
        
        # Combine OHLC and volume
        result = pd.concat([ohlc, volume], axis=1)
        result.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Forward fill missing values
        result.fillna(method='ffill', inplace=True)
        result.dropna(inplace=True)
        
        return result
    
    def _run_analysis_async(self, ohlc_data: pd.DataFrame) -> List[TradeSignal]:
        """Run Elliott Wave analysis asynchronously."""
        try:
            if ohlc_data.empty:
                return []
            
            # Initialize analyzer with latest data
            analyzer = ElliottWaveAnalyzer(ohlc_data)
            strategy = ElliottWaveTradingStrategy(analyzer)
            
            # Run analysis
            analyzer.run_analysis()
            
            # Generate signals
            signals = strategy.generate_signals(analyzer.waves)
            
            # Convert to enhanced trade signals
            trade_signals = []
            for signal in signals:
                trade_signal = TradeSignal(
                    symbol=self.symbol,
                    action=signal.get('type', 'hold'),
                    volume=signal.get('volume', 0.01),
                    price=signal.get('price'),
                    stop_loss=signal.get('stop_loss'),
                    take_profit=signal.get('take_profit'),
                    priority='high',
                    pattern=signal.get('pattern', ''),
                    confidence=signal.get('confidence', 0.0)
                )
                trade_signals.append(trade_signal)
            
            return trade_signals
            
        except Exception as e:
            logger.error(f"Async analysis error: {e}")
            return []
    
    def _on_analysis_complete(self, future):
        """Callback when analysis is complete."""
        try:
            signals = future.result()
            
            # Process each signal
            for signal in signals:
                if signal.action in ['buy', 'sell']:
                    # Add to signal processing queue with priority
                    priority = self._get_signal_priority(signal)
                    self._signal_queue.put((priority, signal))
                    
        except Exception as e:
            logger.error(f"Analysis completion callback error: {e}")
    
    def _get_signal_priority(self, signal: TradeSignal) -> int:
        """Get numeric priority for signal (lower number = higher priority)."""
        priority_map = {
            'critical': 1,
            'high': 2,
            'normal': 3,
            'low': 4
        }
        
        base_priority = priority_map.get(signal.priority, 3)
        
        # Adjust priority based on confidence
        confidence_adjustment = int((1.0 - signal.confidence) * 2)
        
        return base_priority + confidence_adjustment
    
    def _execute_signal(self, signal: TradeSignal):
        """Execute trading signal with enhanced error handling."""
        try:
            if signal.action == 'buy':
                result = self.place_order(
                    order_type='buy',
                    volume=signal.volume,
                    price=signal.price,
                    sl=signal.stop_loss,
                    tp=signal.take_profit,
                    comment=f"Elliott Wave - {signal.pattern}"
                )
            elif signal.action == 'sell':
                result = self.place_order(
                    order_type='sell',
                    volume=signal.volume,
                    price=signal.price,
                    sl=signal.stop_loss,
                    tp=signal.take_profit,
                    comment=f"Elliott Wave - {signal.pattern}"
                )
            elif signal.action == 'close':
                # Close specific position or all positions
                positions = self.get_positions()
                for position in positions:
                    self.close_position(position['ticket'])
            
            logger.info(f"Signal executed: {signal.action} {signal.volume} {signal.symbol}")
            
        except Exception as e:
            logger.error(f"Signal execution error: {e}")
            raise e
    
    def _store_ohlc_async(self, ohlc_data: OHLCData):
        """Store OHLC data in database asynchronously."""
        try:
            # Convert to DataFrame format expected by database
            df = pd.DataFrame([{
                'open': ohlc_data.open,
                'high': ohlc_data.high,
                'low': ohlc_data.low,
                'close': ohlc_data.close,
                'volume': ohlc_data.volume
            }], index=[ohlc_data.timestamp])
            
            # Store in database
            self.db.insert_ohlc_data(ohlc_data.symbol, df)
            
        except Exception as e:
            logger.error(f"Async OHLC storage error: {e}")
    
    # Maintain compatibility with original MT5Bridge methods
    def disconnect(self):
        """Disconnect from MetaTrader 5 with enhanced cleanup."""
        logger.info("Shutting down Enhanced MT5 Bridge...")
        
        # Signal shutdown to all workers
        self._shutdown_event.set()
        
        # Wait for workers to finish
        for name, thread in self._worker_threads.items():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(f"Worker {name} did not shutdown gracefully")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True, timeout=10.0)
        
        # Shutdown async buffer
        if self.async_buffer:
            self.async_buffer.shutdown(timeout=10.0)
        
        # Disconnect from MT5
        if self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            logger.info("Disconnected from MT5")
        
        logger.info("Enhanced MT5 Bridge shutdown complete")
    
    def get_market_data(self, count=1000):
        """Get historical market data (maintains compatibility)."""
        if not self.is_connected:
            logger.error("Not connected to MT5")
            return None
        
        try:
            return self.circuit_breaker.call(self._get_market_data_internal, count)
        except Exception as e:
            logger.error(f"Market data retrieval failed: {e}")
            return None
    
    def _get_market_data_internal(self, count):
        """Internal market data retrieval with circuit breaker protection."""
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
        
        # Add to async buffer if enabled
        if self.async_buffer:
            for index, row in df.iterrows():
                self.async_buffer.add_ohlc_data(
                    symbol=self.symbol,
                    timestamp=index,
                    open_price=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    priority='normal'
                )
        
        logger.info(f"Retrieved {len(df)} bars for {self.symbol}")
        return df
    
    def get_current_price(self):
        """Get current bid/ask prices (maintains compatibility)."""
        if not self.is_connected:
            return None
        
        try:
            return self.circuit_breaker.call(self._get_current_price_internal)
        except Exception as e:
            logger.error(f"Current price retrieval failed: {e}")
            return None
    
    def _get_current_price_internal(self):
        """Internal current price retrieval."""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'time': datetime.fromtimestamp(tick.time)
        }
    
    def place_order(self, order_type, volume, price=None, sl=None, tp=None, comment="Elliott Wave Trade"):
        """Place trading order (maintains compatibility with enhanced error handling)."""
        if not self.is_connected:
            logger.error("Not connected to MT5")
            return None
        
        try:
            return self.circuit_breaker.call(
                self._place_order_internal, 
                order_type, volume, price, sl, tp, comment
            )
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return None
    
    def _place_order_internal(self, order_type, volume, price, sl, tp, comment):
        """Internal order placement with circuit breaker protection."""
        # Get current price for market orders
        current_price = self._get_current_price_internal()
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
    
    # Additional methods maintain compatibility with original MT5Bridge
    def close_position(self, ticket):
        """Close position (maintains compatibility)."""
        if not self.is_connected:
            return False
        
        try:
            return self.circuit_breaker.call(self._close_position_internal, ticket)
        except Exception as e:
            logger.error(f"Position close failed: {e}")
            return False
    
    def _close_position_internal(self, ticket):
        """Internal position closing with circuit breaker protection."""
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
            "comment": "Close by Enhanced Elliott Wave System",
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
    
    def get_positions(self):
        """Get all open positions (maintains compatibility)."""
        if not self.is_connected:
            return []
        
        try:
            return self.circuit_breaker.call(self._get_positions_internal)
        except Exception as e:
            logger.error(f"Get positions failed: {e}")
            return []
    
    def _get_positions_internal(self):
        """Internal get positions with circuit breaker protection."""
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
    
    def run_trading_system(self, update_interval=60):
        """
        Run enhanced Elliott Wave trading system with async capabilities.
        Maintains compatibility while providing enhanced performance.
        """
        if not self.is_connected:
            logger.error("Not connected to MT5")
            return
        
        logger.info("Starting Enhanced Elliott Wave trading system...")
        
        try:
            # If async buffer is enabled, use event-driven approach
            if self.async_buffer:
                logger.info("Running in async mode with real-time tick processing")
                
                # Main loop for periodic tasks
                while not self._shutdown_event.is_set():
                    try:
                        # Periodic maintenance tasks
                        self._perform_maintenance_tasks()
                        
                        # Monitor existing positions
                        self.monitor_positions()
                        
                        # Wait for next maintenance cycle
                        time.sleep(update_interval)
                        
                    except KeyboardInterrupt:
                        logger.info("Trading system stopped by user")
                        break
                    except Exception as e:
                        logger.error(f"Trading system error: {str(e)}")
                        time.sleep(10)  # Brief pause on error
            else:
                # Fallback to original synchronous approach
                logger.info("Running in synchronous mode")
                self._run_synchronous_trading_system(update_interval)
                
        except KeyboardInterrupt:
            logger.info("Trading system stopped by user")
        except Exception as e:
            logger.error(f"Trading system error: {str(e)}")
        finally:
            self.disconnect()
    
    def _perform_maintenance_tasks(self):
        """Perform periodic maintenance tasks."""
        try:
            # Update performance metrics
            self._update_performance_metrics()
            
            # Clean up old data if needed
            if self.async_buffer:
                # Get buffer stats and clean if necessary
                stats = self.async_buffer.get_performance_stats()
                tick_utilization = stats['tick_buffer_stats']['utilization']
                
                if tick_utilization > 90:  # 90% utilization threshold
                    logger.info("High buffer utilization detected, triggering cleanup")
                    # Could implement buffer cleanup logic here
            
            # Database maintenance
            self.executor.submit(self._database_maintenance)
            
        except Exception as e:
            logger.error(f"Maintenance task error: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Update uptime
            uptime = datetime.now() - self._performance_metrics['uptime']
            
            # Calculate error rate (errors per hour)
            hours = max(uptime.total_seconds() / 3600, 1)
            self._performance_metrics['error_rate'] = self._performance_metrics['error_rate'] / hours
            
        except Exception as e:
            logger.error(f"Performance metrics update error: {e}")
    
    def _database_maintenance(self):
        """Perform database maintenance tasks."""
        try:
            # Could implement database optimization, cleanup, etc.
            pass
        except Exception as e:
            logger.error(f"Database maintenance error: {e}")
    
    def _run_synchronous_trading_system(self, update_interval):
        """Fallback synchronous trading system (original behavior)."""
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
    
    def process_signal(self, signal):
        """Process trading signal (maintains compatibility)."""
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
        """Monitor existing positions (maintains compatibility)."""
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
        """Modify position (maintains compatibility)."""
        if not self.is_connected:
            return False
        
        try:
            return self.circuit_breaker.call(self._modify_position_internal, ticket, sl, tp)
        except Exception as e:
            logger.error(f"Position modification failed: {e}")
            return False
    
    def _modify_position_internal(self, ticket, sl, tp):
        """Internal position modification with circuit breaker protection."""
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
    
    def get_enhanced_stats(self) -> Dict:
        """Get comprehensive enhanced bridge statistics."""
        stats = {
            'performance_metrics': self._performance_metrics.copy(),
            'circuit_breaker_state': self.circuit_breaker.state,
            'circuit_breaker_failures': self.circuit_breaker.failure_count,
            'active_trades_count': len(self.active_trades),
            'trade_history_count': len(self.trade_history),
            'worker_threads_status': {
                name: thread.is_alive() 
                for name, thread in self._worker_threads.items()
            },
            'signal_queue_size': self._signal_queue.qsize()
        }
        
        # Add async buffer stats if enabled
        if self.async_buffer:
            stats['async_buffer_stats'] = self.async_buffer.get_performance_stats()
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    # Initialize Enhanced MT5 Bridge
    bridge = EnhancedMT5Bridge(
        symbol="XAUUSD", 
        timeframe=mt5.TIMEFRAME_M1,
        enable_async_buffer=True,
        buffer_config={
            'tick_buffer_size': 10000,
            'ohlc_buffer_size': 5000,
            'queue_size': 2000,
            'max_workers': 4
        }
    )
    
    # Connect to MT5 (you'll need to provide actual credentials)
    if bridge.connect():
        print("Connected to Enhanced MT5 Bridge successfully")
        
        try:
            # Get some market data
            data = bridge.get_market_data(count=100)
            if data is not None:
                print(f"Retrieved {len(data)} bars")
                print(data.head())
            
            # Get current price
            price = bridge.get_current_price()
            if price:
                print(f"Current price: Bid={price['bid']}, Ask={price['ask']}")
            
            # Get enhanced statistics
            stats = bridge.get_enhanced_stats()
            print(f"Enhanced stats: {stats}")
            
            # Run for a short time to test async functionality
            print("Testing async functionality for 30 seconds...")
            time.sleep(30)
            
        finally:
            # Disconnect
            bridge.disconnect()
    else:
        print("Failed to connect to Enhanced MT5 Bridge")

