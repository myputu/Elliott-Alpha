"""
Asynchronous Tick Buffer System for High-Frequency Trading Data
This module provides high-performance buffering and queue management for tick data
between Python trading system and MetaTrader 5 Bridge.

Author: Manus AI
Date: 28 Agustus 2025
Version: 1.0
"""

import asyncio
import threading
import queue
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TickData:
    """Data structure for individual tick data."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    volume: int = 0
    spread: float = field(init=False)
    
    def __post_init__(self):
        self.spread = self.ask - self.bid

@dataclass
class OHLCData:
    """Data structure for OHLC bar data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "M1"

class CircularBuffer:
    """
    High-performance circular buffer for tick data storage.
    Optimized for memory efficiency and fast access patterns.
    """
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)
        self._lock = threading.RLock()
        self._stats = {
            'total_items': 0,
            'overflows': 0,
            'last_update': None
        }
    
    def append(self, item: TickData) -> bool:
        """
        Append item to buffer with thread safety.
        
        Args:
            item: TickData object to append
            
        Returns:
            bool: True if successful, False if buffer overflow
        """
        with self._lock:
            was_full = len(self.buffer) >= self.maxsize
            self.buffer.append(item)
            
            self._stats['total_items'] += 1
            self._stats['last_update'] = datetime.now()
            
            if was_full:
                self._stats['overflows'] += 1
                logger.warning(f"Buffer overflow detected. Total overflows: {self._stats['overflows']}")
                return False
            
            return True
    
    def get_latest(self, count: int = 1) -> List[TickData]:
        """Get latest N items from buffer."""
        with self._lock:
            if count == 1:
                return [self.buffer[-1]] if self.buffer else []
            else:
                return list(self.buffer)[-count:] if len(self.buffer) >= count else list(self.buffer)
    
    def get_range(self, start_time: datetime, end_time: datetime) -> List[TickData]:
        """Get items within time range."""
        with self._lock:
            return [
                item for item in self.buffer 
                if start_time <= item.timestamp <= end_time
            ]
    
    def clear(self):
        """Clear buffer and reset stats."""
        with self._lock:
            self.buffer.clear()
            self._stats = {
                'total_items': 0,
                'overflows': 0,
                'last_update': None
            }
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                **self._stats,
                'current_size': len(self.buffer),
                'max_size': self.maxsize,
                'utilization': len(self.buffer) / self.maxsize * 100
            }

class PriorityTickQueue:
    """
    Priority queue system for different types of market data.
    Ensures critical data (like trade signals) get processed first.
    """
    
    def __init__(self, maxsize: int = 5000):
        self.maxsize = maxsize
        self._queues = {
            'critical': queue.PriorityQueue(maxsize=maxsize//4),    # Trade signals, orders
            'high': queue.PriorityQueue(maxsize=maxsize//4),       # Real-time ticks
            'normal': queue.PriorityQueue(maxsize=maxsize//2),     # OHLC data, analysis
            'low': queue.PriorityQueue(maxsize=maxsize//4)         # Historical data, logs
        }
        self._stats = {priority: {'enqueued': 0, 'dequeued': 0, 'dropped': 0} 
                      for priority in self._queues.keys()}
        self._lock = threading.RLock()
    
    def put(self, item: Any, priority: str = 'normal', timeout: float = 0.1) -> bool:
        """
        Put item in priority queue.
        
        Args:
            item: Data item to queue
            priority: Queue priority ('critical', 'high', 'normal', 'low')
            timeout: Timeout for queue operation
            
        Returns:
            bool: True if successful, False if queue full or timeout
        """
        if priority not in self._queues:
            priority = 'normal'
        
        try:
            # Use timestamp as priority value (lower timestamp = higher priority)
            priority_value = time.time()
            self._queues[priority].put((priority_value, item), timeout=timeout)
            
            with self._lock:
                self._stats[priority]['enqueued'] += 1
            
            return True
            
        except queue.Full:
            with self._lock:
                self._stats[priority]['dropped'] += 1
            logger.warning(f"Queue {priority} is full. Item dropped.")
            return False
    
    def get(self, priority: str = None, timeout: float = 0.1) -> Optional[Any]:
        """
        Get item from priority queue.
        
        Args:
            priority: Specific priority queue to get from (None for highest priority available)
            timeout: Timeout for queue operation
            
        Returns:
            Item from queue or None if timeout/empty
        """
        if priority:
            # Get from specific priority queue
            try:
                _, item = self._queues[priority].get(timeout=timeout)
                with self._lock:
                    self._stats[priority]['dequeued'] += 1
                return item
            except queue.Empty:
                return None
        else:
            # Get from highest priority queue with data
            for priority in ['critical', 'high', 'normal', 'low']:
                try:
                    _, item = self._queues[priority].get_nowait()
                    with self._lock:
                        self._stats[priority]['dequeued'] += 1
                    return item
                except queue.Empty:
                    continue
            return None
    
    def get_stats(self) -> Dict:
        """Get queue statistics."""
        with self._lock:
            stats = dict(self._stats)
            for priority, q in self._queues.items():
                stats[priority]['current_size'] = q.qsize()
                stats[priority]['max_size'] = q.maxsize
        return stats

class AsyncTickBuffer:
    """
    Main asynchronous tick buffer system that coordinates all buffering operations.
    Provides high-performance data handling for MT5 Bridge communication.
    """
    
    def __init__(self, 
                 tick_buffer_size: int = 50000,
                 ohlc_buffer_size: int = 10000,
                 queue_size: int = 5000,
                 max_workers: int = 4):
        """
        Initialize AsyncTickBuffer.
        
        Args:
            tick_buffer_size: Maximum size of tick data circular buffer
            ohlc_buffer_size: Maximum size of OHLC data circular buffer
            queue_size: Maximum size of priority queues
            max_workers: Maximum number of worker threads
        """
        # Buffers for different data types
        self.tick_buffer = CircularBuffer(tick_buffer_size)
        self.ohlc_buffer = CircularBuffer(ohlc_buffer_size)
        
        # Priority queue system
        self.priority_queue = PriorityTickQueue(queue_size)
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TickBuffer")
        self.worker_threads = {}
        self._shutdown_event = threading.Event()
        
        # Performance monitoring
        self._performance_stats = {
            'start_time': datetime.now(),
            'ticks_processed': 0,
            'ohlc_processed': 0,
            'avg_processing_time': 0.0,
            'peak_memory_usage': 0,
            'errors': 0
        }
        
        # Callbacks for data processing
        self._tick_callbacks: List[Callable] = []
        self._ohlc_callbacks: List[Callable] = []
        self._error_callbacks: List[Callable] = []
        
        # Start background workers
        self._start_workers()
        
        logger.info(f"AsyncTickBuffer initialized with {max_workers} workers")
    
    def _start_workers(self):
        """Start background worker threads."""
        # Tick processing worker
        self.worker_threads['tick_processor'] = threading.Thread(
            target=self._tick_processing_worker,
            name="TickProcessor",
            daemon=True
        )
        self.worker_threads['tick_processor'].start()
        
        # OHLC processing worker
        self.worker_threads['ohlc_processor'] = threading.Thread(
            target=self._ohlc_processing_worker,
            name="OHLCProcessor",
            daemon=True
        )
        self.worker_threads['ohlc_processor'].start()
        
        # Performance monitoring worker
        self.worker_threads['monitor'] = threading.Thread(
            target=self._monitoring_worker,
            name="PerformanceMonitor",
            daemon=True
        )
        self.worker_threads['monitor'].start()
        
        logger.info("Background workers started")
    
    def _tick_processing_worker(self):
        """Background worker for processing tick data."""
        while not self._shutdown_event.is_set():
            try:
                # Get tick data from priority queue
                tick_data = self.priority_queue.get(priority='high', timeout=1.0)
                
                if tick_data is None:
                    continue
                
                start_time = time.time()
                
                # Store in circular buffer
                self.tick_buffer.append(tick_data)
                
                # Execute callbacks
                for callback in self._tick_callbacks:
                    try:
                        callback(tick_data)
                    except Exception as e:
                        logger.error(f"Tick callback error: {e}")
                        self._performance_stats['errors'] += 1
                
                # Update performance stats
                processing_time = time.time() - start_time
                self._update_performance_stats('tick', processing_time)
                
            except Exception as e:
                logger.error(f"Tick processing worker error: {e}")
                self._performance_stats['errors'] += 1
                time.sleep(0.1)  # Brief pause on error
    
    def _ohlc_processing_worker(self):
        """Background worker for processing OHLC data."""
        while not self._shutdown_event.is_set():
            try:
                # Get OHLC data from priority queue
                ohlc_data = self.priority_queue.get(priority='normal', timeout=1.0)
                
                if ohlc_data is None:
                    continue
                
                start_time = time.time()
                
                # Store in circular buffer
                self.ohlc_buffer.append(ohlc_data)
                
                # Execute callbacks
                for callback in self._ohlc_callbacks:
                    try:
                        callback(ohlc_data)
                    except Exception as e:
                        logger.error(f"OHLC callback error: {e}")
                        self._performance_stats['errors'] += 1
                
                # Update performance stats
                processing_time = time.time() - start_time
                self._update_performance_stats('ohlc', processing_time)
                
            except Exception as e:
                logger.error(f"OHLC processing worker error: {e}")
                self._performance_stats['errors'] += 1
                time.sleep(0.1)  # Brief pause on error
    
    def _monitoring_worker(self):
        """Background worker for performance monitoring."""
        while not self._shutdown_event.is_set():
            try:
                # Memory usage monitoring
                import psutil
                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                
                if memory_usage > self._performance_stats['peak_memory_usage']:
                    self._performance_stats['peak_memory_usage'] = memory_usage
                
                # Garbage collection if memory usage is high
                if memory_usage > 500:  # 500 MB threshold
                    gc.collect()
                    logger.info(f"Garbage collection triggered. Memory usage: {memory_usage:.2f} MB")
                
                # Log performance stats every 5 minutes
                if int(time.time()) % 300 == 0:
                    self._log_performance_stats()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring worker error: {e}")
                time.sleep(30)  # Longer pause on error
    
    def _update_performance_stats(self, data_type: str, processing_time: float):
        """Update performance statistics."""
        if data_type == 'tick':
            self._performance_stats['ticks_processed'] += 1
        elif data_type == 'ohlc':
            self._performance_stats['ohlc_processed'] += 1
        
        # Update average processing time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        current_avg = self._performance_stats['avg_processing_time']
        self._performance_stats['avg_processing_time'] = (
            alpha * processing_time + (1 - alpha) * current_avg
        )
    
    def _log_performance_stats(self):
        """Log current performance statistics."""
        uptime = datetime.now() - self._performance_stats['start_time']
        
        logger.info(f"AsyncTickBuffer Performance Stats:")
        logger.info(f"  Uptime: {uptime}")
        logger.info(f"  Ticks processed: {self._performance_stats['ticks_processed']}")
        logger.info(f"  OHLC processed: {self._performance_stats['ohlc_processed']}")
        logger.info(f"  Avg processing time: {self._performance_stats['avg_processing_time']:.4f}s")
        logger.info(f"  Peak memory usage: {self._performance_stats['peak_memory_usage']:.2f} MB")
        logger.info(f"  Errors: {self._performance_stats['errors']}")
        
        # Buffer stats
        tick_stats = self.tick_buffer.get_stats()
        ohlc_stats = self.ohlc_buffer.get_stats()
        queue_stats = self.priority_queue.get_stats()
        
        logger.info(f"  Tick buffer utilization: {tick_stats['utilization']:.1f}%")
        logger.info(f"  OHLC buffer utilization: {ohlc_stats['utilization']:.1f}%")
        logger.info(f"  Queue stats: {queue_stats}")
    
    def add_tick_data(self, symbol: str, bid: float, ask: float, 
                     timestamp: datetime = None, volume: int = 0, 
                     priority: str = 'high') -> bool:
        """
        Add tick data to buffer system.
        
        Args:
            symbol: Trading symbol
            bid: Bid price
            ask: Ask price
            timestamp: Tick timestamp (current time if None)
            volume: Tick volume
            priority: Queue priority
            
        Returns:
            bool: True if successful
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        tick_data = TickData(
            symbol=symbol,
            timestamp=timestamp,
            bid=bid,
            ask=ask,
            volume=volume
        )
        
        return self.priority_queue.put(tick_data, priority=priority)
    
    def add_ohlc_data(self, symbol: str, timestamp: datetime,
                     open_price: float, high: float, low: float, close: float,
                     volume: int, timeframe: str = "M1",
                     priority: str = 'normal') -> bool:
        """
        Add OHLC data to buffer system.
        
        Args:
            symbol: Trading symbol
            timestamp: Bar timestamp
            open_price: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Bar volume
            timeframe: Timeframe (M1, M5, etc.)
            priority: Queue priority
            
        Returns:
            bool: True if successful
        """
        ohlc_data = OHLCData(
            symbol=symbol,
            timestamp=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            timeframe=timeframe
        )
        
        return self.priority_queue.put(ohlc_data, priority=priority)
    
    def get_latest_ticks(self, count: int = 100) -> List[TickData]:
        """Get latest tick data from buffer."""
        return self.tick_buffer.get_latest(count)
    
    def get_latest_ohlc(self, count: int = 100) -> List[OHLCData]:
        """Get latest OHLC data from buffer."""
        return self.ohlc_buffer.get_latest(count)
    
    def get_ticks_in_range(self, start_time: datetime, end_time: datetime) -> List[TickData]:
        """Get tick data within time range."""
        return self.tick_buffer.get_range(start_time, end_time)
    
    def register_tick_callback(self, callback: Callable[[TickData], None]):
        """Register callback for tick data processing."""
        self._tick_callbacks.append(callback)
        logger.info(f"Tick callback registered: {callback.__name__}")
    
    def register_ohlc_callback(self, callback: Callable[[OHLCData], None]):
        """Register callback for OHLC data processing."""
        self._ohlc_callbacks.append(callback)
        logger.info(f"OHLC callback registered: {callback.__name__}")
    
    def register_error_callback(self, callback: Callable[[Exception], None]):
        """Register callback for error handling."""
        self._error_callbacks.append(callback)
        logger.info(f"Error callback registered: {callback.__name__}")
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        return {
            'buffer_performance': self._performance_stats.copy(),
            'tick_buffer_stats': self.tick_buffer.get_stats(),
            'ohlc_buffer_stats': self.ohlc_buffer.get_stats(),
            'queue_stats': self.priority_queue.get_stats(),
            'worker_threads': {name: thread.is_alive() for name, thread in self.worker_threads.items()}
        }
    
    def shutdown(self, timeout: float = 30.0):
        """
        Gracefully shutdown the buffer system.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        logger.info("Shutting down AsyncTickBuffer...")
        
        # Signal shutdown to all workers
        self._shutdown_event.set()
        
        # Wait for workers to finish
        start_time = time.time()
        for name, thread in self.worker_threads.items():
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 0:
                thread.join(timeout=remaining_time)
                if thread.is_alive():
                    logger.warning(f"Worker {name} did not shutdown gracefully")
            else:
                logger.warning(f"Timeout waiting for worker {name}")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True, timeout=timeout)
        
        # Clear buffers
        self.tick_buffer.clear()
        self.ohlc_buffer.clear()
        
        logger.info("AsyncTickBuffer shutdown complete")

# Example usage and testing
if __name__ == "__main__":
    # Initialize buffer system
    buffer = AsyncTickBuffer(
        tick_buffer_size=10000,
        ohlc_buffer_size=5000,
        queue_size=2000,
        max_workers=2
    )
    
    # Example callback functions
    def on_tick_received(tick_data: TickData):
        print(f"Tick received: {tick_data.symbol} - Bid: {tick_data.bid}, Ask: {tick_data.ask}")
    
    def on_ohlc_received(ohlc_data: OHLCData):
        print(f"OHLC received: {ohlc_data.symbol} - Close: {ohlc_data.close}")
    
    # Register callbacks
    buffer.register_tick_callback(on_tick_received)
    buffer.register_ohlc_callback(on_ohlc_received)
    
    # Simulate adding data
    import random
    
    try:
        for i in range(100):
            # Add tick data
            buffer.add_tick_data(
                symbol="XAUUSD",
                bid=2000.0 + random.uniform(-10, 10),
                ask=2000.5 + random.uniform(-10, 10),
                volume=random.randint(1, 100)
            )
            
            # Add OHLC data occasionally
            if i % 10 == 0:
                buffer.add_ohlc_data(
                    symbol="XAUUSD",
                    timestamp=datetime.now(),
                    open_price=2000.0,
                    high=2005.0,
                    low=1995.0,
                    close=2002.0,
                    volume=1000
                )
            
            time.sleep(0.01)  # 10ms interval
        
        # Wait a bit for processing
        time.sleep(2)
        
        # Get performance stats
        stats = buffer.get_performance_stats()
        print(f"Performance stats: {stats}")
        
    finally:
        # Shutdown buffer system
        buffer.shutdown()

