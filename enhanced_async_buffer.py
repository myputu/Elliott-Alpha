#!/usr/bin/env python3
"""
Enhanced Async Tick Buffer System - Phase 1 Reliability Enhancement
Full thread-safe implementation with RLock, integrity check, and backpressure handling

Author: Manus AI
Version: 2.0 - Reliability Enhanced
"""

import threading
import time
import queue
import logging
import psutil
import gc
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import json
import weakref

logger = logging.getLogger(__name__)

@dataclass
class TickData:
    """Enhanced tick data structure with integrity validation"""
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    volume: int = 0
    spread: float = field(init=False)
    checksum: str = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields and validate data integrity"""
        self.spread = abs(self.ask - self.bid)
        self.checksum = self._calculate_checksum()
        self._validate_integrity()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for data integrity verification"""
        import hashlib
        data_str = f"{self.symbol}{self.bid}{self.ask}{self.timestamp.isoformat()}{self.volume}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def _validate_integrity(self):
        """Validate tick data integrity"""
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError(f"Invalid price data: bid={self.bid}, ask={self.ask}")
        if self.spread < 0:
            raise ValueError(f"Invalid spread: {self.spread}")
        if self.spread > self.bid * 0.1:  # Spread > 10% of bid price
            logger.warning(f"Unusually large spread detected: {self.spread} for {self.symbol}")

@dataclass
class BufferMetrics:
    """Buffer performance and health metrics"""
    total_ticks_received: int = 0
    total_ticks_processed: int = 0
    total_ticks_dropped: int = 0
    buffer_overflows: int = 0
    integrity_failures: int = 0
    average_processing_time: float = 0.0
    peak_buffer_usage: int = 0
    current_buffer_usage: int = 0
    memory_usage_mb: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization"""
        return {
            'total_ticks_received': self.total_ticks_received,
            'total_ticks_processed': self.total_ticks_processed,
            'total_ticks_dropped': self.total_ticks_dropped,
            'buffer_overflows': self.buffer_overflows,
            'integrity_failures': self.integrity_failures,
            'average_processing_time': self.average_processing_time,
            'peak_buffer_usage': self.peak_buffer_usage,
            'current_buffer_usage': self.current_buffer_usage,
            'memory_usage_mb': self.memory_usage_mb,
            'last_update': self.last_update.isoformat(),
            'processing_rate': self.total_ticks_processed / max(1, (datetime.now() - self.last_update).total_seconds()),
            'drop_rate': self.total_ticks_dropped / max(1, self.total_ticks_received) * 100,
            'buffer_efficiency': self.total_ticks_processed / max(1, self.total_ticks_received) * 100
        }

class BackpressureController:
    """Advanced backpressure control system"""
    
    def __init__(self, max_buffer_size: int = 10000):
        self.max_buffer_size = max_buffer_size
        self.warning_threshold = int(max_buffer_size * 0.7)  # 70%
        self.critical_threshold = int(max_buffer_size * 0.9)  # 90%
        self.emergency_threshold = int(max_buffer_size * 0.95)  # 95%
        
        self._lock = threading.RLock()
        self._backpressure_active = False
        self._emergency_mode = False
        self._last_warning = 0
        
    def check_backpressure(self, current_size: int) -> tuple[bool, str]:
        """Check if backpressure should be applied"""
        with self._lock:
            now = time.time()
            
            if current_size >= self.emergency_threshold:
                if not self._emergency_mode:
                    self._emergency_mode = True
                    logger.critical(f"EMERGENCY: Buffer at {current_size}/{self.max_buffer_size} ({current_size/self.max_buffer_size*100:.1f}%)")
                return True, "EMERGENCY"
            
            elif current_size >= self.critical_threshold:
                if not self._backpressure_active:
                    self._backpressure_active = True
                    logger.error(f"CRITICAL: Buffer at {current_size}/{self.max_buffer_size} ({current_size/self.max_buffer_size*100:.1f}%)")
                return True, "CRITICAL"
            
            elif current_size >= self.warning_threshold:
                if now - self._last_warning > 30:  # Warn every 30 seconds
                    self._last_warning = now
                    logger.warning(f"WARNING: Buffer at {current_size}/{self.max_buffer_size} ({current_size/self.max_buffer_size*100:.1f}%)")
                return False, "WARNING"
            
            else:
                # Reset flags when buffer size is normal
                if self._backpressure_active or self._emergency_mode:
                    logger.info(f"Buffer pressure relieved: {current_size}/{self.max_buffer_size}")
                    self._backpressure_active = False
                    self._emergency_mode = False
                return False, "NORMAL"

class EnhancedAsyncTickBuffer:
    """
    Enhanced Async Tick Buffer with full thread-safety, integrity checking,
    and advanced backpressure handling for 24/7 reliability
    """
    
    def __init__(self, 
                 max_buffer_size: int = 15000,
                 max_memory_mb: int = 512,
                 cleanup_interval: int = 300,  # 5 minutes
                 integrity_check_interval: int = 60):  # 1 minute
        
        self.max_buffer_size = max_buffer_size
        self.max_memory_mb = max_memory_mb
        self.cleanup_interval = cleanup_interval
        self.integrity_check_interval = integrity_check_interval
        
        # Thread-safe data structures
        self._buffer = deque(maxlen=max_buffer_size)
        self._priority_buffer = queue.PriorityQueue(maxsize=1000)
        self._symbol_buffers = defaultdict(lambda: deque(maxlen=1000))
        
        # Thread synchronization
        self._main_lock = threading.RLock()
        self._buffer_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        
        # Control flags
        self._running = False
        self._emergency_stop = False
        self._processing_enabled = True
        
        # Components
        self.backpressure = BackpressureController(max_buffer_size)
        self.metrics = BufferMetrics()
        
        # Callbacks and processors
        self._tick_processors: List[Callable[[TickData], None]] = []
        self._error_handlers: List[Callable[[Exception, TickData], None]] = []
        
        # Background threads
        self._processor_thread = None
        self._cleanup_thread = None
        self._integrity_thread = None
        self._monitor_thread = None
        
        # Performance tracking
        self._processing_times = deque(maxlen=1000)
        self._last_cleanup = time.time()
        self._last_integrity_check = time.time()
        
        logger.info(f"Enhanced Async Tick Buffer initialized: max_size={max_buffer_size}, max_memory={max_memory_mb}MB")
    
    def start(self):
        """Start the enhanced async buffer system"""
        with self._main_lock:
            if self._running:
                logger.warning("Buffer system already running")
                return
            
            self._running = True
            self._emergency_stop = False
            
            # Start background threads
            self._processor_thread = threading.Thread(target=self._process_ticks, daemon=True, name="TickProcessor")
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True, name="CleanupWorker")
            self._integrity_thread = threading.Thread(target=self._integrity_worker, daemon=True, name="IntegrityWorker")
            self._monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True, name="MonitorWorker")
            
            self._processor_thread.start()
            self._cleanup_thread.start()
            self._integrity_thread.start()
            self._monitor_thread.start()
            
            logger.info("Enhanced Async Tick Buffer started successfully")
    
    def stop(self, timeout: float = 10.0):
        """Stop the buffer system gracefully"""
        with self._main_lock:
            if not self._running:
                return
            
            logger.info("Stopping Enhanced Async Tick Buffer...")
            self._running = False
            
            # Wait for threads to finish
            threads = [self._processor_thread, self._cleanup_thread, 
                      self._integrity_thread, self._monitor_thread]
            
            for thread in threads:
                if thread and thread.is_alive():
                    thread.join(timeout=timeout/len(threads))
            
            logger.info("Enhanced Async Tick Buffer stopped")
    
    def emergency_stop(self):
        """Emergency stop - immediate shutdown"""
        logger.critical("EMERGENCY STOP activated!")
        self._emergency_stop = True
        self._running = False
        self._processing_enabled = False
    
    def add_tick(self, tick_data: TickData, priority: int = 0) -> bool:
        """
        Add tick data to buffer with backpressure control
        
        Args:
            tick_data: The tick data to add
            priority: Priority level (0=normal, 1=high, 2=critical)
            
        Returns:
            bool: True if tick was added, False if dropped due to backpressure
        """
        if self._emergency_stop or not self._running:
            return False
        
        try:
            with self._metrics_lock:
                self.metrics.total_ticks_received += 1
            
            # Validate tick data integrity
            if not self._validate_tick(tick_data):
                with self._metrics_lock:
                    self.metrics.integrity_failures += 1
                return False
            
            # Check backpressure
            current_size = len(self._buffer)
            apply_backpressure, pressure_level = self.backpressure.check_backpressure(current_size)
            
            if apply_backpressure and pressure_level == "EMERGENCY":
                # Emergency mode - drop all normal priority ticks
                if priority < 2:
                    with self._metrics_lock:
                        self.metrics.total_ticks_dropped += 1
                    return False
                else:
                    # Force emergency cleanup
                    self._emergency_cleanup()
            
            elif apply_backpressure and pressure_level == "CRITICAL":
                # Critical mode - drop low priority ticks
                if priority < 1:
                    with self._metrics_lock:
                        self.metrics.total_ticks_dropped += 1
                    return False
            
            # Add to appropriate buffer
            with self._buffer_lock:
                if priority > 0:
                    # High priority - use priority queue
                    try:
                        self._priority_buffer.put_nowait((priority, time.time(), tick_data))
                    except queue.Full:
                        # Priority buffer full - force add to main buffer
                        self._buffer.append(tick_data)
                else:
                    # Normal priority - use main buffer
                    self._buffer.append(tick_data)
                
                # Update symbol-specific buffer
                self._symbol_buffers[tick_data.symbol].append(tick_data)
                
                # Update metrics
                with self._metrics_lock:
                    self.metrics.current_buffer_usage = len(self._buffer)
                    if self.metrics.current_buffer_usage > self.metrics.peak_buffer_usage:
                        self.metrics.peak_buffer_usage = self.metrics.current_buffer_usage
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding tick to buffer: {e}")
            self._handle_error(e, tick_data)
            return False
    
    def _validate_tick(self, tick_data: TickData) -> bool:
        """Validate tick data integrity"""
        try:
            # Basic validation
            if not isinstance(tick_data, TickData):
                return False
            
            # Price validation
            if tick_data.bid <= 0 or tick_data.ask <= 0:
                return False
            
            # Spread validation
            if tick_data.spread < 0 or tick_data.spread > tick_data.bid * 0.2:
                return False
            
            # Timestamp validation
            now = datetime.now()
            if tick_data.timestamp > now + timedelta(seconds=10):  # Future timestamp
                return False
            
            if tick_data.timestamp < now - timedelta(hours=1):  # Too old
                return False
            
            # Checksum validation
            expected_checksum = tick_data._calculate_checksum()
            if tick_data.checksum != expected_checksum:
                logger.warning(f"Checksum mismatch for {tick_data.symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Tick validation error: {e}")
            return False
    
    def _process_ticks(self):
        """Main tick processing loop"""
        logger.info("Tick processor started")
        
        while self._running and not self._emergency_stop:
            try:
                if not self._processing_enabled:
                    time.sleep(0.1)
                    continue
                
                tick_data = None
                processing_start = time.time()
                
                # Process priority ticks first
                try:
                    if not self._priority_buffer.empty():
                        priority, timestamp, tick_data = self._priority_buffer.get_nowait()
                        logger.debug(f"Processing priority tick: {tick_data.symbol} (priority={priority})")
                except queue.Empty:
                    pass
                
                # Process normal ticks
                if tick_data is None:
                    with self._buffer_lock:
                        if self._buffer:
                            tick_data = self._buffer.popleft()
                
                if tick_data:
                    # Process the tick
                    self._execute_processors(tick_data)
                    
                    # Update metrics
                    processing_time = time.time() - processing_start
                    self._processing_times.append(processing_time)
                    
                    with self._metrics_lock:
                        self.metrics.total_ticks_processed += 1
                        self.metrics.current_buffer_usage = len(self._buffer)
                        
                        # Update average processing time
                        if self._processing_times:
                            self.metrics.average_processing_time = sum(self._processing_times) / len(self._processing_times)
                
                else:
                    # No ticks to process - brief sleep
                    time.sleep(0.001)  # 1ms
                
            except Exception as e:
                logger.error(f"Error in tick processing loop: {e}")
                if tick_data:
                    self._handle_error(e, tick_data)
                time.sleep(0.01)  # Brief pause on error
        
        logger.info("Tick processor stopped")
    
    def _execute_processors(self, tick_data: TickData):
        """Execute all registered tick processors"""
        for processor in self._tick_processors:
            try:
                processor(tick_data)
            except Exception as e:
                logger.error(f"Error in tick processor {processor.__name__}: {e}")
                self._handle_error(e, tick_data)
    
    def _cleanup_worker(self):
        """Background cleanup worker"""
        logger.info("Cleanup worker started")
        
        while self._running and not self._emergency_stop:
            try:
                current_time = time.time()
                
                if current_time - self._last_cleanup >= self.cleanup_interval:
                    self._perform_cleanup()
                    self._last_cleanup = current_time
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                time.sleep(30)  # Longer pause on error
        
        logger.info("Cleanup worker stopped")
    
    def _perform_cleanup(self):
        """Perform regular cleanup operations"""
        logger.debug("Performing regular cleanup...")
        
        # Memory check
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        with self._metrics_lock:
            self.metrics.memory_usage_mb = memory_usage
        
        if memory_usage > self.max_memory_mb * 0.8:  # 80% threshold
            logger.warning(f"High memory usage: {memory_usage:.1f}MB / {self.max_memory_mb}MB")
            
            if memory_usage > self.max_memory_mb:
                logger.critical(f"Memory limit exceeded: {memory_usage:.1f}MB")
                self._emergency_cleanup()
        
        # Clean old symbol buffers
        cutoff_time = datetime.now() - timedelta(minutes=30)
        symbols_to_clean = []
        
        for symbol, buffer in self._symbol_buffers.items():
            if buffer and buffer[-1].timestamp < cutoff_time:
                symbols_to_clean.append(symbol)
        
        for symbol in symbols_to_clean:
            del self._symbol_buffers[symbol]
            logger.debug(f"Cleaned old buffer for {symbol}")
        
        # Force garbage collection
        gc.collect()
        
        logger.debug("Regular cleanup completed")
    
    def _emergency_cleanup(self):
        """Emergency cleanup when memory/buffer limits exceeded"""
        logger.critical("Performing EMERGENCY cleanup!")
        
        with self._buffer_lock:
            # Keep only the most recent 50% of ticks
            keep_size = len(self._buffer) // 2
            if keep_size > 0:
                # Convert to list, keep last items, convert back to deque
                buffer_list = list(self._buffer)
                self._buffer.clear()
                self._buffer.extend(buffer_list[-keep_size:])
                
                with self._metrics_lock:
                    self.metrics.buffer_overflows += 1
                    self.metrics.total_ticks_dropped += len(buffer_list) - keep_size
        
        # Clear symbol buffers
        for symbol_buffer in self._symbol_buffers.values():
            if len(symbol_buffer) > 100:
                # Keep only last 100 ticks per symbol
                while len(symbol_buffer) > 100:
                    symbol_buffer.popleft()
        
        # Clear priority buffer
        while not self._priority_buffer.empty():
            try:
                self._priority_buffer.get_nowait()
            except queue.Empty:
                break
        
        # Force aggressive garbage collection
        gc.collect()
        
        logger.critical("Emergency cleanup completed")
    
    def _integrity_worker(self):
        """Background integrity checker"""
        logger.info("Integrity worker started")
        
        while self._running and not self._emergency_stop:
            try:
                current_time = time.time()
                
                if current_time - self._last_integrity_check >= self.integrity_check_interval:
                    self._check_system_integrity()
                    self._last_integrity_check = current_time
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in integrity worker: {e}")
                time.sleep(60)  # Longer pause on error
        
        logger.info("Integrity worker stopped")
    
    def _check_system_integrity(self):
        """Check overall system integrity"""
        logger.debug("Checking system integrity...")
        
        issues = []
        
        # Check buffer consistency
        with self._buffer_lock:
            buffer_size = len(self._buffer)
            
            if buffer_size != self.metrics.current_buffer_usage:
                issues.append(f"Buffer size mismatch: actual={buffer_size}, metrics={self.metrics.current_buffer_usage}")
                self.metrics.current_buffer_usage = buffer_size
        
        # Check memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_usage > self.max_memory_mb * 1.2:  # 120% of limit
            issues.append(f"Excessive memory usage: {memory_usage:.1f}MB > {self.max_memory_mb * 1.2:.1f}MB")
        
        # Check thread health
        thread_count = threading.active_count()
        if thread_count > 50:  # Arbitrary threshold
            issues.append(f"High thread count: {thread_count}")
        
        # Check processing rate
        if self.metrics.total_ticks_received > 1000:  # Only check after some activity
            processing_rate = self.metrics.total_ticks_processed / max(1, self.metrics.total_ticks_received)
            if processing_rate < 0.8:  # Less than 80% processing rate
                issues.append(f"Low processing rate: {processing_rate:.1%}")
        
        if issues:
            logger.warning(f"Integrity issues detected: {'; '.join(issues)}")
        else:
            logger.debug("System integrity check passed")
    
    def _monitor_worker(self):
        """Background monitoring and metrics worker"""
        logger.info("Monitor worker started")
        
        while self._running and not self._emergency_stop:
            try:
                # Update metrics
                with self._metrics_lock:
                    self.metrics.last_update = datetime.now()
                    
                    # Calculate memory usage
                    self.metrics.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    # Log periodic status
                    if self.metrics.total_ticks_received % 1000 == 0 and self.metrics.total_ticks_received > 0:
                        logger.info(f"Buffer status: {self.get_status_summary()}")
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitor worker: {e}")
                time.sleep(10)
        
        logger.info("Monitor worker stopped")
    
    def _handle_error(self, error: Exception, tick_data: Optional[TickData] = None):
        """Handle errors with registered error handlers"""
        for handler in self._error_handlers:
            try:
                handler(error, tick_data)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    def add_tick_processor(self, processor: Callable[[TickData], None]):
        """Add a tick processor callback"""
        self._tick_processors.append(processor)
        logger.info(f"Added tick processor: {processor.__name__}")
    
    def add_error_handler(self, handler: Callable[[Exception, TickData], None]):
        """Add an error handler callback"""
        self._error_handlers.append(handler)
        logger.info(f"Added error handler: {handler.__name__}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current buffer metrics"""
        with self._metrics_lock:
            return self.metrics.to_dict()
    
    def get_status_summary(self) -> str:
        """Get a brief status summary"""
        metrics = self.get_metrics()
        return (f"Received: {metrics['total_ticks_received']}, "
                f"Processed: {metrics['total_ticks_processed']}, "
                f"Dropped: {metrics['total_ticks_dropped']}, "
                f"Buffer: {metrics['current_buffer_usage']}/{self.max_buffer_size}, "
                f"Memory: {metrics['memory_usage_mb']:.1f}MB, "
                f"Rate: {metrics['processing_rate']:.1f}/s")
    
    def get_symbol_buffer(self, symbol: str) -> List[TickData]:
        """Get recent ticks for a specific symbol"""
        return list(self._symbol_buffers.get(symbol, []))
    
    def clear_buffers(self):
        """Clear all buffers (use with caution)"""
        with self._buffer_lock:
            self._buffer.clear()
            self._symbol_buffers.clear()
            
            # Clear priority buffer
            while not self._priority_buffer.empty():
                try:
                    self._priority_buffer.get_nowait()
                except queue.Empty:
                    break
        
        logger.warning("All buffers cleared")
    
    def pause_processing(self):
        """Pause tick processing"""
        self._processing_enabled = False
        logger.info("Tick processing paused")
    
    def resume_processing(self):
        """Resume tick processing"""
        self._processing_enabled = True
        logger.info("Tick processing resumed")
    
    def is_healthy(self) -> bool:
        """Check if the buffer system is healthy"""
        if self._emergency_stop or not self._running:
            return False
        
        metrics = self.get_metrics()
        
        # Check memory usage
        if metrics['memory_usage_mb'] > self.max_memory_mb:
            return False
        
        # Check buffer overflow
        if metrics['current_buffer_usage'] >= self.max_buffer_size * 0.95:
            return False
        
        # Check processing rate
        if metrics['total_ticks_received'] > 100:  # Only check after some activity
            if metrics['processing_rate'] < 0.5:  # Less than 50% processing rate
                return False
        
        return True

# Example usage and testing
if __name__ == "__main__":
    import random
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create enhanced buffer
    buffer = EnhancedAsyncTickBuffer(max_buffer_size=1000, max_memory_mb=100)
    
    # Add a simple tick processor
    def simple_processor(tick: TickData):
        logger.debug(f"Processing tick: {tick.symbol} {tick.bid}/{tick.ask}")
    
    buffer.add_tick_processor(simple_processor)
    
    # Start the buffer
    buffer.start()
    
    try:
        # Simulate tick data
        symbols = ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDJPY']
        
        for i in range(100):
            symbol = random.choice(symbols)
            bid = random.uniform(1.0, 2000.0)
            ask = bid + random.uniform(0.0001, 0.01)
            
            tick = TickData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                timestamp=datetime.now(),
                volume=random.randint(1, 100)
            )
            
            buffer.add_tick(tick)
            time.sleep(0.01)  # 10ms between ticks
        
        # Wait a bit for processing
        time.sleep(2)
        
        # Print metrics
        print("\nBuffer Metrics:")
        metrics = buffer.get_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        print(f"\nStatus: {buffer.get_status_summary()}")
        print(f"Healthy: {buffer.is_healthy()}")
        
    finally:
        buffer.stop()
        print("Buffer stopped")

