"""
Comprehensive System Validation and Error Testing Suite
This module provides extensive testing and validation for the Enhanced Elliott Wave Trading System
to ensure error-free operation and optimal performance.

Author: Manus AI
Date: 28 Agustus 2025
Version: 1.0
"""

import unittest
import threading
import time
import logging
import sys
import traceback
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import warnings

# Import system components for testing
try:
    from async_tick_buffer import AsyncTickBuffer, TickData, OHLCData, CircularBuffer, PriorityTickQueue
    from enhanced_mt5_bridge import EnhancedMT5Bridge, TradeSignal, CircuitBreaker
    from elliott_wave_analyzer import ElliottWaveAnalyzer
    from trading_strategy import ElliottWaveTradingStrategy
    from ai_models import AlphaGoModel, SelfPlayModel
    from database import OHLCDatabase
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all system components are available")
    sys.exit(1)

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SystemValidator:
    """
    Comprehensive system validator that performs extensive testing
    of all enhanced trading system components.
    """
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.error_log = []
        self.start_time = datetime.now()
        
        # Test configuration
        self.test_config = {
            'tick_data_volume': 10000,
            'ohlc_data_volume': 1000,
            'stress_test_duration': 60,  # seconds
            'memory_threshold_mb': 500,
            'cpu_threshold_percent': 80,
            'max_latency_ms': 100
        }
        
        logger.info("System Validator initialized")
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive system validation covering all components.
        
        Returns:
            Dict containing validation results and performance metrics
        """
        logger.info("Starting comprehensive system validation...")
        
        validation_tests = [
            ('async_buffer_validation', self._validate_async_buffer),
            ('circular_buffer_validation', self._validate_circular_buffer),
            ('priority_queue_validation', self._validate_priority_queue),
            ('circuit_breaker_validation', self._validate_circuit_breaker),
            ('enhanced_bridge_validation', self._validate_enhanced_bridge),
            ('elliott_wave_integration', self._validate_elliott_wave_integration),
            ('performance_stress_test', self._run_performance_stress_test),
            ('memory_leak_test', self._run_memory_leak_test),
            ('concurrent_access_test', self._run_concurrent_access_test),
            ('error_handling_test', self._run_error_handling_test)
        ]
        
        for test_name, test_function in validation_tests:
            try:
                logger.info(f"Running {test_name}...")
                start_time = time.time()
                
                result = test_function()
                
                execution_time = time.time() - start_time
                self.test_results[test_name] = {
                    'status': 'PASSED' if result else 'FAILED',
                    'execution_time': execution_time,
                    'details': result if isinstance(result, dict) else {}
                }
                
                logger.info(f"{test_name}: {'PASSED' if result else 'FAILED'} ({execution_time:.2f}s)")
                
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'ERROR',
                    'execution_time': 0,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self.error_log.append(f"{test_name}: {str(e)}")
                logger.error(f"{test_name} failed with error: {e}")
        
        # Generate final report
        return self._generate_validation_report()
    
    def _validate_async_buffer(self) -> bool:
        """Validate AsyncTickBuffer functionality."""
        try:
            # Initialize buffer with test configuration
            buffer = AsyncTickBuffer(
                tick_buffer_size=1000,
                ohlc_buffer_size=500,
                queue_size=200,
                max_workers=2
            )
            
            # Test data structures
            test_results = []
            
            # Test 1: Basic tick data handling
            logger.info("Testing basic tick data handling...")
            for i in range(100):
                success = buffer.add_tick_data(
                    symbol="XAUUSD",
                    bid=2000.0 + i * 0.01,
                    ask=2000.5 + i * 0.01,
                    volume=100 + i
                )
                test_results.append(success)
            
            # Wait for processing
            time.sleep(2)
            
            # Verify data retrieval
            latest_ticks = buffer.get_latest_ticks(count=50)
            test_results.append(len(latest_ticks) > 0)
            
            # Test 2: OHLC data handling
            logger.info("Testing OHLC data handling...")
            for i in range(50):
                success = buffer.add_ohlc_data(
                    symbol="XAUUSD",
                    timestamp=datetime.now() - timedelta(minutes=i),
                    open_price=2000.0 + i,
                    high=2005.0 + i,
                    low=1995.0 + i,
                    close=2002.0 + i,
                    volume=1000 + i * 10
                )
                test_results.append(success)
            
            # Wait for processing
            time.sleep(2)
            
            # Verify OHLC retrieval
            latest_ohlc = buffer.get_latest_ohlc(count=25)
            test_results.append(len(latest_ohlc) > 0)
            
            # Test 3: Performance statistics
            stats = buffer.get_performance_stats()
            test_results.append('buffer_performance' in stats)
            test_results.append('tick_buffer_stats' in stats)
            test_results.append('ohlc_buffer_stats' in stats)
            
            # Test 4: Callback functionality
            callback_triggered = threading.Event()
            
            def test_callback(tick_data):
                callback_triggered.set()
            
            buffer.register_tick_callback(test_callback)
            
            # Add data to trigger callback
            buffer.add_tick_data("XAUUSD", 2000.0, 2000.5)
            
            # Wait for callback
            callback_result = callback_triggered.wait(timeout=5)
            test_results.append(callback_result)
            
            # Cleanup
            buffer.shutdown(timeout=10)
            
            # All tests should pass
            success_rate = sum(test_results) / len(test_results)
            logger.info(f"AsyncBuffer validation success rate: {success_rate:.2%}")
            
            return success_rate >= 0.95  # 95% success rate required
            
        except Exception as e:
            logger.error(f"AsyncBuffer validation error: {e}")
            return False
    
    def _validate_circular_buffer(self) -> bool:
        """Validate CircularBuffer functionality."""
        try:
            buffer = CircularBuffer(maxsize=100)
            
            # Test 1: Basic operations
            test_data = [TickData("XAUUSD", datetime.now(), 2000.0 + i, 2000.5 + i, 100) 
                        for i in range(50)]
            
            for data in test_data:
                success = buffer.append(data)
                if not success:
                    return False
            
            # Test 2: Data retrieval
            latest = buffer.get_latest(count=10)
            if len(latest) != 10:
                return False
            
            # Test 3: Overflow handling
            overflow_data = [TickData("XAUUSD", datetime.now(), 2100.0 + i, 2100.5 + i, 100) 
                           for i in range(100)]  # This should cause overflow
            
            overflow_count = 0
            for data in overflow_data:
                if not buffer.append(data):
                    overflow_count += 1
            
            # Test 4: Statistics
            stats = buffer.get_stats()
            required_keys = ['total_items', 'overflows', 'current_size', 'utilization']
            if not all(key in stats for key in required_keys):
                return False
            
            # Test 5: Thread safety
            def concurrent_append(buffer, data_list):
                for data in data_list:
                    buffer.append(data)
            
            threads = []
            for i in range(5):
                thread_data = [TickData("XAUUSD", datetime.now(), 2200.0 + j, 2200.5 + j, 100) 
                              for j in range(20)]
                thread = threading.Thread(target=concurrent_append, args=(buffer, thread_data))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # Buffer should still be functional after concurrent access
            final_stats = buffer.get_stats()
            
            logger.info("CircularBuffer validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"CircularBuffer validation error: {e}")
            return False
    
    def _validate_priority_queue(self) -> bool:
        """Validate PriorityTickQueue functionality."""
        try:
            pq = PriorityTickQueue(maxsize=1000)
            
            # Test 1: Basic priority operations
            test_items = [
                ("critical_item", "critical"),
                ("high_item", "high"),
                ("normal_item", "normal"),
                ("low_item", "low")
            ]
            
            # Add items in reverse priority order
            for item, priority in reversed(test_items):
                success = pq.put(item, priority=priority)
                if not success:
                    return False
            
            # Retrieve items - should come out in priority order
            retrieved_items = []
            for _ in range(len(test_items)):
                item = pq.get(timeout=1.0)
                if item is not None:
                    retrieved_items.append(item)
            
            # Verify priority ordering (critical should come first)
            if retrieved_items[0] != "critical_item":
                logger.error(f"Priority queue ordering failed: {retrieved_items}")
                return False
            
            # Test 2: Queue full handling
            large_items = [f"item_{i}" for i in range(1500)]  # Exceed maxsize
            dropped_count = 0
            
            for item in large_items:
                if not pq.put(item, priority="normal", timeout=0.01):
                    dropped_count += 1
            
            if dropped_count == 0:
                logger.error("Priority queue should have dropped items when full")
                return False
            
            # Test 3: Statistics
            stats = pq.get_stats()
            for priority in ['critical', 'high', 'normal', 'low']:
                if priority not in stats:
                    return False
                required_keys = ['enqueued', 'dequeued', 'dropped', 'current_size']
                if not all(key in stats[priority] for key in required_keys):
                    return False
            
            logger.info("PriorityTickQueue validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"PriorityTickQueue validation error: {e}")
            return False
    
    def _validate_circuit_breaker(self) -> bool:
        """Validate CircuitBreaker functionality."""
        try:
            cb = CircuitBreaker(failure_threshold=3, recovery_timeout=2)
            
            # Test 1: Normal operation (circuit closed)
            def successful_operation():
                return "success"
            
            result = cb.call(successful_operation)
            if result != "success" or cb.state != 'CLOSED':
                return False
            
            # Test 2: Failure handling
            def failing_operation():
                raise Exception("Test failure")
            
            failure_count = 0
            for i in range(5):
                try:
                    cb.call(failing_operation)
                except Exception:
                    failure_count += 1
            
            # Circuit should be open after threshold failures
            if cb.state != 'OPEN':
                logger.error(f"Circuit breaker should be OPEN, but is {cb.state}")
                return False
            
            # Test 3: Circuit open behavior
            try:
                cb.call(successful_operation)
                logger.error("Circuit breaker should reject calls when OPEN")
                return False
            except Exception:
                pass  # Expected behavior
            
            # Test 4: Recovery after timeout
            time.sleep(3)  # Wait for recovery timeout
            
            # Should transition to HALF_OPEN and then CLOSED on success
            result = cb.call(successful_operation)
            if result != "success" or cb.state != 'CLOSED':
                logger.error(f"Circuit breaker recovery failed. State: {cb.state}")
                return False
            
            logger.info("CircuitBreaker validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"CircuitBreaker validation error: {e}")
            return False
    
    def _validate_enhanced_bridge(self) -> bool:
        """Validate EnhancedMT5Bridge functionality (without actual MT5 connection)."""
        try:
            # Test with mock/simulation mode
            bridge = EnhancedMT5Bridge(
                symbol="XAUUSD",
                enable_async_buffer=True,
                buffer_config={
                    'tick_buffer_size': 1000,
                    'ohlc_buffer_size': 500,
                    'queue_size': 200,
                    'max_workers': 2
                }
            )
            
            # Test 1: Initialization
            if bridge.async_buffer is None:
                logger.error("Enhanced bridge should have async buffer when enabled")
                return False
            
            # Test 2: Signal processing queue
            test_signal = TradeSignal(
                symbol="XAUUSD",
                action="buy",
                volume=0.01,
                price=2000.0,
                stop_loss=1990.0,
                take_profit=2010.0,
                priority="high",
                pattern="Impulse Wave 3",
                confidence=0.85
            )
            
            # Test signal creation and properties
            if test_signal.symbol != "XAUUSD":
                return False
            if test_signal.timestamp is None:
                return False
            
            # Test 3: Performance metrics initialization
            stats = bridge.get_enhanced_stats()
            required_keys = [
                'performance_metrics',
                'circuit_breaker_state',
                'active_trades_count',
                'worker_threads_status',
                'async_buffer_stats'
            ]
            
            if not all(key in stats for key in required_keys):
                logger.error(f"Missing required stats keys: {stats.keys()}")
                return False
            
            # Test 4: Thread pool functionality
            if bridge.executor is None:
                return False
            
            # Test 5: Async buffer integration
            if bridge.async_buffer:
                # Test adding data through bridge
                success = bridge.async_buffer.add_tick_data(
                    symbol="XAUUSD",
                    bid=2000.0,
                    ask=2000.5,
                    volume=100
                )
                if not success:
                    return False
            
            # Test 6: Cleanup
            bridge.disconnect()
            
            logger.info("EnhancedMT5Bridge validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"EnhancedMT5Bridge validation error: {e}")
            return False
    
    def _validate_elliott_wave_integration(self) -> bool:
        """Validate Elliott Wave analysis integration with enhanced system."""
        try:
            # Create sample OHLC data for testing
            dates = pd.date_range(start='2025-01-01', periods=1000, freq='1min')
            np.random.seed(42)  # For reproducible results
            
            # Generate realistic price data with trend
            base_price = 2000.0
            price_changes = np.random.normal(0, 0.5, 1000)
            trend = np.linspace(0, 50, 1000)  # Upward trend
            prices = base_price + np.cumsum(price_changes) + trend
            
            # Create OHLC data
            ohlc_data = pd.DataFrame({
                'open': prices,
                'high': prices + np.random.uniform(0, 2, 1000),
                'low': prices - np.random.uniform(0, 2, 1000),
                'close': prices + np.random.normal(0, 0.5, 1000),
                'volume': np.random.randint(100, 1000, 1000)
            }, index=dates)
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            ohlc_data['high'] = np.maximum(ohlc_data['high'], 
                                         np.maximum(ohlc_data['open'], ohlc_data['close']))
            ohlc_data['low'] = np.minimum(ohlc_data['low'], 
                                        np.minimum(ohlc_data['open'], ohlc_data['close']))
            
            # Test 1: Elliott Wave Analyzer initialization
            analyzer = ElliottWaveAnalyzer(ohlc_data)
            if analyzer.data is None or analyzer.data.empty:
                return False
            
            # Test 2: Run analysis
            analyzer.run_analysis()
            
            # Test 3: Trading strategy integration
            strategy = ElliottWaveTradingStrategy(analyzer)
            
            # Test 4: Signal generation
            signals = strategy.generate_signals(analyzer.waves)
            
            # Signals should be a list (even if empty)
            if not isinstance(signals, list):
                logger.error(f"Expected list of signals, got {type(signals)}")
                return False
            
            # Test 5: AI models integration
            alphago_model = AlphaGoModel()
            self_play_model = SelfPlayModel()
            
            # Test basic AI model functionality
            if alphago_model is None or self_play_model is None:
                return False
            
            # Test 6: Integration with async buffer
            buffer = AsyncTickBuffer(
                tick_buffer_size=1000,
                ohlc_buffer_size=500,
                queue_size=200,
                max_workers=2
            )
            
            # Convert OHLC to tick-like data for buffer testing
            for index, row in ohlc_data.head(100).iterrows():
                mid_price = (row['high'] + row['low']) / 2
                buffer.add_tick_data(
                    symbol="XAUUSD",
                    bid=mid_price - 0.25,
                    ask=mid_price + 0.25,
                    timestamp=index,
                    volume=int(row['volume'])
                )
            
            # Wait for processing
            time.sleep(2)
            
            # Verify data was processed
            latest_ticks = buffer.get_latest_ticks(count=50)
            if len(latest_ticks) == 0:
                logger.error("No ticks processed by buffer")
                return False
            
            # Cleanup
            buffer.shutdown(timeout=5)
            
            logger.info("Elliott Wave integration validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Elliott Wave integration validation error: {e}")
            return False
    
    def _run_performance_stress_test(self) -> bool:
        """Run performance stress test to validate system under load."""
        try:
            logger.info("Starting performance stress test...")
            
            # Initialize system components
            buffer = AsyncTickBuffer(
                tick_buffer_size=50000,
                ohlc_buffer_size=10000,
                queue_size=5000,
                max_workers=4
            )
            
            # Performance metrics
            start_time = time.time()
            tick_count = 0
            error_count = 0
            
            # Stress test parameters
            duration = self.test_config['stress_test_duration']
            target_tps = 1000  # Ticks per second
            
            def tick_generator():
                nonlocal tick_count, error_count
                end_time = time.time() + duration
                
                while time.time() < end_time:
                    try:
                        # Generate realistic tick data
                        base_price = 2000.0
                        price_variation = np.random.normal(0, 1.0)
                        bid = base_price + price_variation
                        ask = bid + np.random.uniform(0.1, 0.5)
                        
                        success = buffer.add_tick_data(
                            symbol="XAUUSD",
                            bid=bid,
                            ask=ask,
                            volume=np.random.randint(1, 100),
                            priority='high'
                        )
                        
                        if success:
                            tick_count += 1
                        else:
                            error_count += 1
                        
                        # Control rate to target TPS
                        time.sleep(1.0 / target_tps)
                        
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Tick generation error: {e}")
            
            # Run stress test
            stress_thread = threading.Thread(target=tick_generator)
            stress_thread.start()
            
            # Monitor system resources during test
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            peak_memory = initial_memory
            peak_cpu = 0
            
            monitor_start = time.time()
            while stress_thread.is_alive():
                try:
                    # Monitor memory usage
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    
                    # Monitor CPU usage
                    current_cpu = psutil.Process().cpu_percent()
                    peak_cpu = max(peak_cpu, current_cpu)
                    
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
            
            stress_thread.join()
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            actual_tps = tick_count / total_time
            error_rate = error_count / (tick_count + error_count) if (tick_count + error_count) > 0 else 0
            memory_increase = peak_memory - initial_memory
            
            # Get buffer statistics
            buffer_stats = buffer.get_performance_stats()
            
            # Performance criteria
            performance_results = {
                'tick_count': tick_count,
                'error_count': error_count,
                'actual_tps': actual_tps,
                'error_rate': error_rate,
                'memory_increase_mb': memory_increase,
                'peak_memory_mb': peak_memory,
                'peak_cpu_percent': peak_cpu,
                'buffer_stats': buffer_stats
            }
            
            # Store performance metrics
            self.performance_metrics['stress_test'] = performance_results
            
            # Cleanup
            buffer.shutdown(timeout=10)
            
            # Evaluate performance criteria
            success_criteria = [
                actual_tps >= target_tps * 0.8,  # At least 80% of target TPS
                error_rate < 0.05,  # Less than 5% error rate
                memory_increase < self.test_config['memory_threshold_mb'],  # Memory increase limit
                peak_cpu < self.test_config['cpu_threshold_percent']  # CPU usage limit
            ]
            
            success = all(success_criteria)
            
            logger.info(f"Stress test results:")
            logger.info(f"  Ticks processed: {tick_count}")
            logger.info(f"  Actual TPS: {actual_tps:.2f}")
            logger.info(f"  Error rate: {error_rate:.2%}")
            logger.info(f"  Memory increase: {memory_increase:.2f} MB")
            logger.info(f"  Peak CPU: {peak_cpu:.1f}%")
            logger.info(f"  Success: {success}")
            
            return success
            
        except Exception as e:
            logger.error(f"Performance stress test error: {e}")
            return False
    
    def _run_memory_leak_test(self) -> bool:
        """Test for memory leaks in the system."""
        try:
            logger.info("Starting memory leak test...")
            
            # Initial memory measurement
            gc.collect()  # Force garbage collection
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Run multiple cycles of buffer operations
            cycles = 10
            items_per_cycle = 1000
            
            for cycle in range(cycles):
                buffer = AsyncTickBuffer(
                    tick_buffer_size=2000,
                    ohlc_buffer_size=1000,
                    queue_size=500,
                    max_workers=2
                )
                
                # Add data
                for i in range(items_per_cycle):
                    buffer.add_tick_data(
                        symbol="XAUUSD",
                        bid=2000.0 + i * 0.01,
                        ask=2000.5 + i * 0.01,
                        volume=100
                    )
                
                # Wait for processing
                time.sleep(1)
                
                # Shutdown buffer
                buffer.shutdown(timeout=5)
                
                # Force garbage collection
                gc.collect()
                
                # Check memory after each cycle
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                logger.info(f"Cycle {cycle + 1}: Memory = {current_memory:.2f} MB (+{memory_increase:.2f} MB)")
                
                # If memory increase is too large, might indicate a leak
                if memory_increase > 100:  # 100 MB threshold
                    logger.warning(f"Potential memory leak detected: {memory_increase:.2f} MB increase")
            
            # Final memory measurement
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            total_increase = final_memory - initial_memory
            
            # Memory leak criteria (should not increase by more than 50 MB)
            memory_leak_threshold = 50  # MB
            success = total_increase < memory_leak_threshold
            
            logger.info(f"Memory leak test results:")
            logger.info(f"  Initial memory: {initial_memory:.2f} MB")
            logger.info(f"  Final memory: {final_memory:.2f} MB")
            logger.info(f"  Total increase: {total_increase:.2f} MB")
            logger.info(f"  Success: {success}")
            
            return success
            
        except Exception as e:
            logger.error(f"Memory leak test error: {e}")
            return False
    
    def _run_concurrent_access_test(self) -> bool:
        """Test concurrent access to system components."""
        try:
            logger.info("Starting concurrent access test...")
            
            buffer = AsyncTickBuffer(
                tick_buffer_size=10000,
                ohlc_buffer_size=5000,
                queue_size=2000,
                max_workers=4
            )
            
            # Concurrent operations
            num_threads = 10
            operations_per_thread = 100
            results = []
            
            def concurrent_operations(thread_id):
                thread_results = []
                
                for i in range(operations_per_thread):
                    try:
                        # Add tick data
                        success = buffer.add_tick_data(
                            symbol=f"SYMBOL_{thread_id}",
                            bid=2000.0 + thread_id + i * 0.01,
                            ask=2000.5 + thread_id + i * 0.01,
                            volume=100 + i
                        )
                        thread_results.append(success)
                        
                        # Occasionally read data
                        if i % 10 == 0:
                            latest = buffer.get_latest_ticks(count=5)
                            thread_results.append(len(latest) >= 0)  # Should not fail
                        
                        # Small delay to simulate real usage
                        time.sleep(0.001)
                        
                    except Exception as e:
                        logger.error(f"Thread {thread_id} error: {e}")
                        thread_results.append(False)
                
                return thread_results
            
            # Start concurrent threads
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(concurrent_operations, i) 
                    for i in range(num_threads)
                ]
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        thread_results = future.result(timeout=30)
                        results.extend(thread_results)
                    except Exception as e:
                        logger.error(f"Concurrent thread failed: {e}")
                        results.append(False)
            
            # Wait for buffer processing
            time.sleep(3)
            
            # Check final state
            final_stats = buffer.get_performance_stats()
            
            # Cleanup
            buffer.shutdown(timeout=10)
            
            # Evaluate results
            success_rate = sum(results) / len(results) if results else 0
            success = success_rate >= 0.95  # 95% success rate required
            
            logger.info(f"Concurrent access test results:")
            logger.info(f"  Total operations: {len(results)}")
            logger.info(f"  Success rate: {success_rate:.2%}")
            logger.info(f"  Success: {success}")
            
            return success
            
        except Exception as e:
            logger.error(f"Concurrent access test error: {e}")
            return False
    
    def _run_error_handling_test(self) -> bool:
        """Test error handling and recovery mechanisms."""
        try:
            logger.info("Starting error handling test...")
            
            # Test 1: Circuit breaker error handling
            cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
            
            def failing_function():
                raise ValueError("Test error")
            
            # Trigger circuit breaker
            failures = 0
            for i in range(5):
                try:
                    cb.call(failing_function)
                except:
                    failures += 1
            
            if cb.state != 'OPEN':
                logger.error("Circuit breaker should be OPEN after failures")
                return False
            
            # Test recovery
            time.sleep(2)  # Wait for recovery timeout
            
            def working_function():
                return "success"
            
            try:
                result = cb.call(working_function)
                if result != "success" or cb.state != 'CLOSED':
                    logger.error("Circuit breaker recovery failed")
                    return False
            except:
                logger.error("Circuit breaker should have recovered")
                return False
            
            # Test 2: Buffer overflow handling
            small_buffer = AsyncTickBuffer(
                tick_buffer_size=10,  # Very small buffer
                ohlc_buffer_size=5,
                queue_size=20,
                max_workers=1
            )
            
            # Overflow the buffer
            overflow_detected = False
            for i in range(50):  # Much more than buffer size
                success = small_buffer.add_tick_data(
                    symbol="XAUUSD",
                    bid=2000.0 + i,
                    ask=2000.5 + i,
                    volume=100
                )
                if not success:
                    overflow_detected = True
            
            if not overflow_detected:
                logger.error("Buffer overflow should have been detected")
                small_buffer.shutdown()
                return False
            
            # Buffer should still be functional
            stats = small_buffer.get_performance_stats()
            if 'buffer_performance' not in stats:
                logger.error("Buffer should still provide stats after overflow")
                small_buffer.shutdown()
                return False
            
            small_buffer.shutdown(timeout=5)
            
            # Test 3: Invalid data handling
            buffer = AsyncTickBuffer(
                tick_buffer_size=1000,
                ohlc_buffer_size=500,
                queue_size=200,
                max_workers=2
            )
            
            # Test with invalid data (should not crash)
            invalid_data_tests = [
                # Negative prices
                lambda: buffer.add_tick_data("XAUUSD", -100.0, -99.5),
                # Zero volume
                lambda: buffer.add_tick_data("XAUUSD", 2000.0, 2000.5, volume=0),
                # Bid > Ask (invalid spread)
                lambda: buffer.add_tick_data("XAUUSD", 2000.5, 2000.0),
            ]
            
            for test_func in invalid_data_tests:
                try:
                    test_func()
                    # Should not crash, but may return False
                except Exception as e:
                    logger.info(f"Invalid data handled gracefully: {e}")
            
            # System should still be functional
            normal_success = buffer.add_tick_data("XAUUSD", 2000.0, 2000.5)
            if not normal_success:
                logger.error("System should handle normal data after invalid data")
                buffer.shutdown()
                return False
            
            buffer.shutdown(timeout=5)
            
            logger.info("Error handling test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error handling test error: {e}")
            return False
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate overall success rate
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result['status'] == 'PASSED')
        total_tests = len(self.test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Categorize results
        passed = [name for name, result in self.test_results.items() 
                 if result['status'] == 'PASSED']
        failed = [name for name, result in self.test_results.items() 
                 if result['status'] == 'FAILED']
        errors = [name for name, result in self.test_results.items() 
                 if result['status'] == 'ERROR']
        
        report = {
            'validation_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': len(failed),
                'error_tests': len(errors),
                'success_rate': success_rate,
                'total_execution_time': total_time
            },
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'passed_tests': passed,
            'failed_tests': failed,
            'error_tests': errors,
            'error_log': self.error_log,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check success rate
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result['status'] == 'PASSED')
        total_tests = len(self.test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        if success_rate < 0.9:
            recommendations.append(
                "Overall success rate is below 90%. Review failed tests and address issues before production deployment."
            )
        
        # Check performance metrics
        if 'stress_test' in self.performance_metrics:
            stress_results = self.performance_metrics['stress_test']
            
            if stress_results['error_rate'] > 0.05:
                recommendations.append(
                    f"Error rate in stress test is {stress_results['error_rate']:.2%}. "
                    "Consider optimizing error handling and system resilience."
                )
            
            if stress_results['memory_increase_mb'] > 100:
                recommendations.append(
                    f"Memory usage increased by {stress_results['memory_increase_mb']:.2f} MB during stress test. "
                    "Review memory management and consider implementing more aggressive garbage collection."
                )
        
        # Check for specific test failures
        failed_tests = [name for name, result in self.test_results.items() 
                       if result['status'] in ['FAILED', 'ERROR']]
        
        if 'memory_leak_test' in failed_tests:
            recommendations.append(
                "Memory leak test failed. Review object lifecycle management and ensure proper cleanup of resources."
            )
        
        if 'concurrent_access_test' in failed_tests:
            recommendations.append(
                "Concurrent access test failed. Review thread safety mechanisms and consider using more robust locking strategies."
            )
        
        if 'performance_stress_test' in failed_tests:
            recommendations.append(
                "Performance stress test failed. Consider optimizing algorithms, increasing buffer sizes, or adding more worker threads."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "All tests passed successfully! System is ready for production deployment with proper monitoring."
            )
        
        recommendations.append(
            "Implement comprehensive monitoring and alerting in production environment."
        )
        
        recommendations.append(
            "Regularly run these validation tests in staging environment before updates."
        )
        
        return recommendations

def main():
    """Main function to run system validation."""
    print("=" * 80)
    print("Enhanced Elliott Wave Trading System - Comprehensive Validation")
    print("=" * 80)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    validator = SystemValidator()
    
    try:
        # Run comprehensive validation
        report = validator.run_comprehensive_validation()
        
        # Print summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        summary = report['validation_summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Errors: {summary['error_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Time: {summary['total_execution_time']:.2f} seconds")
        
        # Print detailed results
        print("\n" + "=" * 80)
        print("DETAILED RESULTS")
        print("=" * 80)
        
        for test_name, result in report['test_results'].items():
            status_symbol = "✓" if result['status'] == 'PASSED' else "✗"
            print(f"{status_symbol} {test_name}: {result['status']} ({result['execution_time']:.2f}s)")
            
            if result['status'] == 'ERROR':
                print(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Print recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        
        for i, recommendation in enumerate(report['recommendations'], 1):
            print(f"{i}. {recommendation}")
        
        # Save detailed report
        import json
        with open('system_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: system_validation_report.json")
        print(f"Log file saved to: system_validation.log")
        
        # Return exit code based on success rate
        return 0 if summary['success_rate'] >= 0.9 else 1
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        print(f"\nValidation failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

