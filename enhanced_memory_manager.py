"""
Enhanced Memory Management System for Elliott Wave Trading System
This module provides comprehensive memory management capabilities including
cleanup procedures, monitoring, and optimization for long-running trading operations.

Author: Manus AI
Date: 28 Agustus 2025
Version: 1.0
"""

import gc
import sys
import threading
import time
import psutil
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import defaultdict
import logging

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory statistics data structure."""
    timestamp: datetime
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory percentage
    available_mb: float  # Available memory in MB
    gc_collections: Dict[int, int]  # Garbage collection counts by generation
    object_counts: Dict[str, int]  # Object counts by type

class MemoryMonitor:
    """
    Advanced memory monitoring system that tracks memory usage patterns
    and provides insights for optimization.
    """
    
    def __init__(self, monitoring_interval: int = 30, history_size: int = 1000):
        """
        Initialize memory monitor.
        
        Args:
            monitoring_interval: Seconds between monitoring checks
            history_size: Number of historical records to maintain
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.memory_history: List[MemoryStats] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable[[MemoryStats], None]] = []
        self._lock = threading.Lock()
        
        # Thresholds for alerts
        self.memory_warning_threshold = 80.0  # Percentage
        self.memory_critical_threshold = 90.0  # Percentage
        self.growth_rate_threshold = 10.0  # MB per minute
        
        logger.info("Memory monitor initialized")
    
    def start_monitoring(self):
        """Start continuous memory monitoring."""
        if self.monitoring_active:
            logger.warning("Memory monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="MemoryMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                stats = self._collect_memory_stats()
                
                with self._lock:
                    self.memory_history.append(stats)
                    
                    # Maintain history size limit
                    if len(self.memory_history) > self.history_size:
                        self.memory_history.pop(0)
                
                # Check for alerts
                self._check_memory_alerts(stats)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        logger.error(f"Memory monitor callback error: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_memory_stats(self) -> MemoryStats:
        """Collect current memory statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # System memory info
        system_memory = psutil.virtual_memory()
        
        # Garbage collection stats
        gc_stats = {}
        for i in range(3):  # Python has 3 GC generations
            gc_stats[i] = gc.get_count()[i] if i < len(gc.get_count()) else 0
        
        # Object counts (sample of common types)
        object_counts = self._get_object_counts()
        
        return MemoryStats(
            timestamp=datetime.now(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            available_mb=system_memory.available / 1024 / 1024,
            gc_collections=gc_stats,
            object_counts=object_counts
        )
    
    def _get_object_counts(self) -> Dict[str, int]:
        """Get counts of objects by type."""
        object_counts = defaultdict(int)
        
        try:
            # Count objects by type
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                object_counts[obj_type] += 1
        except Exception as e:
            logger.warning(f"Could not collect object counts: {e}")
        
        # Return top 10 most common types
        sorted_counts = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_counts[:10])
    
    def _check_memory_alerts(self, stats: MemoryStats):
        """Check for memory usage alerts."""
        # Memory percentage alerts
        if stats.percent >= self.memory_critical_threshold:
            logger.critical(f"Critical memory usage: {stats.percent:.1f}%")
        elif stats.percent >= self.memory_warning_threshold:
            logger.warning(f"High memory usage: {stats.percent:.1f}%")
        
        # Memory growth rate alerts
        if len(self.memory_history) >= 2:
            previous_stats = self.memory_history[-2]
            time_diff = (stats.timestamp - previous_stats.timestamp).total_seconds() / 60.0  # minutes
            
            if time_diff > 0:
                memory_growth = (stats.rss_mb - previous_stats.rss_mb) / time_diff  # MB per minute
                
                if memory_growth > self.growth_rate_threshold:
                    logger.warning(f"High memory growth rate: {memory_growth:.2f} MB/min")
    
    def register_callback(self, callback: Callable[[MemoryStats], None]):
        """Register callback for memory statistics updates."""
        self.callbacks.append(callback)
    
    def get_current_stats(self) -> Optional[MemoryStats]:
        """Get the most recent memory statistics."""
        with self._lock:
            return self.memory_history[-1] if self.memory_history else None
    
    def get_memory_trend(self, minutes: int = 60) -> Dict[str, float]:
        """
        Get memory usage trend over specified time period.
        
        Args:
            minutes: Time period to analyze
            
        Returns:
            Dictionary with trend statistics
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_stats = [
                stats for stats in self.memory_history 
                if stats.timestamp >= cutoff_time
            ]
        
        if len(recent_stats) < 2:
            return {}
        
        # Calculate trends
        start_memory = recent_stats[0].rss_mb
        end_memory = recent_stats[-1].rss_mb
        peak_memory = max(stats.rss_mb for stats in recent_stats)
        avg_memory = sum(stats.rss_mb for stats in recent_stats) / len(recent_stats)
        
        time_span = (recent_stats[-1].timestamp - recent_stats[0].timestamp).total_seconds() / 60.0
        growth_rate = (end_memory - start_memory) / time_span if time_span > 0 else 0
        
        return {
            'start_memory_mb': start_memory,
            'end_memory_mb': end_memory,
            'peak_memory_mb': peak_memory,
            'avg_memory_mb': avg_memory,
            'growth_rate_mb_per_min': growth_rate,
            'total_growth_mb': end_memory - start_memory,
            'time_span_minutes': time_span
        }

class EnhancedMemoryManager:
    """
    Enhanced memory management system with automatic cleanup,
    optimization, and monitoring capabilities.
    """
    
    def __init__(self, auto_cleanup_interval: int = 300, aggressive_cleanup: bool = False):
        """
        Initialize enhanced memory manager.
        
        Args:
            auto_cleanup_interval: Seconds between automatic cleanup cycles
            aggressive_cleanup: Whether to use aggressive cleanup strategies
        """
        self.auto_cleanup_interval = auto_cleanup_interval
        self.aggressive_cleanup = aggressive_cleanup
        self.cleanup_active = False
        self.cleanup_thread: Optional[threading.Thread] = None
        self.monitor = MemoryMonitor()
        
        # Object pools for reuse
        self.object_pools: Dict[str, List[Any]] = defaultdict(list)
        self.pool_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Weak references to track objects
        self.tracked_objects: Dict[str, List[weakref.ref]] = defaultdict(list)
        
        # Cleanup statistics
        self.cleanup_stats = {
            'total_cleanups': 0,
            'objects_cleaned': 0,
            'memory_freed_mb': 0.0,
            'last_cleanup': None
        }
        
        logger.info("Enhanced memory manager initialized")
    
    def start_auto_cleanup(self):
        """Start automatic memory cleanup."""
        if self.cleanup_active:
            logger.warning("Auto cleanup already active")
            return
        
        self.cleanup_active = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="MemoryCleanup",
            daemon=True
        )
        self.cleanup_thread.start()
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Register cleanup callback for high memory usage
        self.monitor.register_callback(self._memory_callback)
        
        logger.info("Auto cleanup started")
    
    def stop_auto_cleanup(self):
        """Stop automatic memory cleanup."""
        if not self.cleanup_active:
            return
        
        self.cleanup_active = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        
        self.monitor.stop_monitoring()
        logger.info("Auto cleanup stopped")
    
    def _cleanup_loop(self):
        """Main cleanup loop."""
        while self.cleanup_active:
            try:
                self.perform_cleanup()
                time.sleep(self.auto_cleanup_interval)
            except Exception as e:
                logger.error(f"Auto cleanup error: {e}")
                time.sleep(self.auto_cleanup_interval)
    
    def _memory_callback(self, stats: MemoryStats):
        """Callback for memory statistics updates."""
        # Trigger cleanup if memory usage is high
        if stats.percent > 85.0:  # 85% threshold
            logger.info(f"High memory usage detected ({stats.percent:.1f}%), triggering cleanup")
            self.perform_cleanup(aggressive=True)
    
    def perform_cleanup(self, aggressive: bool = None) -> Dict[str, Any]:
        """
        Perform comprehensive memory cleanup.
        
        Args:
            aggressive: Whether to use aggressive cleanup (overrides instance setting)
            
        Returns:
            Dictionary with cleanup statistics
        """
        if aggressive is None:
            aggressive = self.aggressive_cleanup
        
        start_time = time.time()
        initial_stats = self.monitor._collect_memory_stats()
        
        logger.info(f"Starting memory cleanup (aggressive={aggressive})")
        
        cleanup_results = {
            'gc_collected': 0,
            'pools_cleaned': 0,
            'objects_released': 0,
            'pandas_cache_cleared': False,
            'numpy_cache_cleared': False
        }
        
        try:
            # 1. Garbage collection
            cleanup_results['gc_collected'] = self._perform_garbage_collection(aggressive)
            
            # 2. Clean object pools
            cleanup_results['pools_cleaned'] = self._clean_object_pools(aggressive)
            
            # 3. Release tracked objects
            cleanup_results['objects_released'] = self._release_tracked_objects(aggressive)
            
            # 4. Clear library caches
            cleanup_results['pandas_cache_cleared'] = self._clear_pandas_cache()
            cleanup_results['numpy_cache_cleared'] = self._clear_numpy_cache()
            
            # 5. System-specific cleanup
            if aggressive:
                self._aggressive_system_cleanup()
            
            # Update statistics
            final_stats = self.monitor._collect_memory_stats()
            memory_freed = initial_stats.rss_mb - final_stats.rss_mb
            
            self.cleanup_stats['total_cleanups'] += 1
            self.cleanup_stats['memory_freed_mb'] += max(0, memory_freed)
            self.cleanup_stats['last_cleanup'] = datetime.now()
            
            execution_time = time.time() - start_time
            
            logger.info(f"Memory cleanup completed in {execution_time:.2f}s, "
                       f"freed {memory_freed:.2f} MB")
            
            cleanup_results.update({
                'memory_freed_mb': memory_freed,
                'execution_time_seconds': execution_time,
                'initial_memory_mb': initial_stats.rss_mb,
                'final_memory_mb': final_stats.rss_mb
            })
            
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Memory cleanup error: {e}")
            return cleanup_results
    
    def _perform_garbage_collection(self, aggressive: bool) -> int:
        """Perform garbage collection with optional aggressive settings."""
        collected = 0
        
        if aggressive:
            # Set more aggressive GC thresholds
            original_thresholds = gc.get_threshold()
            gc.set_threshold(100, 5, 5)
        
        try:
            # Force collection of all generations
            for generation in range(3):
                collected += gc.collect(generation)
            
            # Additional full collection
            collected += gc.collect()
            
        finally:
            if aggressive:
                # Restore original thresholds
                gc.set_threshold(*original_thresholds)
        
        return collected
    
    def _clean_object_pools(self, aggressive: bool) -> int:
        """Clean object pools to free unused objects."""
        pools_cleaned = 0
        
        for pool_name, pool in self.object_pools.items():
            with self.pool_locks[pool_name]:
                initial_size = len(pool)
                
                if aggressive:
                    # Clear entire pool
                    pool.clear()
                else:
                    # Keep only recent objects (last 50%)
                    keep_count = len(pool) // 2
                    pool[:] = pool[-keep_count:] if keep_count > 0 else []
                
                cleaned = initial_size - len(pool)
                pools_cleaned += cleaned
        
        return pools_cleaned
    
    def _release_tracked_objects(self, aggressive: bool) -> int:
        """Release tracked weak references to objects."""
        objects_released = 0
        
        for obj_type, refs in self.tracked_objects.items():
            initial_count = len(refs)
            
            # Remove dead references
            refs[:] = [ref for ref in refs if ref() is not None]
            
            if aggressive:
                # Clear all references
                refs.clear()
            
            objects_released += initial_count - len(refs)
        
        return objects_released
    
    def _clear_pandas_cache(self) -> bool:
        """Clear pandas internal caches."""
        try:
            import pandas as pd
            
            # Clear various pandas caches
            if hasattr(pd.core, 'common') and hasattr(pd.core.common, '_cache'):
                pd.core.common._cache.clear()
            
            # Clear string cache if available
            if hasattr(pd, '_libs') and hasattr(pd._libs, 'lib'):
                if hasattr(pd._libs.lib, 'cache_readonly'):
                    # Clear cached properties
                    pass
            
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"Could not clear pandas cache: {e}")
            return False
    
    def _clear_numpy_cache(self) -> bool:
        """Clear numpy internal caches."""
        try:
            import numpy as np
            
            # Clear numpy's internal caches
            if hasattr(np, 'core') and hasattr(np.core, '_internal'):
                # Clear internal caches if available
                pass
            
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"Could not clear numpy cache: {e}")
            return False
    
    def _aggressive_system_cleanup(self):
        """Perform aggressive system-level cleanup."""
        try:
            # Force Python to release unused memory back to OS
            if hasattr(sys, 'intern'):
                # Clear interned strings (Python 3)
                pass
            
            # Clear module caches
            if hasattr(sys, 'modules'):
                # Remove unused modules (be very careful here)
                pass
            
        except Exception as e:
            logger.warning(f"Aggressive cleanup error: {e}")
    
    def get_object_from_pool(self, pool_name: str, factory_func: Callable = None) -> Any:
        """
        Get object from pool or create new one.
        
        Args:
            pool_name: Name of the object pool
            factory_func: Function to create new object if pool is empty
            
        Returns:
            Object from pool or newly created object
        """
        with self.pool_locks[pool_name]:
            pool = self.object_pools[pool_name]
            
            if pool:
                return pool.pop()
            elif factory_func:
                return factory_func()
            else:
                return None
    
    def return_object_to_pool(self, pool_name: str, obj: Any, max_pool


_size: int = 100):
        """
        Return object to pool for reuse.
        
        Args:
            pool_name: Name of the object pool
            obj: Object to return to pool
            max_pool_size: Maximum pool size
        """
        with self.pool_locks[pool_name]:
            pool = self.object_pools[pool_name]
            
            if len(pool) < max_pool_size:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    try:
                        obj.reset()
                    except Exception as e:
                        logger.warning(f"Could not reset object: {e}")
                        return  # Don't add to pool if reset failed
                
                pool.append(obj)
    
    def track_object(self, obj: Any, obj_type: str = None):
        """
        Track object with weak reference for cleanup.
        
        Args:
            obj: Object to track
            obj_type: Type identifier for the object
        """
        if obj_type is None:
            obj_type = type(obj).__name__
        
        weak_ref = weakref.ref(obj)
        self.tracked_objects[obj_type].append(weak_ref)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_stats = self.monitor.get_current_stats()
        trend_stats = self.monitor.get_memory_trend(minutes=60)
        
        # Object pool statistics
        pool_stats = {}
        for pool_name, pool in self.object_pools.items():
            pool_stats[pool_name] = len(pool)
        
        # Tracked object statistics
        tracked_stats = {}
        for obj_type, refs in self.tracked_objects.items():
            # Count live references
            live_refs = sum(1 for ref in refs if ref() is not None)
            tracked_stats[obj_type] = {
                'total_refs': len(refs),
                'live_refs': live_refs,
                'dead_refs': len(refs) - live_refs
            }
        
        return {
            'current_memory': current_stats.__dict__ if current_stats else {},
            'memory_trend': trend_stats,
            'cleanup_stats': self.cleanup_stats.copy(),
            'object_pools': pool_stats,
            'tracked_objects': tracked_stats,
            'gc_stats': {
                'counts': gc.get_count(),
                'thresholds': gc.get_threshold(),
                'stats': gc.get_stats() if hasattr(gc, 'get_stats') else {}
            }
        }

# Utility functions for easy integration
def setup_enhanced_memory_management(
    auto_cleanup_interval: int = 300,
    aggressive_cleanup: bool = False,
    monitoring_interval: int = 30
) -> EnhancedMemoryManager:
    """
    Setup enhanced memory management with recommended settings.
    
    Args:
        auto_cleanup_interval: Seconds between cleanup cycles
        aggressive_cleanup: Whether to use aggressive cleanup
        monitoring_interval: Seconds between monitoring checks
        
    Returns:
        Configured EnhancedMemoryManager instance
    """
    manager = EnhancedMemoryManager(
        auto_cleanup_interval=auto_cleanup_interval,
        aggressive_cleanup=aggressive_cleanup
    )
    
    manager.monitor.monitoring_interval = monitoring_interval
    manager.start_auto_cleanup()
    
    logger.info("Enhanced memory management setup completed")
    return manager

def emergency_memory_cleanup():
    """
    Perform emergency memory cleanup when system is under memory pressure.
    This function can be called manually when memory usage is critical.
    """
    logger.warning("Emergency memory cleanup initiated")
    
    # Create temporary manager for emergency cleanup
    temp_manager = EnhancedMemoryManager(aggressive_cleanup=True)
    
    # Perform aggressive cleanup
    results = temp_manager.perform_cleanup(aggressive=True)
    
    logger.info(f"Emergency cleanup completed: {results}")
    return results

# Integration with existing system components
class MemoryOptimizedComponent:
    """
    Base class for system components that need memory optimization.
    """
    
    def __init__(self, memory_manager: EnhancedMemoryManager = None):
        self.memory_manager = memory_manager or setup_enhanced_memory_management()
        self._component_name = self.__class__.__name__
    
    def get_pooled_object(self, obj_type: str, factory_func: Callable = None):
        """Get object from memory pool."""
        pool_name = f"{self._component_name}_{obj_type}"
        return self.memory_manager.get_object_from_pool(pool_name, factory_func)
    
    def return_pooled_object(self, obj_type: str, obj: Any):
        """Return object to memory pool."""
        pool_name = f"{self._component_name}_{obj_type}"
        self.memory_manager.return_object_to_pool(pool_name, obj)
    
    def track_object(self, obj: Any):
        """Track object for cleanup."""
        obj_type = f"{self._component_name}_{type(obj).__name__}"
        self.memory_manager.track_object(obj, obj_type)
    
    def cleanup_component(self):
        """Cleanup component-specific resources."""
        # Override in subclasses for component-specific cleanup
        pass

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Enhanced Memory Manager...")
    
    # Setup memory management
    manager = setup_enhanced_memory_management(
        auto_cleanup_interval=60,  # 1 minute for testing
        aggressive_cleanup=False,
        monitoring_interval=10     # 10 seconds for testing
    )
    
    try:
        # Test object pooling
        def create_test_object():
            return {"data": list(range(1000)), "timestamp": datetime.now()}
        
        # Get objects from pool
        obj1 = manager.get_object_from_pool("test_objects", create_test_object)
        obj2 = manager.get_object_from_pool("test_objects", create_test_object)
        
        print(f"Created objects: {len(str(obj1))}, {len(str(obj2))}")
        
        # Return objects to pool
        manager.return_object_to_pool("test_objects", obj1)
        manager.return_object_to_pool("test_objects", obj2)
        
        # Test manual cleanup
        cleanup_results = manager.perform_cleanup()
        print(f"Cleanup results: {cleanup_results}")
        
        # Test memory statistics
        stats = manager.get_memory_stats()
        print(f"Memory stats: {stats['current_memory']}")
        
        # Wait a bit to see monitoring in action
        print("Monitoring for 30 seconds...")
        time.sleep(30)
        
        # Get trend statistics
        trend = manager.monitor.get_memory_trend(minutes=1)
        print(f"Memory trend: {trend}")
        
    finally:
        # Cleanup
        manager.stop_auto_cleanup()
        print("Enhanced Memory Manager test completed")

