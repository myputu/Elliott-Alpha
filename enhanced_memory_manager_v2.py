#!/usr/bin/env python3
"""
Enhanced Memory Manager v2.0 - Phase 1 Reliability Enhancement
Auto-cleanup, memory tracking, and emergency management for 24/7 stability

Author: Manus AI
Version: 2.0 - Reliability Enhanced
"""

import threading
import time
import gc
import psutil
import logging
import weakref
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class MemoryMetrics:
    """Memory usage metrics and statistics"""
    current_usage_mb: float = 0.0
    peak_usage_mb: float = 0.0
    average_usage_mb: float = 0.0
    cleanup_count: int = 0
    emergency_cleanup_count: int = 0
    objects_cleaned: int = 0
    last_cleanup: datetime = field(default_factory=datetime.now)
    last_emergency: Optional[datetime] = None
    memory_trend: List[float] = field(default_factory=list)
    gc_collections: Dict[str, int] = field(default_factory=lambda: {'gen0': 0, 'gen1': 0, 'gen2': 0})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization"""
        return {
            'current_usage_mb': self.current_usage_mb,
            'peak_usage_mb': self.peak_usage_mb,
            'average_usage_mb': self.average_usage_mb,
            'cleanup_count': self.cleanup_count,
            'emergency_cleanup_count': self.emergency_cleanup_count,
            'objects_cleaned': self.objects_cleaned,
            'last_cleanup': self.last_cleanup.isoformat(),
            'last_emergency': self.last_emergency.isoformat() if self.last_emergency else None,
            'memory_trend_last_10': self.memory_trend[-10:] if self.memory_trend else [],
            'gc_collections': self.gc_collections.copy(),
            'memory_efficiency': self._calculate_efficiency(),
            'trend_direction': self._calculate_trend()
        }
    
    def _calculate_efficiency(self) -> float:
        """Calculate memory efficiency score (0-100)"""
        if self.peak_usage_mb == 0:
            return 100.0
        
        efficiency = (1 - (self.current_usage_mb / self.peak_usage_mb)) * 100
        return max(0.0, min(100.0, efficiency))
    
    def _calculate_trend(self) -> str:
        """Calculate memory usage trend"""
        if len(self.memory_trend) < 5:
            return "STABLE"
        
        recent = self.memory_trend[-5:]
        if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
            return "INCREASING"
        elif all(recent[i] >= recent[i+1] for i in range(len(recent)-1)):
            return "DECREASING"
        else:
            return "STABLE"

class ObjectTracker:
    """Track and manage object lifecycle for cleanup"""
    
    def __init__(self):
        self._tracked_objects: Dict[str, Set[weakref.ref]] = defaultdict(set)
        self._object_counts: Dict[str, int] = defaultdict(int)
        self._creation_times: Dict[int, datetime] = {}
        self._lock = threading.RLock()
    
    def track_object(self, obj: Any, category: str = "general", max_age_minutes: int = 30):
        """Track an object for automatic cleanup"""
        with self._lock:
            obj_id = id(obj)
            self._creation_times[obj_id] = datetime.now()
            
            # Create weak reference with cleanup callback
            def cleanup_callback(ref):
                with self._lock:
                    self._tracked_objects[category].discard(ref)
                    if obj_id in self._creation_times:
                        del self._creation_times[obj_id]
            
            weak_ref = weakref.ref(obj, cleanup_callback)
            self._tracked_objects[category].add(weak_ref)
            self._object_counts[category] += 1
    
    def cleanup_old_objects(self, max_age_minutes: int = 30) -> int:
        """Clean up objects older than specified age"""
        cleaned_count = 0
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        
        with self._lock:
            # Find objects to clean
            objects_to_remove = []
            
            for obj_id, creation_time in list(self._creation_times.items()):
                if creation_time < cutoff_time:
                    objects_to_remove.append(obj_id)
            
            # Remove old objects
            for obj_id in objects_to_remove:
                if obj_id in self._creation_times:
                    del self._creation_times[obj_id]
                    cleaned_count += 1
            
            # Clean up dead weak references
            for category in self._tracked_objects:
                dead_refs = [ref for ref in self._tracked_objects[category] if ref() is None]
                for ref in dead_refs:
                    self._tracked_objects[category].discard(ref)
        
        return cleaned_count
    
    def get_object_counts(self) -> Dict[str, int]:
        """Get current object counts by category"""
        with self._lock:
            counts = {}
            for category, refs in self._tracked_objects.items():
                # Count only live references
                live_count = sum(1 for ref in refs if ref() is not None)
                counts[category] = live_count
            return counts
    
    def force_cleanup_category(self, category: str) -> int:
        """Force cleanup of all objects in a category"""
        with self._lock:
            if category not in self._tracked_objects:
                return 0
            
            refs_to_clean = list(self._tracked_objects[category])
            cleaned_count = 0
            
            for ref in refs_to_clean:
                obj = ref()
                if obj is not None:
                    # Try to clean the object
                    try:
                        if hasattr(obj, 'cleanup'):
                            obj.cleanup()
                        elif hasattr(obj, 'close'):
                            obj.close()
                        elif hasattr(obj, 'clear'):
                            obj.clear()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Error cleaning object: {e}")
                
                self._tracked_objects[category].discard(ref)
            
            return cleaned_count

class EnhancedMemoryManagerV2:
    """
    Enhanced Memory Manager v2.0 with auto-cleanup, tracking, and emergency management
    Designed for 24/7 trading system reliability according to Phase 1 specifications
    """
    
    def __init__(self, 
                 cleanup_interval: int = 300,  # 5 minutes as specified
                 warning_threshold: float = 0.8,  # 80% as specified
                 emergency_threshold: float = 0.9,  # 90%
                 critical_threshold: float = 0.95,  # 95%
                 max_memory_mb: int = 1024):  # 1GB default
        
        self.cleanup_interval = cleanup_interval
        self.warning_threshold = warning_threshold
        self.emergency_threshold = emergency_threshold
        self.critical_threshold = critical_threshold
        self.max_memory_mb = max_memory_mb
        
        # Core components
        self.metrics = MemoryMetrics()
        self.object_tracker = ObjectTracker()
        
        # Control flags
        self._running = False
        self._emergency_mode = False
        self._cleanup_paused = False
        self._trade_execution_paused = False  # For emergency pause
        
        # Thread synchronization
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._monitor_thread = None
        
        # Callbacks
        self._cleanup_callbacks: List[Callable[[], int]] = []
        self._warning_callbacks: List[Callable[[float], None]] = []
        self._emergency_callbacks: List[Callable[[float], None]] = []
        
        # Memory tracking
        self._memory_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self._last_gc_stats = gc.get_stats()
        
        # Emergency state
        self._emergency_start_time = None
        self._consecutive_emergencies = 0
        
        logger.info(f"Enhanced Memory Manager v2.0 initialized: max={max_memory_mb}MB, "
                   f"warning={warning_threshold*100:.0f}%, emergency={emergency_threshold*100:.0f}%")
    
    def start(self):
        """Start the memory management system"""
        with self._lock:
            if self._running:
                logger.warning("Memory manager already running")
                return
            
            self._running = True
            self._emergency_mode = False
            self._trade_execution_paused = False
            
            # Start background threads
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True, name="MemoryCleanup")
            self._monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True, name="MemoryMonitor")
            
            self._cleanup_thread.start()
            self._monitor_thread.start()
            
            # Initial memory measurement
            self._update_memory_metrics()
            
            logger.info("Enhanced Memory Manager v2.0 started successfully")
    
    def stop(self, timeout: float = 10.0):
        """Stop the memory management system"""
        with self._lock:
            if not self._running:
                return
            
            logger.info("Stopping Enhanced Memory Manager v2.0...")
            self._running = False
            
            # Wait for threads to finish
            threads = [self._cleanup_thread, self._monitor_thread]
            
            for thread in threads:
                if thread and thread.is_alive():
                    thread.join(timeout=timeout/len(threads))
            
            # Final cleanup
            self._perform_cleanup(force=True)
            
            logger.info("Enhanced Memory Manager v2.0 stopped")
    
    def _cleanup_worker(self):
        """Background cleanup worker thread - runs every 300 seconds as specified"""
        logger.info("Memory cleanup worker started (300s interval)")
        
        while self._running:
            try:
                if not self._cleanup_paused:
                    self._perform_cleanup()
                
                # Sleep with interruption check (300 seconds = 5 minutes)
                for _ in range(self.cleanup_interval):
                    if not self._running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                time.sleep(30)  # Longer pause on error
        
        logger.info("Memory cleanup worker stopped")
    
    def _monitor_worker(self):
        """Background memory monitoring worker"""
        logger.info("Memory monitor worker started")
        
        while self._running:
            try:
                self._update_memory_metrics()
                self._check_memory_thresholds()
                
                # Sleep with interruption check
                for _ in range(10):  # Check every 10 seconds
                    if not self._running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in monitor worker: {e}")
                time.sleep(30)
        
        logger.info("Memory monitor worker stopped")
    
    def _update_memory_metrics(self):
        """Update current memory metrics using psutil as specified"""
        try:
            # Get current memory usage using psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            current_mb = memory_info.rss / 1024 / 1024
            
            with self._lock:
                # Update metrics
                self.metrics.current_usage_mb = current_mb
                
                if current_mb > self.metrics.peak_usage_mb:
                    self.metrics.peak_usage_mb = current_mb
                
                # Update memory history
                self._memory_history.append(current_mb)
                self.metrics.memory_trend = list(self._memory_history)
                
                # Calculate average
                if self._memory_history:
                    self.metrics.average_usage_mb = sum(self._memory_history) / len(self._memory_history)
                
                # Update GC stats
                current_gc_stats = gc.get_stats()
                for i, stats in enumerate(current_gc_stats):
                    gen_name = f"gen{i}"
                    if gen_name in self.metrics.gc_collections:
                        self.metrics.gc_collections[gen_name] = stats.get('collections', 0)
        
        except Exception as e:
            logger.error(f"Error updating memory metrics: {e}")
    
    def _check_memory_thresholds(self):
        """Check memory usage against thresholds and take action as specified"""
        current_usage = self.metrics.current_usage_mb
        usage_ratio = current_usage / self.max_memory_mb
        
        if usage_ratio >= self.critical_threshold:
            # Critical level - immediate emergency action
            logger.critical(f"CRITICAL memory usage: {current_usage:.1f}MB ({usage_ratio:.1%})")
            self._trigger_emergency_cleanup()
            
        elif usage_ratio >= self.emergency_threshold:
            # Emergency level - >80% as specified
            if not self._emergency_mode:
                logger.error(f"EMERGENCY memory usage: {current_usage:.1f}MB ({usage_ratio:.1%})")
                self._trigger_emergency_cleanup()
                self._pause_trade_execution()  # Pause new trade execution as specified
            
        elif usage_ratio >= self.warning_threshold:
            # Warning level - 80% as specified
            if not self._emergency_mode:  # Don't spam warnings during emergency
                logger.warning(f"HIGH memory usage: {current_usage:.1f}MB ({usage_ratio:.1%})")
                self._trigger_warning_callbacks(usage_ratio)
                
                # Trigger preventive cleanup
                self._perform_cleanup(aggressive=True)
        
        else:
            # Normal level - reset emergency mode
            if self._emergency_mode:
                logger.info(f"Memory usage normalized: {current_usage:.1f}MB ({usage_ratio:.1%})")
                self._emergency_mode = False
                self._emergency_start_time = None
                self._consecutive_emergencies = 0
                self._resume_trade_execution()
    
    def _perform_cleanup(self, force: bool = False, aggressive: bool = False):
        """Perform memory cleanup operations as specified"""
        if self._cleanup_paused and not force:
            return
        
        logger.debug(f"Performing {'aggressive' if aggressive else 'regular'} cleanup...")
        
        cleanup_start = time.time()
        total_cleaned = 0
        
        try:
            with self._lock:
                # 1. Clean tracked objects (tick/history data that's not used)
                if aggressive:
                    # Clean objects older than 15 minutes
                    cleaned = self.object_tracker.cleanup_old_objects(max_age_minutes=15)
                else:
                    # Clean objects older than 30 minutes
                    cleaned = self.object_tracker.cleanup_old_objects(max_age_minutes=30)
                
                total_cleaned += cleaned
                
                # 2. Execute custom cleanup callbacks
                for callback in self._cleanup_callbacks:
                    try:
                        cleaned = callback()
                        if isinstance(cleaned, int):
                            total_cleaned += cleaned
                    except Exception as e:
                        logger.error(f"Error in cleanup callback: {e}")
                
                # 3. Force garbage collection
                if aggressive or force:
                    # Aggressive GC
                    for generation in range(3):
                        collected = gc.collect(generation)
                        total_cleaned += collected
                        logger.debug(f"GC generation {generation}: collected {collected} objects")
                else:
                    # Regular GC
                    collected = gc.collect()
                    total_cleaned += collected
                
                # 4. Update metrics
                self.metrics.cleanup_count += 1
                self.metrics.objects_cleaned += total_cleaned
                self.metrics.last_cleanup = datetime.now()
                
                cleanup_time = time.time() - cleanup_start
                logger.debug(f"Cleanup completed: {total_cleaned} objects cleaned in {cleanup_time:.2f}s")
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _trigger_emergency_cleanup(self):
        """Trigger emergency cleanup procedures as specified"""
        if self._emergency_mode:
            return  # Already in emergency mode
        
        logger.critical("TRIGGERING EMERGENCY CLEANUP!")
        
        with self._lock:
            self._emergency_mode = True
            self._emergency_start_time = datetime.now()
            self._consecutive_emergencies += 1
            self.metrics.emergency_cleanup_count += 1
            self.metrics.last_emergency = datetime.now()
        
        # Execute emergency callbacks
        current_usage = self.metrics.current_usage_mb
        usage_ratio = current_usage / self.max_memory_mb
        
        for callback in self._emergency_callbacks:
            try:
                callback(usage_ratio)
            except Exception as e:
                logger.error(f"Error in emergency callback: {e}")
        
        # Aggressive cleanup
        self._perform_cleanup(force=True, aggressive=True)
        
        # Force cleanup of all tracked object categories
        for category in ['tick_data', 'ohlc_data', 'analysis_results', 'temporary']:
            cleaned = self.object_tracker.force_cleanup_category(category)
            logger.info(f"Emergency cleanup of {category}: {cleaned} objects")
        
        # If still in emergency after cleanup, consider more drastic measures
        self._update_memory_metrics()
        if self.metrics.current_usage_mb / self.max_memory_mb >= self.critical_threshold:
            logger.critical("Emergency cleanup insufficient - pausing new operations")
            self._pause_trade_execution()
    
    def _pause_trade_execution(self):
        """Pause new trade execution as specified when memory >80%"""
        if not self._trade_execution_paused:
            self._trade_execution_paused = True
            logger.critical("PAUSING NEW TRADE EXECUTION due to high memory usage")
            
            # Notify all emergency callbacks about trade pause
            for callback in self._emergency_callbacks:
                try:
                    callback(self.metrics.current_usage_mb / self.max_memory_mb)
                except Exception as e:
                    logger.error(f"Error in emergency callback: {e}")
    
    def _resume_trade_execution(self):
        """Resume trade execution when memory usage normalizes"""
        if self._trade_execution_paused:
            self._trade_execution_paused = False
            logger.info("RESUMING TRADE EXECUTION - memory usage normalized")
    
    def _trigger_warning_callbacks(self, usage_ratio: float):
        """Trigger warning callbacks"""
        for callback in self._warning_callbacks:
            try:
                callback(usage_ratio)
            except Exception as e:
                logger.error(f"Error in warning callback: {e}")
    
    def track_object(self, obj: Any, category: str = "general", max_age_minutes: int = 30):
        """Track an object for automatic cleanup"""
        self.object_tracker.track_object(obj, category, max_age_minutes)
    
    def add_cleanup_callback(self, callback: Callable[[], int]):
        """Add a custom cleanup callback"""
        self._cleanup_callbacks.append(callback)
        logger.info(f"Added cleanup callback: {callback.__name__}")
    
    def add_warning_callback(self, callback: Callable[[float], None]):
        """Add a warning callback"""
        self._warning_callbacks.append(callback)
        logger.info(f"Added warning callback: {callback.__name__}")
    
    def add_emergency_callback(self, callback: Callable[[float], None]):
        """Add an emergency callback"""
        self._emergency_callbacks.append(callback)
        logger.info(f"Added emergency callback: {callback.__name__}")
    
    def force_cleanup(self):
        """Force immediate cleanup"""
        logger.info("Forcing immediate cleanup")
        self._perform_cleanup(force=True, aggressive=True)
    
    def pause_cleanup(self):
        """Pause automatic cleanup"""
        self._cleanup_paused = True
        logger.info("Automatic cleanup paused")
    
    def resume_cleanup(self):
        """Resume automatic cleanup"""
        self._cleanup_paused = False
        logger.info("Automatic cleanup resumed")
    
    def is_trade_execution_paused(self) -> bool:
        """Check if trade execution is paused due to memory issues"""
        return self._trade_execution_paused
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current memory metrics"""
        with self._lock:
            metrics = self.metrics.to_dict()
            metrics['object_counts'] = self.object_tracker.get_object_counts()
            metrics['emergency_mode'] = self._emergency_mode
            metrics['cleanup_paused'] = self._cleanup_paused
            metrics['trade_execution_paused'] = self._trade_execution_paused
            metrics['consecutive_emergencies'] = self._consecutive_emergencies
            
            if self._emergency_start_time:
                emergency_duration = (datetime.now() - self._emergency_start_time).total_seconds()
                metrics['emergency_duration_seconds'] = emergency_duration
            
            return metrics
    
    def get_status_summary(self) -> str:
        """Get a brief status summary"""
        metrics = self.get_metrics()
        usage_ratio = metrics['current_usage_mb'] / self.max_memory_mb
        
        status = "EMERGENCY" if self._emergency_mode else "NORMAL"
        trade_status = "PAUSED" if self._trade_execution_paused else "ACTIVE"
        
        return (f"Memory: {metrics['current_usage_mb']:.1f}MB/{self.max_memory_mb}MB "
                f"({usage_ratio:.1%}) | Peak: {metrics['peak_usage_mb']:.1f}MB | "
                f"Status: {status} | Trading: {trade_status} | Cleanups: {metrics['cleanup_count']} | "
                f"Objects: {sum(metrics['object_counts'].values())}")
    
    def is_healthy(self) -> bool:
        """Check if memory management is healthy"""
        if not self._running:
            return False
        
        usage_ratio = self.metrics.current_usage_mb / self.max_memory_mb
        
        # Not healthy if in emergency mode for too long
        if self._emergency_mode and self._emergency_start_time:
            emergency_duration = (datetime.now() - self._emergency_start_time).total_seconds()
            if emergency_duration > 300:  # 5 minutes
                return False
        
        # Not healthy if usage is too high
        if usage_ratio >= self.emergency_threshold:
            return False
        
        # Not healthy if too many consecutive emergencies
        if self._consecutive_emergencies > 5:
            return False
        
        return True
    
    def save_metrics_to_file(self, filepath: str):
        """Save current metrics to JSON file for logging as specified"""
        try:
            metrics = self.get_metrics()
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.debug(f"Memory metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics to file: {e}")
    
    def load_metrics_from_file(self, filepath: str) -> bool:
        """Load metrics from JSON file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Restore relevant metrics
                if 'peak_usage_mb' in data:
                    self.metrics.peak_usage_mb = data['peak_usage_mb']
                if 'cleanup_count' in data:
                    self.metrics.cleanup_count = data['cleanup_count']
                if 'emergency_cleanup_count' in data:
                    self.metrics.emergency_cleanup_count = data['emergency_cleanup_count']
                
                logger.info(f"Memory metrics loaded from {filepath}")
                return True
        except Exception as e:
            logger.error(f"Error loading metrics from file: {e}")
        
        return False

# Example usage and testing
if __name__ == "__main__":
    import random
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create memory manager with Phase 1 specifications
    memory_manager = EnhancedMemoryManagerV2(
        cleanup_interval=300,  # 5 minutes as specified
        warning_threshold=0.8,  # 80% as specified
        max_memory_mb=100     # 100MB limit for testing
    )
    
    # Add custom cleanup callback
    def custom_cleanup():
        # Simulate cleaning some objects (tick/history data)
        cleaned = random.randint(5, 20)
        logger.info(f"Custom cleanup: cleaned {cleaned} objects")
        return cleaned
    
    memory_manager.add_cleanup_callback(custom_cleanup)
    
    # Add warning callback
    def memory_warning(usage_ratio):
        logger.warning(f"Memory warning triggered at {usage_ratio:.1%}")
    
    memory_manager.add_warning_callback(memory_warning)
    
    # Add emergency callback
    def memory_emergency(usage_ratio):
        logger.critical(f"Memory emergency triggered at {usage_ratio:.1%}")
    
    memory_manager.add_emergency_callback(memory_emergency)
    
    # Start memory manager
    memory_manager.start()
    
    try:
        # Simulate some objects to track
        test_objects = []
        
        for i in range(50):
            # Create some test objects (simulate tick data)
            obj = [random.random() for _ in range(1000)]  # List of random numbers
            test_objects.append(obj)
            
            # Track the object as tick_data
            memory_manager.track_object(obj, category="tick_data", max_age_minutes=1)
            
            time.sleep(0.1)
        
        # Wait and observe
        time.sleep(10)
        
        # Print metrics
        print("\nMemory Metrics:")
        metrics = memory_manager.get_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        print(f"\nStatus: {memory_manager.get_status_summary()}")
        print(f"Healthy: {memory_manager.is_healthy()}")
        print(f"Trade Execution Paused: {memory_manager.is_trade_execution_paused()}")
        
        # Force cleanup
        memory_manager.force_cleanup()
        
        # Wait a bit more
        time.sleep(5)
        
        print(f"\nAfter cleanup: {memory_manager.get_status_summary()}")
        
    finally:
        memory_manager.stop()
        print("Memory manager stopped")

