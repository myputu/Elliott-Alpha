"""
Comprehensive Performance Monitoring System for Elliott Wave Trading System
This module provides real-time performance monitoring, metrics collection,
and alerting capabilities for production trading environments.

Author: Manus AI
Date: 28 Agustus 2025
Version: 1.0
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import json
import psutil

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    tick_processing_rate: float  # ticks per second
    signal_processing_time: float  # average milliseconds
    memory_usage_mb: float
    cpu_usage_percent: float
    active_trades: int
    total_signals: int
    error_count: int
    uptime_seconds: float
    latency_ms: float

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system with real-time metrics,
    alerting, and historical analysis capabilities.
    """
    
    def __init__(self, history_size: int = 10000, monitoring_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            history_size: Number of historical metrics to maintain
            monitoring_interval: Seconds between monitoring cycles
        """
        self.history_size = history_size
        self.monitoring_interval = monitoring_interval
        self.metrics_history: deque = deque(maxlen=history_size)
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.start_time = datetime.now()
        
        # Metrics counters
        self.tick_counter = 0
        self.signal_counter = 0
        self.error_counter = 0
        self.processing_times = deque(maxlen=1000)
        self.latency_measurements = deque(maxlen=1000)
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'error_rate': 5.0,  # errors per minute
            'latency_ms': 100.0,
            'processing_time_ms': 50.0
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("Performance monitor initialized")
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        current_time = datetime.now()
        uptime = (current_time - self.start_time).total_seconds()
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.Process().memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Calculate rates
        tick_rate = self.tick_counter / max(uptime, 1)
        
        # Average processing time
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0.0
        )
        
        # Average latency
        avg_latency = (
            sum(self.latency_measurements) / len(self.latency_measurements)
            if self.latency_measurements else 0.0
        )
        
        return PerformanceMetrics(
            timestamp=current_time,
            tick_processing_rate=tick_rate,
            signal_processing_time=avg_processing_time,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            active_trades=0,  # Will be updated by trading system
            total_signals=self.signal_counter,
            error_count=self.error_counter,
            uptime_seconds=uptime,
            latency_ms=avg_latency
        )
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check metrics against alert thresholds."""
        alerts = []
        
        # CPU usage alert
        if metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage']:
            alerts.append({
                'type': 'cpu_usage',
                'severity': 'warning',
                'message': f"High CPU usage: {metrics.cpu_usage_percent:.1f}%",
                'value': metrics.cpu_usage_percent,
                'threshold': self.alert_thresholds['cpu_usage']
            })
        
        # Memory usage alert
        if metrics.memory_usage_mb > self.alert_thresholds['memory_usage']:
            alerts.append({
                'type': 'memory_usage',
                'severity': 'warning',
                'message': f"High memory usage: {metrics.memory_usage_mb:.1f} MB",
                'value': metrics.memory_usage_mb,
                'threshold': self.alert_thresholds['memory_usage']
            })
        
        # Latency alert
        if metrics.latency_ms > self.alert_thresholds['latency_ms']:
            alerts.append({
                'type': 'latency',
                'severity': 'warning',
                'message': f"High latency: {metrics.latency_ms:.1f} ms",
                'value': metrics.latency_ms,
                'threshold': self.alert_thresholds['latency_ms']
            })
        
        # Processing time alert
        if metrics.signal_processing_time > self.alert_thresholds['processing_time_ms']:
            alerts.append({
                'type': 'processing_time',
                'severity': 'warning',
                'message': f"Slow processing: {metrics.signal_processing_time:.1f} ms",
                'value': metrics.signal_processing_time,
                'threshold': self.alert_thresholds['processing_time_ms']
            })
        
        # Error rate alert
        error_rate = self._calculate_error_rate()
        if error_rate > self.alert_thresholds['error_rate']:
            alerts.append({
                'type': 'error_rate',
                'severity': 'critical',
                'message': f"High error rate: {error_rate:.1f} errors/min",
                'value': error_rate,
                'threshold': self.alert_thresholds['error_rate']
            })
        
        # Trigger alert callbacks
        for alert in alerts:
            self._trigger_alert(alert)
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate per minute."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Get metrics from last minute
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= one_minute_ago
        ]
        
        if len(recent_metrics) < 2:
            return 0.0
        
        # Calculate error rate
        start_errors = recent_metrics[0].error_count
        end_errors = recent_metrics[-1].error_count
        
        return end_errors - start_errors
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger alert callbacks."""
        logger.warning(f"ALERT: {alert['message']}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def record_tick_processed(self):
        """Record that a tick was processed."""
        with self._lock:
            self.tick_counter += 1
    
    def record_signal_processed(self, processing_time_ms: float):
        """Record signal processing."""
        with self._lock:
            self.signal_counter += 1
            self.processing_times.append(processing_time_ms)
    
    def record_error(self):
        """Record an error occurrence."""
        with self._lock:
            self.error_counter += 1
    
    def record_latency(self, latency_ms: float):
        """Record latency measurement."""
        with self._lock:
            self.latency_measurements.append(latency_ms)
    
    def update_active_trades(self, count: int):
        """Update active trades count."""
        # This will be reflected in the next metrics collection
        pass
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """
        Get performance metrics summary for specified time period.
        
        Args:
            minutes: Time period to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {}
        
        # Calculate summary statistics
        tick_rates = [m.tick_processing_rate for m in recent_metrics]
        processing_times = [m.signal_processing_time for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
        latencies = [m.latency_ms for m in recent_metrics]
        
        return {
            'time_period_minutes': minutes,
            'total_metrics': len(recent_metrics),
            'tick_processing': {
                'avg_rate': sum(tick_rates) / len(tick_rates),
                'max_rate': max(tick_rates),
                'min_rate': min(tick_rates),
                'total_ticks': recent_metrics[-1].tick_counter if recent_metrics else 0
            },
            'signal_processing': {
                'avg_time_ms': sum(processing_times) / len(processing_times) if processing_times else 0,
                'max_time_ms': max(processing_times) if processing_times else 0,
                'min_time_ms': min(processing_times) if processing_times else 0,
                'total_signals': recent_metrics[-1].total_signals if recent_metrics else 0
            },
            'system_resources': {
                'avg_memory_mb': sum(memory_usage) / len(memory_usage),
                'peak_memory_mb': max(memory_usage),
                'avg_cpu_percent': sum(cpu_usage) / len(cpu_usage),
                'peak_cpu_percent': max(cpu_usage)
            },
            'latency': {
                'avg_ms': sum(latencies) / len(latencies) if latencies else 0,
                'max_ms': max(latencies) if latencies else 0,
                'min_ms': min(latencies) if latencies else 0
            },
            'errors': {
                'total_errors': recent_metrics[-1].error_count if recent_metrics else 0,
                'error_rate_per_minute': self._calculate_error_rate()
            },
            'uptime_hours': recent_metrics[-1].uptime_seconds / 3600 if recent_metrics else 0
        }
    
    def export_metrics(self, filename: str, minutes: int = 60):
        """
        Export metrics to JSON file.
        
        Args:
            filename: Output filename
            minutes: Time period to export
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_metrics = [
                asdict(m) for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
        
        # Convert datetime objects to strings
        for metric in recent_metrics:
            metric['timestamp'] = metric['timestamp'].isoformat()
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'time_period_minutes': minutes,
            'total_metrics': len(recent_metrics),
            'metrics': recent_metrics,
            'summary': self.get_metrics_summary(minutes)
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filename}")

class TradingSystemMonitor:
    """
    Specialized monitor for Elliott Wave trading system components.
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.perf_monitor = performance_monitor
        self.component_metrics = defaultdict(dict)
        self._lock = threading.Lock()
        
        logger.info("Trading system monitor initialized")
    
    def record_elliott_wave_analysis(self, analysis_time_ms: float, patterns_found: int):
        """Record Elliott Wave analysis performance."""
        with self._lock:
            self.component_metrics['elliott_wave'] = {
                'last_analysis_time_ms': analysis_time_ms,
                'patterns_found': patterns_found,
                'timestamp': datetime.now()
            }
        
        # Also record in main performance monitor
        self.perf_monitor.record_signal_processed(analysis_time_ms)
    
    def record_trade_execution(self, execution_time_ms: float, success: bool):
        """Record trade execution performance."""
        with self._lock:
            if 'trade_execution' not in self.component_metrics:
                self.component_metrics['trade_execution'] = {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'failed_trades': 0,
                    'avg_execution_time_ms': 0
                }
            
            metrics = self.component_metrics['trade_execution']
            metrics['total_trades'] += 1
            
            if success:
                metrics['successful_trades'] += 1
            else:
                metrics['failed_trades'] += 1
                self.perf_monitor.record_error()
            
            # Update average execution time
            current_avg = metrics['avg_execution_time_ms']
            total_trades = metrics['total_trades']
            metrics['avg_execution_time_ms'] = (
                (current_avg * (total_trades - 1) + execution_time_ms) / total_trades
            )
            
            metrics['timestamp'] = datetime.now()
    
    def record_data_processing(self, ticks_processed: int, processing_time_ms: float):
        """Record data processing performance."""
        with self._lock:
            self.component_metrics['data_processing'] = {
                'ticks_processed': ticks_processed,
                'processing_time_ms': processing_time_ms,
                'ticks_per_second': ticks_processed / (processing_time_ms / 1000) if processing_time_ms > 0 else 0,
                'timestamp': datetime.now()
            }
        
        # Record individual ticks
        for _ in range(ticks_processed):
            self.perf_monitor.record_tick_processed()
    
    def get_component_metrics(self) -> Dict[str, Any]:
        """Get all component metrics."""
        with self._lock:
            return dict(self.component_metrics)

def setup_monitoring_system() -> tuple[PerformanceMonitor, TradingSystemMonitor]:
    """
    Setup complete monitoring system with recommended configuration.
    
    Returns:
        Tuple of (PerformanceMonitor, TradingSystemMonitor)
    """
    # Create performance monitor
    perf_monitor = PerformanceMonitor(
        history_size=10000,
        monitoring_interval=1.0
    )
    
    # Create trading system monitor
    trading_monitor = TradingSystemMonitor(perf_monitor)
    
    # Setup alert callback for logging
    def log_alert(alert):
        severity = alert.get('severity', 'info').upper()
        message = alert.get('message', 'Unknown alert')
        logger.log(
            logging.CRITICAL if severity == 'CRITICAL' else logging.WARNING,
            f"PERFORMANCE ALERT: {message}"
        )
    
    perf_monitor.register_alert_callback(log_alert)
    
    # Start monitoring
    perf_monitor.start_monitoring()
    
    logger.info("Monitoring system setup completed")
    return perf_monitor, trading_monitor

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Performance Monitor...")
    
    # Setup monitoring
    perf_monitor, trading_monitor = setup_monitoring_system()
    
    try:
        # Simulate some activity
        for i in range(100):
            # Simulate tick processing
            perf_monitor.record_tick_processed()
            
            # Simulate signal processing
            processing_time = 10 + (i % 20)  # 10-30ms
            perf_monitor.record_signal_processed(processing_time)
            
            # Simulate Elliott Wave analysis
            trading_monitor.record_elliott_wave_analysis(
                analysis_time_ms=processing_time,
                patterns_found=i % 3
            )
            
            # Simulate trade execution
            if i % 10 == 0:
                trading_monitor.record_trade_execution(
                    execution_time_ms=50 + (i % 30),
                    success=i % 15 != 0  # Occasional failures
                )
            
            time.sleep(0.1)  # 100ms intervals
        
        # Wait for monitoring to collect data
        time.sleep(5)
        
        # Get current metrics
        current_metrics = perf_monitor.get_current_metrics()
        if current_metrics:
            print(f"Current metrics: {asdict(current_metrics)}")
        
        # Get summary
        summary = perf_monitor.get_metrics_summary(minutes=1)
        print(f"Summary: {summary}")
        
        # Get component metrics
        component_metrics = trading_monitor.get_component_metrics()
        print(f"Component metrics: {component_metrics}")
        
        # Export metrics
        perf_monitor.export_metrics("test_metrics.json", minutes=1)
        print("Metrics exported to test_metrics.json")
        
    finally:
        # Cleanup
        perf_monitor.stop_monitoring()
        print("Performance Monitor test completed")

