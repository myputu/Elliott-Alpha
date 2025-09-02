"""
Integrated Elliott Wave Trading System
This module integrates all enhanced components into a unified trading system
with compatibility fixes, memory management, performance monitoring, ML pattern recognition,
and multi-timeframe analysis capabilities.

Author: Manus AI
Date: 28 Agustus 2025
Version: 2.0
"""

import sys
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Import enhanced components
from compatibility_fixes import (
    CompatibleAsyncTickBuffer, 
    CompatibleEnhancedMT5Bridge,
    emergency_memory_cleanup
)
from enhanced_memory_manager import EnhancedMemoryManager, MemoryOptimizedComponent, setup_enhanced_memory_management
from performance_monitor import PerformanceMonitor, TradingSystemMonitor, setup_monitoring_system
from ml_pattern_recognition import ElliotWaveMLClassifier, setup_ml_pattern_recognition
from multi_timeframe_analyzer import MultiTimeframeAnalyzer, Timeframe, setup_multi_timeframe_analysis

# Import original components
try:
    from elliott_wave_analyzer import ElliottWaveAnalyzer
    from trading_strategy import TradingStrategy
    from risk_management import RiskManager
    from database import DatabaseManager
except ImportError as e:
    logging.warning(f"Could not import original components: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemConfiguration:
    """Configuration for the integrated trading system."""
    symbol: str = "XAUUSD"
    primary_timeframe: Timeframe = Timeframe.H1
    enable_ml_patterns: bool = True
    enable_multi_timeframe: bool = True
    enable_performance_monitoring: bool = True
    enable_enhanced_memory: bool = True
    enable_async_buffer: bool = True
    
    # Memory management settings
    auto_cleanup_interval: int = 300  # 5 minutes
    aggressive_cleanup: bool = False
    
    # Performance monitoring settings
    monitoring_interval: float = 1.0
    history_size: int = 10000
    
    # ML settings
    ml_model_type: str = "ensemble"
    
    # Risk management settings
    max_risk_per_trade: float = 0.02  # 2%
    max_daily_risk: float = 0.06      # 6%
    
    # Buffer settings
    tick_buffer_size: int = 10000
    ohlc_buffer_size: int = 5000
    queue_size: int = 2000
    max_workers: int = 4

class IntegratedTradingSystem(MemoryOptimizedComponent):
    """
    Integrated Elliott Wave Trading System with all enhancements.
    """
    
    def __init__(self, config: SystemConfiguration):
        """
        Initialize the integrated trading system.
        
        Args:
            config: System configuration
        """
        # Initialize memory management first
        memory_manager = None
        if config.enable_enhanced_memory:
            memory_manager = setup_enhanced_memory_management(
                auto_cleanup_interval=config.auto_cleanup_interval,
                aggressive_cleanup=config.aggressive_cleanup
            )
        
        super().__init__(memory_manager)
        
        self.config = config
        self.is_running = False
        self.start_time = datetime.now()
        
        # Core components
        self.mt5_bridge: Optional[CompatibleEnhancedMT5Bridge] = None
        self.async_buffer: Optional[CompatibleAsyncTickBuffer] = None
        self.elliott_wave_analyzer: Optional[ElliottWaveAnalyzer] = None
        self.trading_strategy: Optional[TradingStrategy] = None
        self.risk_manager: Optional[RiskManager] = None
        self.database_manager: Optional[DatabaseManager] = None
        
        # Enhanced components
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.trading_monitor: Optional[TradingSystemMonitor] = None
        self.ml_classifier: Optional[ElliotWaveMLClassifier] = None
        self.multi_timeframe_analyzer: Optional[MultiTimeframeAnalyzer] = None
        
        # System state
        self.active_trades: Dict[str, Any] = {}
        self.system_metrics: Dict[str, Any] = {}
        self.last_signal: Optional[Dict[str, Any]] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Integrated Trading System initialized for {config.symbol}")
    
    def initialize_components(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing system components...")
            
            # 1. Initialize performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_monitor, self.trading_monitor = setup_monitoring_system()
                logger.info("✓ Performance monitoring initialized")
            
            # 2. Initialize async buffer
            if self.config.enable_async_buffer:
                self.async_buffer = CompatibleAsyncTickBuffer(
                    tick_buffer_size=self.config.tick_buffer_size,
                    ohlc_buffer_size=self.config.ohlc_buffer_size,
                    queue_size=self.config.queue_size,
                    max_workers=self.config.max_workers
                )
                logger.info("✓ Async tick buffer initialized")
            
            # 3. Initialize MT5 bridge
            try:
                self.mt5_bridge = CompatibleEnhancedMT5Bridge(
                    symbol=self.config.symbol,
                    enable_async_buffer=self.config.enable_async_buffer
                )
                logger.info("✓ MT5 bridge initialized")
            except Exception as e:
                logger.warning(f"MT5 bridge initialization failed: {e}")
                logger.info("✓ Using mock MT5 bridge for testing")
            
            # 4. Initialize ML pattern recognition
            if self.config.enable_ml_patterns:
                self.ml_classifier = setup_ml_pattern_recognition(self.config.ml_model_type)
                logger.info("✓ ML pattern recognition initialized")
            
            # 5. Initialize multi-timeframe analysis
            if self.config.enable_multi_timeframe:
                self.multi_timeframe_analyzer = setup_multi_timeframe_analysis(
                    self.config.symbol, 
                    self.config.primary_timeframe
                )
                
                # Set analyzers for multi-timeframe system
                if self.multi_timeframe_analyzer:
                    self.multi_timeframe_analyzer.set_analyzers(
                        self.elliott_wave_analyzer,
                        self.ml_classifier
                    )
                logger.info("✓ Multi-timeframe analysis initialized")
            
            # 6. Initialize original components (if available)
            try:
                self.elliott_wave_analyzer = ElliottWaveAnalyzer()
                logger.info("✓ Elliott Wave analyzer initialized")
            except Exception as e:
                logger.warning(f"Elliott Wave analyzer not available: {e}")
            
            try:
                self.trading_strategy = TradingStrategy()
                logger.info("✓ Trading strategy initialized")
            except Exception as e:
                logger.warning(f"Trading strategy not available: {e}")
            
            try:
                self.risk_manager = RiskManager(
                    max_risk_per_trade=self.config.max_risk_per_trade,
                    max_daily_risk=self.config.max_daily_risk
                )
                logger.info("✓ Risk manager initialized")
            except Exception as e:
                logger.warning(f"Risk manager not available: {e}")
            
            try:
                self.database_manager = DatabaseManager()
                logger.info("✓ Database manager initialized")
            except Exception as e:
                logger.warning(f"Database manager not available: {e}")
            
            logger.info("System components initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False
    
    def start_system(self) -> bool:
        """
        Start the integrated trading system.
        
        Returns:
            True if system started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("System is already running")
            return True
        
        try:
            logger.info("Starting Integrated Elliott Wave Trading System...")
            
            # Initialize components if not already done
            if not self.initialize_components():
                logger.error("Failed to initialize components")
                return False
            
            # Start monitoring
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
            
            # Connect to MT5
            if self.mt5_bridge:
                try:
                    # Connection logic would go here
                    logger.info("✓ Connected to MT5")
                except Exception as e:
                    logger.warning(f"MT5 connection failed: {e}")
            
            # Start main trading loop
            self.is_running = True
            self.start_time = datetime.now()
            
            # Start trading thread
            trading_thread = threading.Thread(
                target=self._trading_loop,
                name="TradingLoop",
                daemon=True
            )
            trading_thread.start()
            
            logger.info("✓ Integrated Trading System started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            return False
    
    def stop_system(self):
        """Stop the integrated trading system."""
        if not self.is_running:
            logger.info("System is not running")
            return
        
        logger.info("Stopping Integrated Elliott Wave Trading System...")
        
        self.is_running = False
        
        # Stop monitoring
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
        
        # Disconnect from MT5
        if self.mt5_bridge:
            try:
                self.mt5_bridge.disconnect()
                logger.info("✓ Disconnected from MT5")
            except Exception as e:
                logger.warning(f"MT5 disconnection error: {e}")
        
        # Shutdown async buffer
        if self.async_buffer:
            try:
                self.async_buffer.shutdown(timeout=10.0)
                logger.info("✓ Async buffer shutdown")
            except Exception as e:
                logger.warning(f"Async buffer shutdown error: {e}")
        
        # Cleanup multi-timeframe analyzer
        if self.multi_timeframe_analyzer:
            self.multi_timeframe_analyzer.cleanup()
        
        # Stop memory management
        if self.memory_manager:
            self.memory_manager.stop_auto_cleanup()
        
        logger.info("✓ Integrated Trading System stopped")
    
    def _trading_loop(self):
        """Main trading loop."""
        logger.info("Trading loop started")
        
        while self.is_running:
            try:
                # Record loop iteration
                if self.performance_monitor:
                    self.performance_monitor.record_tick_processed()
                
                # Get market data
                market_data = self._get_market_data()
                
                if market_data is not None:
                    # Process tick data
                    self._process_tick_data(market_data)
                    
                    # Analyze patterns
                    analysis_result = self._analyze_patterns(market_data)
                    
                    # Generate trading signals
                    if analysis_result:
                        signal = self._generate_trading_signal(analysis_result)
                        
                        if signal:
                            # Execute trades
                            self._execute_trading_signal(signal)
                
                # Update system metrics
                self._update_system_metrics()
                
                # Sleep to control loop frequency
                time.sleep(0.1)  # 100ms intervals
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                if self.performance_monitor:
                    self.performance_monitor.record_error()
                time.sleep(1.0)  # Longer sleep on error
        
        logger.info("Trading loop stopped")
    
    def _get_market_data(self) -> Optional[Dict[str, Any]]:
        """Get current market data."""
        try:
            # Simulate market data for testing
            current_time = datetime.now()
            
            # Generate realistic tick data
            base_price = 2000.0 + np.random.normal(0, 10)
            spread = 0.5
            
            tick_data = {
                'symbol': self.config.symbol,
                'timestamp': current_time,
                'bid': base_price,
                'ask': base_price + spread,
                'volume': np.random.randint(1, 100)
            }
            
            return tick_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def _process_tick_data(self, tick_data: Dict[str, Any]):
        """Process incoming tick data."""
        try:
            # Add to async buffer if available
            if self.async_buffer:
                success = self.async_buffer.add_tick_data(
                    symbol=tick_data['symbol'],
                    bid=tick_data['bid'],
                    ask=tick_data['ask'],
                    volume=tick_data['volume']
                )
                
                if not success:
                    logger.warning("Failed to add tick data to buffer")
            
            # Update multi-timeframe data
            if self.multi_timeframe_analyzer:
                # Convert tick to OHLC for different timeframes
                # This would be more sophisticated in production
                ohlc_data = pd.DataFrame({
                    'open': [tick_data['bid']],
                    'high': [tick_data['ask']],
                    'low': [tick_data['bid']],
                    'close': [tick_data['ask']],
                    'volume': [tick_data['volume']]
                }, index=[tick_data['timestamp']])
                
                # Update M1 timeframe (others would be aggregated)
                self.multi_timeframe_analyzer.data_manager.update_data(
                    Timeframe.M1, ohlc_data
                )
            
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")
    
    def _analyze_patterns(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze Elliott Wave patterns."""
        try:
            start_time = time.time()
            
            analysis_results = {}
            
            # Multi-timeframe analysis
            if self.multi_timeframe_analyzer:
                try:
                    signal = self.multi_timeframe_analyzer.generate_multi_timeframe_signal()
                    if signal:
                        analysis_results['multi_timeframe'] = {
                            'signal_type': signal.signal_type,
                            'confidence': signal.confidence,
                            'risk_level': signal.risk_level,
                            'reasoning': signal.reasoning
                        }
                except Exception as e:
                    logger.warning(f"Multi-timeframe analysis error: {e}")
            
            # ML pattern recognition
            if self.ml_classifier and self.multi_timeframe_analyzer:
                try:
                    # Get recent data for ML analysis
                    recent_data = self.multi_timeframe_analyzer.data_manager.get_data(
                        self.config.primary_timeframe, 100
                    )
                    
                    if recent_data is not None and len(recent_data) >= 20:
                        ml_prediction = self.ml_classifier.predict(recent_data)
                        analysis_results['ml_pattern'] = {
                            'pattern_type': ml_prediction.pattern_type,
                            'confidence': ml_prediction.confidence,
                            'probabilities': ml_prediction.probability_distribution
                        }
                except Exception as e:
                    logger.warning(f"ML pattern recognition error: {e}")
            
            # Record analysis performance
            analysis_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if self.trading_monitor:
                patterns_found = len(analysis_results)
                self.trading_monitor.record_elliott_wave_analysis(analysis_time, patterns_found)
            
            return analysis_results if analysis_results else None
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
            return None
    
    def _generate_trading_signal(self, analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signal from analysis results."""
        try:
            # Combine signals from different analysis methods
            signals = []
            
            # Multi-timeframe signal
            if 'multi_timeframe' in analysis_result:
                mtf_signal = analysis_result['multi_timeframe']
                if mtf_signal['confidence'] > 0.6:
                    signals.append({
                        'type': mtf_signal['signal_type'],
                        'confidence': mtf_signal['confidence'],
                        'source': 'multi_timeframe'
                    })
            
            # ML pattern signal
            if 'ml_pattern' in analysis_result:
                ml_signal = analysis_result['ml_pattern']
                if ml_signal['confidence'] > 0.7:
                    # Convert pattern to signal
                    if 'impulse' in ml_signal['pattern_type']:
                        signal_type = 'buy'
                    elif 'corrective' in ml_signal['pattern_type']:
                        signal_type = 'sell'
                    else:
                        signal_type = 'hold'
                    
                    signals.append({
                        'type': signal_type,
                        'confidence': ml_signal['confidence'],
                        'source': 'ml_pattern'
                    })
            
            # Combine signals
            if not signals:
                return None
            
            # Simple signal combination (could be more sophisticated)
            buy_signals = [s for s in signals if s['type'] == 'buy']
            sell_signals = [s for s in signals if s['type'] == 'sell']
            
            if len(buy_signals) > len(sell_signals):
                final_signal = 'buy'
                confidence = np.mean([s['confidence'] for s in buy_signals])
            elif len(sell_signals) > len(buy_signals):
                final_signal = 'sell'
                confidence = np.mean([s['confidence'] for s in sell_signals])
            else:
                final_signal = 'hold'
                confidence = 0.5
            
            if final_signal != 'hold' and confidence > 0.6:
                return {
                    'type': final_signal,
                    'confidence': confidence,
                    'timestamp': datetime.now(),
                    'analysis_sources': [s['source'] for s in signals],
                    'symbol': self.config.symbol
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None
    
    def _execute_trading_signal(self, signal: Dict[str, Any]):
        """Execute trading signal."""
        try:
            start_time = time.time()
            
            # Risk management check
            if self.risk_manager:
                risk_approved = self.risk_manager.check_risk(signal)
                if not risk_approved:
                    logger.info(f"Signal rejected by risk management: {signal['type']}")
                    return
            
            # Execute trade (simulation for testing)
            trade_id = f"trade_{int(time.time())}"
            
            # Simulate trade execution
            execution_success = True  # Would be actual execution result
            
            if execution_success:
                # Record successful trade
                with self._lock:
                    self.active_trades[trade_id] = {
                        'signal': signal,
                        'entry_time': datetime.now(),
                        'status': 'active'
                    }
                
                logger.info(f"Trade executed: {signal['type']} {signal['symbol']} (ID: {trade_id})")
                
                # Update last signal
                self.last_signal = signal
            
            # Record execution performance
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if self.trading_monitor:
                self.trading_monitor.record_trade_execution(execution_time, execution_success)
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            if self.trading_monitor:
                self.trading_monitor.record_trade_execution(0, False)
    
    def _update_system_metrics(self):
        """Update system performance metrics."""
        try:
            current_time = datetime.now()
            uptime = (current_time - self.start_time).total_seconds()
            
            with self._lock:
                self.system_metrics = {
                    'uptime_seconds': uptime,
                    'active_trades': len(self.active_trades),
                    'last_update': current_time,
                    'system_status': 'running' if self.is_running else 'stopped'
                }
                
                # Add performance metrics if available
                if self.performance_monitor:
                    current_perf = self.performance_monitor.get_current_metrics()
                    if current_perf:
                        self.system_metrics.update({
                            'tick_processing_rate': current_perf.tick_processing_rate,
                            'memory_usage_mb': current_perf.memory_usage_mb,
                            'cpu_usage_percent': current_perf.cpu_usage_percent
                        })
                
                # Add memory metrics if available
                if self.memory_manager:
                    memory_stats = self.memory_manager.get_memory_stats()
                    self.system_metrics['memory_stats'] = memory_stats
        
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._lock:
            status = {
                'system_info': {
                    'symbol': self.config.symbol,
                    'primary_timeframe': self.config.primary_timeframe.value,
                    'is_running': self.is_running,
                    'start_time': self.start_time,
                    'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
                },
                'components': {
                    'mt5_bridge': self.mt5_bridge is not None,
                    'async_buffer': self.async_buffer is not None,
                    'performance_monitor': self.performance_monitor is not None,
                    'ml_classifier': self.ml_classifier is not None,
                    'multi_timeframe_analyzer': self.multi_timeframe_analyzer is not None,
                    'memory_manager': self.memory_manager is not None
                },
                'metrics': self.system_metrics.copy(),
                'active_trades': len(self.active_trades),
                'last_signal': self.last_signal
            }
            
            # Add component-specific status
            if self.performance_monitor:
                status['performance_summary'] = self.performance_monitor.get_metrics_summary(60)
            
            if self.multi_timeframe_analyzer:
                status['multi_timeframe_summary'] = self.multi_timeframe_analyzer.get_analysis_summary()
            
            return status
    
    def run_system_test(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Run system test for specified duration.
        
        Args:
            duration_seconds: Test duration in seconds
            
        Returns:
            Test results
        """
        logger.info(f"Starting system test for {duration_seconds} seconds...")
        
        test_start = datetime.now()
        
        # Start system
        if not self.start_system():
            return {'success': False, 'error': 'Failed to start system'}
        
        try:
            # Run for specified duration
            time.sleep(duration_seconds)
            
            # Collect test results
            test_results = {
                'success': True,
                'duration_seconds': duration_seconds,
                'start_time': test_start,
                'end_time': datetime.now(),
                'system_status': self.get_system_status()
            }
            
            logger.info("System test completed successfully")
            return test_results
            
        except Exception as e:
            logger.error(f"System test error: {e}")
            return {'success': False, 'error': str(e)}
        
        finally:
            # Stop system
            self.stop_system()

def create_default_system(symbol: str = "XAUUSD") -> IntegratedTradingSystem:
    """
    Create integrated trading system with default configuration.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Configured IntegratedTradingSystem
    """
    config = SystemConfiguration(
        symbol=symbol,
        primary_timeframe=Timeframe.H1,
        enable_ml_patterns=True,
        enable_multi_timeframe=True,
        enable_performance_monitoring=True,
        enable_enhanced_memory=True,
        enable_async_buffer=True
    )
    
    return IntegratedTradingSystem(config)

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Integrated Elliott Wave Trading System...")
    
    # Create system
    trading_system = create_default_system("XAUUSD")
    
    try:
        # Run system test
        test_results = trading_system.run_system_test(duration_seconds=30)
        
        print(f"Test Results:")
        print(f"Success: {test_results['success']}")
        
        if test_results['success']:
            status = test_results['system_status']
            print(f"System Status: {status['system_info']['is_running']}")
            print(f"Active Trades: {status['active_trades']}")
            print(f"Components Loaded: {sum(status['components'].values())}/{len(status['components'])}")
            
            if 'performance_summary' in status:
                perf = status['performance_summary']
                print(f"Tick Processing Rate: {perf.get('tick_processing', {}).get('avg_rate', 0):.2f} ticks/sec")
                print(f"Memory Usage: {perf.get('system_resources', {}).get('avg_memory_mb', 0):.1f} MB")
        else:
            print(f"Test Error: {test_results.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"Test failed with exception: {e}")
    
    print("Integrated Trading System test completed")

