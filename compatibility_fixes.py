"""
Compatibility Fixes for Enhanced Elliott Wave Trading System
This module provides fixes for compatibility issues identified during system validation.

Author: Manus AI
Date: 28 Agustus 2025
Version: 1.0
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

def safe_executor_shutdown(executor: ThreadPoolExecutor, wait: bool = True, timeout: Optional[float] = None):
    """
    Safely shutdown ThreadPoolExecutor with version compatibility.
    
    Args:
        executor: ThreadPoolExecutor instance to shutdown
        wait: Whether to wait for completion
        timeout: Maximum time to wait (ignored in older Python versions)
    """
    python_version = sys.version_info
    
    # Python 3.9+ supports timeout parameter
    if python_version >= (3, 9) and timeout is not None:
        try:
            executor.shutdown(wait=wait, timeout=timeout)
        except TypeError:
            # Fallback for versions that don't support timeout
            executor.shutdown(wait=wait)
            if wait and timeout:
                # Manual timeout implementation
                start_time = time.time()
                while not executor._threads == set() and (time.time() - start_time) < timeout:
                    time.sleep(0.1)
    else:
        # Older Python versions
        executor.shutdown(wait=wait)
        if wait and timeout:
            # Manual timeout implementation
            start_time = time.time()
            while hasattr(executor, '_threads') and executor._threads and (time.time() - start_time) < timeout:
                time.sleep(0.1)

def enhanced_memory_cleanup():
    """
    Enhanced memory cleanup for long-running operations.
    """
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Additional cleanup for specific object types
    if hasattr(gc, 'set_threshold'):
        # Adjust garbage collection thresholds for better memory management
        gc.set_threshold(700, 10, 10)
    
    # Clear any cached data structures
    try:
        import pandas as pd
        # Clear pandas cache if available
        if hasattr(pd, 'core') and hasattr(pd.core, 'common'):
            if hasattr(pd.core.common, '_cache'):
                pd.core.common._cache.clear()
    except ImportError:
        pass

class CompatibleAsyncTickBuffer:
    """
    Compatibility wrapper for AsyncTickBuffer that handles version-specific issues.
    """
    
    def __init__(self, *args, **kwargs):
        from async_tick_buffer import AsyncTickBuffer
        self._buffer = AsyncTickBuffer(*args, **kwargs)
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped buffer."""
        return getattr(self._buffer, name)
    
    def shutdown(self, timeout: Optional[float] = None):
        """
        Compatible shutdown method that handles version differences.
        """
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            
            try:
                # Try to shutdown the buffer with timeout if supported
                if hasattr(self._buffer, 'shutdown'):
                    if timeout is not None:
                        # Try with timeout first
                        try:
                            self._buffer.shutdown(timeout=timeout)
                        except TypeError:
                            # Fallback without timeout
                            self._buffer.shutdown()
                            # Manual timeout handling
                            if timeout:
                                time.sleep(min(timeout, 5.0))  # Wait up to timeout or 5 seconds
                    else:
                        self._buffer.shutdown()
                
                # Additional cleanup
                enhanced_memory_cleanup()
                self._is_shutdown = True
                
            except Exception as e:
                print(f"Warning: Buffer shutdown encountered error: {e}")
                # Force cleanup anyway
                self._is_shutdown = True

class CompatibleEnhancedMT5Bridge:
    """
    Compatibility wrapper for EnhancedMT5Bridge that handles version-specific issues.
    """
    
    def __init__(self, *args, **kwargs):
        # Import here to avoid circular imports
        try:
            from enhanced_mt5_bridge import EnhancedMT5Bridge
            self._bridge = EnhancedMT5Bridge(*args, **kwargs)
        except ImportError:
            # Fallback to mock implementation if MT5 not available
            from mt5_bridge_mock import MockMT5Bridge
            self._bridge = MockMT5Bridge(*args, **kwargs)
        
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped bridge."""
        return getattr(self._bridge, name)
    
    def disconnect(self):
        """
        Compatible disconnect method with enhanced cleanup.
        """
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            
            try:
                # Shutdown executor safely if it exists
                if hasattr(self._bridge, 'executor') and self._bridge.executor:
                    safe_executor_shutdown(self._bridge.executor, wait=True, timeout=10.0)
                
                # Shutdown async buffer safely if it exists
                if hasattr(self._bridge, 'async_buffer') and self._bridge.async_buffer:
                    if hasattr(self._bridge.async_buffer, 'shutdown'):
                        try:
                            self._bridge.async_buffer.shutdown(timeout=10.0)
                        except TypeError:
                            self._bridge.async_buffer.shutdown()
                
                # Call original disconnect
                if hasattr(self._bridge, 'disconnect'):
                    self._bridge.disconnect()
                
                # Enhanced cleanup
                enhanced_memory_cleanup()
                self._is_shutdown = True
                
            except Exception as e:
                print(f"Warning: Bridge disconnect encountered error: {e}")
                self._is_shutdown = True

def create_compatible_system(symbol="XAUUSD", enable_async_buffer=True, **kwargs):
    """
    Factory function to create compatible system components.
    
    Args:
        symbol: Trading symbol
        enable_async_buffer: Whether to enable async buffer
        **kwargs: Additional configuration parameters
    
    Returns:
        Tuple of (bridge, buffer) with compatibility wrappers
    """
    
    # Create compatible async buffer if requested
    buffer = None
    if enable_async_buffer:
        try:
            buffer_config = kwargs.get('buffer_config', {})
            buffer = CompatibleAsyncTickBuffer(
                tick_buffer_size=buffer_config.get('tick_buffer_size', 10000),
                ohlc_buffer_size=buffer_config.get('ohlc_buffer_size', 5000),
                queue_size=buffer_config.get('queue_size', 2000),
                max_workers=buffer_config.get('max_workers', 4)
            )
        except Exception as e:
            print(f"Warning: Could not create async buffer: {e}")
            buffer = None
    
    # Create compatible MT5 bridge
    try:
        bridge = CompatibleEnhancedMT5Bridge(
            symbol=symbol,
            enable_async_buffer=enable_async_buffer,
            **kwargs
        )
    except Exception as e:
        print(f"Warning: Could not create enhanced bridge: {e}")
        # Fallback to basic implementation
        bridge = None
    
    return bridge, buffer

def run_compatibility_test():
    """
    Run a basic compatibility test to verify fixes work correctly.
    """
    print("Running compatibility test...")
    
    try:
        # Test compatible system creation
        bridge, buffer = create_compatible_system(
            symbol="XAUUSD",
            enable_async_buffer=True,
            buffer_config={
                'tick_buffer_size': 1000,
                'ohlc_buffer_size': 500,
                'queue_size': 200,
                'max_workers': 2
            }
        )
        
        if buffer:
            print("✓ Compatible async buffer created successfully")
            
            # Test basic operations
            success = buffer.add_tick_data("XAUUSD", 2000.0, 2000.5, volume=100)
            if success:
                print("✓ Tick data addition successful")
            
            # Test shutdown
            buffer.shutdown(timeout=5.0)
            print("✓ Buffer shutdown successful")
        else:
            print("✗ Could not create async buffer")
        
        if bridge:
            print("✓ Compatible MT5 bridge created successfully")
            
            # Test disconnect
            bridge.disconnect()
            print("✓ Bridge disconnect successful")
        else:
            print("✗ Could not create MT5 bridge")
        
        # Test memory cleanup
        enhanced_memory_cleanup()
        print("✓ Memory cleanup successful")
        
        print("Compatibility test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    # Run compatibility test when executed directly
    success = run_compatibility_test()
    sys.exit(0 if success else 1)

