"""
Multi-Timeframe Analysis System for Elliott Wave Trading
This module provides comprehensive analysis across multiple timeframes
to improve Elliott Wave pattern recognition and trading accuracy.

Author: Manus AI
Date: 28 Agustus 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logger = logging.getLogger(__name__)

class Timeframe(Enum):
    """Supported timeframes for analysis."""
    M1 = "1min"
    M5 = "5min"
    M15 = "15min"
    M30 = "30min"
    H1 = "1H"
    H4 = "4H"
    D1 = "1D"
    W1 = "1W"
    MN1 = "1M"

@dataclass
class TimeframeAnalysis:
    """Analysis result for a specific timeframe."""
    timeframe: Timeframe
    pattern_type: str
    confidence: float
    wave_count: Dict[str, int]
    trend_direction: str  # "bullish", "bearish", "sideways"
    support_resistance: Dict[str, float]
    fibonacci_levels: Dict[str, float]
    volume_analysis: Dict[str, float]
    momentum_indicators: Dict[str, float]
    timestamp: datetime
    data_quality: float  # 0-1 score for data completeness

@dataclass
class MultiTimeframeSignal:
    """Combined signal from multiple timeframe analysis."""
    primary_timeframe: Timeframe
    signal_type: str  # "buy", "sell", "hold"
    confidence: float
    timeframe_alignment: Dict[Timeframe, str]  # Agreement across timeframes
    risk_level: str  # "low", "medium", "high"
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    reasoning: str
    timestamp: datetime

class TimeframeDataManager:
    """
    Manages market data across multiple timeframes with automatic
    synchronization and data quality validation.
    """
    
    def __init__(self, symbol: str):
        """
        Initialize timeframe data manager.
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
        """
        self.symbol = symbol
        self.data_cache: Dict[Timeframe, pd.DataFrame] = {}
        self.last_update: Dict[Timeframe, datetime] = {}
        self.data_quality: Dict[Timeframe, float] = {}
        self._lock = threading.Lock()
        
        # Data retention periods (in number of bars)
        self.retention_periods = {
            Timeframe.M1: 1440,   # 24 hours
            Timeframe.M5: 2016,   # 7 days
            Timeframe.M15: 2016,  # 3 weeks
            Timeframe.M30: 1440,  # 30 days
            Timeframe.H1: 720,    # 30 days
            Timeframe.H4: 720,    # 120 days
            Timeframe.D1: 365,    # 1 year
            Timeframe.W1: 260,    # 5 years
            Timeframe.MN1: 120    # 10 years
        }
        
        logger.info(f"Timeframe data manager initialized for {symbol}")
    
    def update_data(self, timeframe: Timeframe, new_data: pd.DataFrame):
        """
        Update data for specific timeframe.
        
        Args:
            timeframe: Target timeframe
            new_data: New OHLCV data
        """
        with self._lock:
            if timeframe not in self.data_cache:
                self.data_cache[timeframe] = pd.DataFrame()
            
            # Merge new data with existing
            if not self.data_cache[timeframe].empty:
                # Remove duplicates and sort
                combined_data = pd.concat([self.data_cache[timeframe], new_data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data = combined_data.sort_index()
            else:
                combined_data = new_data.copy()
            
            # Apply retention period
            max_bars = self.retention_periods.get(timeframe, 1000)
            if len(combined_data) > max_bars:
                combined_data = combined_data.tail(max_bars)
            
            self.data_cache[timeframe] = combined_data
            self.last_update[timeframe] = datetime.now()
            
            # Calculate data quality
            self.data_quality[timeframe] = self._calculate_data_quality(combined_data)
            
            logger.debug(f"Updated {timeframe.value} data: {len(combined_data)} bars")
    
    def get_data(self, timeframe: Timeframe, bars: int = None) -> Optional[pd.DataFrame]:
        """
        Get data for specific timeframe.
        
        Args:
            timeframe: Target timeframe
            bars: Number of recent bars to return (None for all)
            
        Returns:
            DataFrame with OHLCV data or None if not available
        """
        with self._lock:
            if timeframe not in self.data_cache:
                return None
            
            data = self.data_cache[timeframe].copy()
            
            if bars is not None and len(data) > bars:
                data = data.tail(bars)
            
            return data if not data.empty else None
    
    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """
        Calculate data quality score (0-1).
        
        Args:
            data: OHLCV data
            
        Returns:
            Quality score
        """
        if data.empty:
            return 0.0
        
        quality_factors = []
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        quality_factors.append(1.0 - missing_ratio)
        
        # Check for data consistency (OHLC relationships)
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= Open, Close, Low
            high_consistency = (
                (data['high'] >= data['open']).sum() +
                (data['high'] >= data['close']).sum() +
                (data['high'] >= data['low']).sum()
            ) / (len(data) * 3)
            quality_factors.append(high_consistency)
            
            # Low should be <= Open, Close, High
            low_consistency = (
                (data['low'] <= data['open']).sum() +
                (data['low'] <= data['close']).sum() +
                (data['low'] <= data['high']).sum()
            ) / (len(data) * 3)
            quality_factors.append(low_consistency)
        
        # Check for reasonable price movements (no extreme gaps)
        if 'close' in data.columns and len(data) > 1:
            price_changes = data['close'].pct_change().dropna()
            extreme_moves = (abs(price_changes) > 0.1).sum()  # >10% moves
            stability_score = 1.0 - (extreme_moves / len(price_changes))
            quality_factors.append(max(0.5, stability_score))  # Minimum 0.5 for this factor
        
        # Check data freshness
        if self.last_update.get(Timeframe.M1):  # Use M1 as reference
            time_since_update = (datetime.now() - self.last_update[Timeframe.M1]).total_seconds()
            freshness_score = max(0.0, 1.0 - (time_since_update / 3600))  # Decay over 1 hour
            quality_factors.append(freshness_score)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def get_data_status(self) -> Dict[str, Any]:
        """Get status of all timeframe data."""
        with self._lock:
            status = {}
            for timeframe in Timeframe:
                if timeframe in self.data_cache:
                    data = self.data_cache[timeframe]
                    status[timeframe.value] = {
                        'bars_available': len(data),
                        'last_update': self.last_update.get(timeframe),
                        'data_quality': self.data_quality.get(timeframe, 0.0),
                        'date_range': {
                            'start': data.index[0] if not data.empty else None,
                            'end': data.index[-1] if not data.empty else None
                        }
                    }
                else:
                    status[timeframe.value] = {
                        'bars_available': 0,
                        'last_update': None,
                        'data_quality': 0.0,
                        'date_range': {'start': None, 'end': None}
                    }
            
            return status

class MultiTimeframeAnalyzer:
    """
    Advanced multi-timeframe Elliott Wave analyzer that provides
    comprehensive analysis across different time horizons.
    """
    
    def __init__(self, symbol: str, primary_timeframe: Timeframe = Timeframe.H1):
        """
        Initialize multi-timeframe analyzer.
        
        Args:
            symbol: Trading symbol
            primary_timeframe: Primary timeframe for trading decisions
        """
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.data_manager = TimeframeDataManager(symbol)
        
        # Analysis components (would be imported from other modules)
        self.elliott_wave_analyzer = None  # Will be set externally
        self.ml_classifier = None          # Will be set externally
        
        # Timeframe hierarchy for analysis
        self.timeframe_hierarchy = {
            Timeframe.M1: 1,
            Timeframe.M5: 2,
            Timeframe.M15: 3,
            Timeframe.M30: 4,
            Timeframe.H1: 5,
            Timeframe.H4: 6,
            Timeframe.D1: 7,
            Timeframe.W1: 8,
            Timeframe.MN1: 9
        }
        
        # Analysis cache
        self.analysis_cache: Dict[Timeframe, TimeframeAnalysis] = {}
        self.cache_expiry: Dict[Timeframe, datetime] = {}
        
        # Thread pool for parallel analysis
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="MTF_Analyzer")
        
        self._lock = threading.Lock()
        
        logger.info(f"Multi-timeframe analyzer initialized for {symbol}")
    
    def set_analyzers(self, elliott_wave_analyzer, ml_classifier):
        """
        Set the Elliott Wave analyzer and ML classifier.
        
        Args:
            elliott_wave_analyzer: Elliott Wave analysis component
            ml_classifier: Machine learning classifier
        """
        self.elliott_wave_analyzer = elliott_wave_analyzer
        self.ml_classifier = ml_classifier
        logger.info("Analyzers set for multi-timeframe analysis")
    
    def analyze_timeframe(self, timeframe: Timeframe, bars: int = 200) -> Optional[TimeframeAnalysis]:
        """
        Analyze Elliott Wave patterns for specific timeframe.
        
        Args:
            timeframe: Target timeframe
            bars: Number of bars to analyze
            
        Returns:
            Analysis result or None if insufficient data
        """
        # Get data for timeframe
        data = self.data_manager.get_data(timeframe, bars)
        if data is None or len(data) < 20:
            logger.warning(f"Insufficient data for {timeframe.value} analysis")
            return None
        
        try:
            # Basic trend analysis
            trend_direction = self._analyze_trend(data)
            
            # Support and resistance levels
            support_resistance = self._find_support_resistance(data)
            
            # Fibonacci levels
            fibonacci_levels = self._calculate_fibonacci_levels(data)
            
            # Volume analysis
            volume_analysis = self._analyze_volume(data)
            
            # Momentum indicators
            momentum_indicators = self._calculate_momentum_indicators(data)
            
            # Elliott Wave pattern recognition
            pattern_type = "unknown"
            confidence = 0.5
            wave_count = {}
            
            if self.elliott_wave_analyzer:
                try:
                    # Use Elliott Wave analyzer
                    ew_result = self.elliott_wave_analyzer.analyze_patterns(data)
                    if ew_result:
                        pattern_type = ew_result.get('pattern_type', 'unknown')
                        confidence = ew_result.get('confidence', 0.5)
                        wave_count = ew_result.get('wave_count', {})
                except Exception as e:
                    logger.warning(f"Elliott Wave analysis error for {timeframe.value}: {e}")
            
            # ML pattern recognition (if available)
            if self.ml_classifier and hasattr(self.ml_classifier, 'predict'):
                try:
                    ml_prediction = self.ml_classifier.predict(data)
                    if ml_prediction.confidence > confidence:
                        pattern_type = ml_prediction.pattern_type
                        confidence = ml_prediction.confidence
                except Exception as e:
                    logger.warning(f"ML prediction error for {timeframe.value}: {e}")
            
            # Data quality
            data_quality = self.data_manager.data_quality.get(timeframe, 0.5)
            
            analysis = TimeframeAnalysis(
                timeframe=timeframe,
                pattern_type=pattern_type,
                confidence=confidence,
                wave_count=wave_count,
                trend_direction=trend_direction,
                support_resistance=support_resistance,
                fibonacci_levels=fibonacci_levels,
                volume_analysis=volume_analysis,
                momentum_indicators=momentum_indicators,
                timestamp=datetime.now(),
                data_quality=data_quality
            )
            
            # Cache the analysis
            with self._lock:
                self.analysis_cache[timeframe] = analysis
                # Set cache expiry based on timeframe
                expiry_minutes = self._get_cache_expiry_minutes(timeframe)
                self.cache_expiry[timeframe] = datetime.now() + timedelta(minutes=expiry_minutes)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {timeframe.value}: {e}")
            return None
    
    def _analyze_trend(self, data: pd.DataFrame) -> str:
        """Analyze overall trend direction."""
        if len(data) < 20:
            return "sideways"
        
        # Simple trend analysis using moving averages
        short_ma = data['close'].rolling(window=10).mean().iloc[-1]
        long_ma = data['close'].rolling(window=20).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if current_price > short_ma > long_ma:
            return "bullish"
        elif current_price < short_ma < long_ma:
            return "bearish"
        else:
            return "sideways"
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Find key support and resistance levels."""
        if len(data) < 20:
            return {"support": 0.0, "resistance": 0.0}
        
        # Simple support/resistance using recent highs and lows
        recent_data = data.tail(50)  # Last 50 bars
        
        # Find local maxima and minima
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Resistance: highest high in recent period
        resistance = highs.max()
        
        # Support: lowest low in recent period
        support = lows.min()
        
        return {
            "support": support,
            "resistance": resistance,
            "current_support": lows.tail(10).min(),  # More recent support
            "current_resistance": highs.tail(10).max()  # More recent resistance
        }
    
    def _calculate_fibonacci_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        if len(data) < 20:
            return {}
        
        # Find swing high and low
        recent_data = data.tail(100)
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        
        diff = swing_high - swing_low
        
        fibonacci_levels = {
            "0.0": swing_high,
            "23.6": swing_high - (diff * 0.236),
            "38.2": swing_high - (diff * 0.382),
            "50.0": swing_high - (diff * 0.5),
            "61.8": swing_high - (diff * 0.618),
            "78.6": swing_high - (diff * 0.786),
            "100.0": swing_low
        }
        
        return fibonacci_levels
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume characteristics."""
        if 'volume' not in data.columns or len(data) < 10:
            return {"avg_volume": 0.0, "volume_trend": 0.0}
        
        volumes = data['volume']
        
        # Average volume
        avg_volume = volumes.mean()
        
        # Volume trend (recent vs historical)
        recent_volume = volumes.tail(10).mean()
        historical_volume = volumes.head(len(volumes) - 10).mean()
        
        volume_trend = (recent_volume - historical_volume) / historical_volume if historical_volume > 0 else 0
        
        return {
            "avg_volume": avg_volume,
            "recent_volume": recent_volume,
            "volume_trend": volume_trend,
            "current_volume": volumes.iloc[-1] if len(volumes) > 0 else 0
        }
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators."""
        if len(data) < 14:
            return {"rsi": 50.0, "macd": 0.0}
        
        closes = data['close']
        
        # RSI calculation
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50.0
        
        # Simple MACD
        ema_12 = closes.ewm(span=12).mean()
        ema_26 = closes.ewm(span=26).mean()
        macd = ema_12 - ema_26
        current_macd = macd.iloc[-1] if not macd.empty else 0.0
        
        return {
            "rsi": current_rsi,
            "macd": current_macd,
            "momentum": (closes.iloc[-1] - closes.iloc[-10]) / closes.iloc[-10] if len(closes) >= 10 else 0.0
        }
    
    def _get_cache_expiry_minutes(self, timeframe: Timeframe) -> int:
        """Get cache expiry time in minutes based on timeframe."""
        expiry_map = {
            Timeframe.M1: 1,
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.M30: 30,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
            Timeframe.D1: 1440,
            Timeframe.W1: 10080,
            Timeframe.MN1: 43200
        }
        return expiry_map.get(timeframe, 60)
    
    def analyze_multiple_timeframes(self, timeframes: List[Timeframe]) -> Dict[Timeframe, TimeframeAnalysis]:
        """
        Analyze multiple timeframes in parallel.
        
        Args:
            timeframes: List of timeframes to analyze
            
        Returns:
            Dictionary of analysis results by timeframe
        """
        results = {}
        
        # Check cache first
        valid_cached_results = {}
        timeframes_to_analyze = []
        
        current_time = datetime.now()
        with self._lock:
            for tf in timeframes:
                if (tf in self.analysis_cache and 
                    tf in self.cache_expiry and 
                    current_time < self.cache_expiry[tf]):
                    valid_cached_results[tf] = self.analysis_cache[tf]
                else:
                    timeframes_to_analyze.append(tf)
        
        # Use cached results
        results.update(valid_cached_results)
        
        # Analyze remaining timeframes in parallel
        if timeframes_to_analyze:
            future_to_timeframe = {
                self.executor.submit(self.analyze_timeframe, tf): tf 
                for tf in timeframes_to_analyze
            }
            
            for future in as_completed(future_to_timeframe):
                tf = future_to_timeframe[future]
                try:
                    analysis = future.result(timeout=30)  # 30 second timeout
                    if analysis:
                        results[tf] = analysis
                except Exception as e:
                    logger.error(f"Error analyzing {tf.value}: {e}")
        
        return results
    
    def generate_multi_timeframe_signal(self, timeframes: List[Timeframe] = None) -> Optional[MultiTimeframeSignal]:
        """
        Generate trading signal based on multiple timeframe analysis.
        
        Args:
            timeframes: List of timeframes to consider (None for default set)
            
        Returns:
            Multi-timeframe trading signal
        """
        if timeframes is None:
            # Default timeframes based on primary timeframe
            primary_level = self.timeframe_hierarchy[self.primary_timeframe]
            timeframes = [
                tf for tf, level in self.timeframe_hierarchy.items()
                if abs(level - primary_level) <= 2  # ±2 levels from primary
            ]
        
        # Analyze all timeframes
        analyses = self.analyze_multiple_timeframes(timeframes)
        
        if not analyses:
            logger.warning("No analysis results available for signal generation")
            return None
        
        # Analyze timeframe alignment
        timeframe_alignment = {}
        bullish_count = 0
        bearish_count = 0
        
        for tf, analysis in analyses.items():
            if analysis.trend_direction == "bullish":
                timeframe_alignment[tf] = "bullish"
                bullish_count += 1
            elif analysis.trend_direction == "bearish":
                timeframe_alignment[tf] = "bearish"
                bearish_count += 1
            else:
                timeframe_alignment[tf] = "sideways"
        
        # Determine overall signal
        total_timeframes = len(analyses)
        bullish_ratio = bullish_count / total_timeframes
        bearish_ratio = bearish_count / total_timeframes
        
        if bullish_ratio >= 0.6:
            signal_type = "buy"
            confidence = bullish_ratio
        elif bearish_ratio >= 0.6:
            signal_type = "sell"
            confidence = bearish_ratio
        else:
            signal_type = "hold"
            confidence = 1.0 - max(bullish_ratio, bearish_ratio)
        
        # Get primary timeframe analysis for entry details
        primary_analysis = analyses.get(self.primary_timeframe)
        if not primary_analysis:
            primary_analysis = list(analyses.values())[0]  # Use first available
        
        # Calculate entry price, stop loss, and take profit
        current_price = 0.0
        if self.data_manager.get_data(self.primary_timeframe):
            current_data = self.data_manager.get_data(self.primary_timeframe, 1)
            if current_data is not None and not current_data.empty:
                current_price = current_data['close'].iloc[-1]
        
        # Risk management based on volatility and timeframe alignment
        risk_level = "medium"
        if confidence >= 0.8 and len([tf for tf, align in timeframe_alignment.items() if align != "sideways"]) >= 3:
            risk_level = "low"
        elif confidence < 0.6 or bullish_ratio == bearish_ratio:
            risk_level = "high"
        
        # Calculate stop loss and take profit levels
        atr_multiplier = 2.0 if risk_level == "low" else 3.0 if risk_level == "medium" else 4.0
        
        stop_loss = current_price
        take_profit = [current_price]
        
        if signal_type == "buy":
            stop_loss = primary_analysis.support_resistance.get("support", current_price * 0.98)
            take_profit = [
                primary_analysis.support_resistance.get("resistance", current_price * 1.02),
                current_price * 1.03,  # Additional target
                current_price * 1.05   # Extended target
            ]
        elif signal_type == "sell":
            stop_loss = primary_analysis.support_resistance.get("resistance", current_price * 1.02)
            take_profit = [
                primary_analysis.support_resistance.get("support", current_price * 0.98),
                current_price * 0.97,  # Additional target
                current_price * 0.95   # Extended target
            ]
        
        # Generate reasoning
        reasoning = self._generate_signal_reasoning(analyses, timeframe_alignment, signal_type)
        
        return MultiTimeframeSignal(
            primary_timeframe=self.primary_timeframe,
            signal_type=signal_type,
            confidence=confidence,
            timeframe_alignment=timeframe_alignment,
            risk_level=risk_level,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def _generate_signal_reasoning(self, analyses: Dict[Timeframe, TimeframeAnalysis], 
                                 alignment: Dict[Timeframe, str], signal_type: str) -> str:
        """Generate human-readable reasoning for the signal."""
        reasoning_parts = []
        
        # Overall alignment
        aligned_timeframes = [tf.value for tf, align in alignment.items() if align == signal_type.replace("buy", "bullish").replace("sell", "bearish")]
        if aligned_timeframes:
            reasoning_parts.append(f"Timeframes {', '.join(aligned_timeframes)} show {signal_type.replace('buy', 'bullish').replace('sell', 'bearish')} alignment")
        
        # Pattern recognition
        primary_analysis = analyses.get(self.primary_timeframe)
        if primary_analysis and primary_analysis.pattern_type != "unknown":
            reasoning_parts.append(f"Primary timeframe shows {primary_analysis.pattern_type} pattern with {primary_analysis.confidence:.1%} confidence")
        
        # Technical indicators
        if primary_analysis:
            rsi = primary_analysis.momentum_indicators.get("rsi", 50)
            if signal_type == "buy" and rsi < 40:
                reasoning_parts.append("RSI indicates oversold conditions")
            elif signal_type == "sell" and rsi > 60:
                reasoning_parts.append("RSI indicates overbought conditions")
        
        return ". ".join(reasoning_parts) if reasoning_parts else f"Multi-timeframe analysis suggests {signal_type} signal"
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all cached analyses."""
        with self._lock:
            summary = {
                "symbol": self.symbol,
                "primary_timeframe": self.primary_timeframe.value,
                "last_update": datetime.now(),
                "cached_analyses": len(self.analysis_cache),
                "data_status": self.data_manager.get_data_status(),
                "timeframe_analyses": {}
            }
            
            for tf, analysis in self.analysis_cache.items():
                summary["timeframe_analyses"][tf.value] = {
                    "pattern_type": analysis.pattern_type,
                    "confidence": analysis.confidence,
                    "trend_direction": analysis.trend_direction,
                    "data_quality": analysis.data_quality,
                    "timestamp": analysis.timestamp
                }
            
            return summary
    
    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("Multi-timeframe analyzer cleanup completed")

def setup_multi_timeframe_analysis(symbol: str, primary_timeframe: Timeframe = Timeframe.H1) -> MultiTimeframeAnalyzer:
    """
    Setup multi-timeframe analysis system.
    
    Args:
        symbol: Trading symbol
        primary_timeframe: Primary timeframe for trading
        
    Returns:
        Configured MultiTimeframeAnalyzer
    """
    analyzer = MultiTimeframeAnalyzer(symbol, primary_timeframe)
    
    # Generate sample data for testing
    current_time = datetime.now()
    
    for timeframe in [Timeframe.M1, Timeframe.M5, Timeframe.H1, Timeframe.H4, Timeframe.D1]:
        # Generate sample OHLCV data
        if timeframe == Timeframe.M1:
            periods = 1440  # 24 hours
            freq = "1min"
        elif timeframe == Timeframe.M5:
            periods = 288   # 24 hours
            freq = "5min"
        elif timeframe == Timeframe.H1:
            periods = 168   # 1 week
            freq = "1H"
        elif timeframe == Timeframe.H4:
            periods = 168   # 4 weeks
            freq = "4H"
        else:  # D1
            periods = 30    # 1 month
            freq = "1D"
        
        dates = pd.date_range(end=current_time, periods=periods, freq=freq)
        
        # Generate realistic price data with trend
        base_price = 2000.0
        trend = np.linspace(0, 50, periods)  # Upward trend
        noise = np.random.normal(0, 10, periods)
        
        closes = base_price + trend + noise
        opens = closes + np.random.normal(0, 2, periods)
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 5, periods))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 5, periods))
        volumes = np.random.uniform(1000, 5000, periods)
        
        sample_data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates)
        
        analyzer.data_manager.update_data(timeframe, sample_data)
    
    logger.info(f"Multi-timeframe analysis setup completed for {symbol}")
    return analyzer

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Multi-Timeframe Analyzer...")
    
    # Setup analyzer
    analyzer = setup_multi_timeframe_analysis("XAUUSD", Timeframe.H1)
    
    try:
        # Test single timeframe analysis
        h1_analysis = analyzer.analyze_timeframe(Timeframe.H1)
        if h1_analysis:
            print(f"H1 Analysis: {h1_analysis.pattern_type} ({h1_analysis.confidence:.2f})")
            print(f"Trend: {h1_analysis.trend_direction}")
        
        # Test multiple timeframe analysis
        timeframes = [Timeframe.M5, Timeframe.H1, Timeframe.H4, Timeframe.D1]
        multi_analysis = analyzer.analyze_multiple_timeframes(timeframes)
        
        print(f"\nMulti-timeframe analysis results:")
        for tf, analysis in multi_analysis.items():
            print(f"{tf.value}: {analysis.trend_direction} - {analysis.pattern_type} ({analysis.confidence:.2f})")
        
        # Generate trading signal
        signal = analyzer.generate_multi_timeframe_signal()
        if signal:
            print(f"\nTrading Signal:")
            print(f"Type: {signal.signal_type}")
            print(f"Confidence: {signal.confidence:.2f}")
            print(f"Risk Level: {signal.risk_level}")
            print(f"Entry: {signal.entry_price:.2f}")
            print(f"Stop Loss: {signal.stop_loss:.2f}")
            print(f"Take Profit: {signal.take_profit}")
            print(f"Reasoning: {signal.reasoning}")
        
        # Get analysis summary
        summary = analyzer.get_analysis_summary()
        print(f"\nAnalysis Summary:")
        print(f"Cached analyses: {summary['cached_analyses']}")
        print(f"Data status: {len(summary['data_status'])} timeframes")
        
    finally:
        # Cleanup
        analyzer.cleanup()
        print("Multi-Timeframe Analyzer test completed")

