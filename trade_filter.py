#!/usr/bin/env python3
"""
Trade Filter - Confluence Filter for Multi-Confirmation Trading Signals
Phase 2 Return Optimization - Only execute trades with high-quality confluences

Author: Manus AI
Version: 2.0 - Return Optimization
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import talib

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

class TrendDirection(Enum):
    """Trend direction"""
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0

@dataclass
class WaveSignal:
    """Elliott Wave signal structure"""
    wave_type: str  # "impulse", "corrective", "triangle", etc.
    wave_position: str  # "wave_1", "wave_2", "wave_3", "wave_4", "wave_5", "wave_a", "wave_b", "wave_c"
    direction: TrendDirection
    confidence: float  # 0.0 to 1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward: Optional[float] = None
    
    def is_valid(self) -> bool:
        """Check if wave signal is valid for trading"""
        # Only trade impulse waves (1, 3, 5) and corrective wave C
        valid_waves = ["wave_1", "wave_3", "wave_5", "wave_c"]
        valid_types = ["impulse", "corrective"]
        
        return (self.wave_position in valid_waves and 
                self.wave_type in valid_types and 
                self.confidence >= 0.6)

@dataclass
class MultiTimeframeSignal:
    """Multi-timeframe analysis signal"""
    timeframe: str
    rsi_signal: TrendDirection
    macd_signal: TrendDirection
    ma_signal: TrendDirection
    fibo_signal: TrendDirection
    overall_signal: TrendDirection
    strength: SignalStrength
    confluence_score: float  # 0.0 to 1.0
    
    def is_bullish(self) -> bool:
        return self.overall_signal == TrendDirection.BULLISH
    
    def is_bearish(self) -> bool:
        return self.overall_signal == TrendDirection.BEARISH

@dataclass
class MLPrediction:
    """Machine Learning prediction result"""
    probability: float  # 0.0 to 1.0
    direction: TrendDirection
    confidence_level: SignalStrength
    model_name: str
    features_used: List[str]
    
    def meets_threshold(self, threshold: float = 0.65) -> bool:
        """Check if ML prediction meets minimum threshold"""
        return self.probability >= threshold

@dataclass
class MarketCondition:
    """Current market condition assessment"""
    spread_normal: bool
    volatility_level: SignalStrength
    news_impact: bool  # True if high-impact news expected
    trading_session: str  # "asian", "london", "newyork", "overlap"
    liquidity_level: SignalStrength
    
    def is_favorable(self) -> bool:
        """Check if market conditions are favorable for trading"""
        return (self.spread_normal and 
                not self.news_impact and
                self.liquidity_level in [SignalStrength.MODERATE, SignalStrength.STRONG, SignalStrength.VERY_STRONG])

@dataclass
class ConfluenceResult:
    """Final confluence filter result"""
    entry_allowed: bool
    confluence_score: float  # 0.0 to 1.0
    direction: TrendDirection
    risk_reward_ratio: float
    position_size_multiplier: float  # 0.0 to 1.0
    reasons: List[str]
    warnings: List[str]
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        status = "APPROVED" if self.entry_allowed else "REJECTED"
        return f"{status}: Score={self.confluence_score:.2f}, RR={self.risk_reward_ratio:.2f}, Size={self.position_size_multiplier:.2f}"

class ConfluenceFilter:
    """
    Advanced confluence filter that validates trading signals through multiple confirmations
    Only allows high-quality entries with proper risk management
    """
    
    def __init__(self, 
                 min_confluence_score: float = 0.7,
                 min_ml_threshold: float = 0.65,
                 min_risk_reward: float = 2.0,
                 max_spread_multiplier: float = 3.0):
        
        self.min_confluence_score = min_confluence_score
        self.min_ml_threshold = min_ml_threshold
        self.min_risk_reward = min_risk_reward
        self.max_spread_multiplier = max_spread_multiplier
        
        # Statistics tracking
        self.total_signals = 0
        self.approved_signals = 0
        self.rejected_signals = 0
        self.rejection_reasons = {}
        
        logger.info(f"Confluence Filter initialized: min_score={min_confluence_score}, "
                   f"ml_threshold={min_ml_threshold}, min_rr={min_risk_reward}")
    
    def evaluate_confluence(self, 
                          wave_signal: WaveSignal,
                          mtf_signals: Dict[str, MultiTimeframeSignal],
                          ml_prediction: MLPrediction,
                          market_condition: MarketCondition,
                          current_price: float,
                          atr_value: float) -> ConfluenceResult:
        """
        Evaluate all confluence factors and determine if entry should be allowed
        
        Args:
            wave_signal: Elliott Wave analysis result
            mtf_signals: Multi-timeframe signals by timeframe
            ml_prediction: Machine learning prediction
            market_condition: Current market conditions
            current_price: Current market price
            atr_value: Current ATR value for volatility assessment
            
        Returns:
            ConfluenceResult with entry decision and details
        """
        self.total_signals += 1
        
        reasons = []
        warnings = []
        confluence_score = 0.0
        direction = TrendDirection.NEUTRAL
        risk_reward_ratio = 0.0
        position_size_multiplier = 0.0
        
        try:
            # 1. Elliott Wave Validation (30% weight)
            wave_score, wave_reasons, wave_warnings = self._evaluate_wave_signal(wave_signal)
            confluence_score += wave_score * 0.3
            reasons.extend(wave_reasons)
            warnings.extend(wave_warnings)
            
            if wave_signal.direction != TrendDirection.NEUTRAL:
                direction = wave_signal.direction
            
            # 2. Multi-Timeframe Confirmation (25% weight)
            mtf_score, mtf_reasons, mtf_warnings = self._evaluate_mtf_signals(mtf_signals, direction)
            confluence_score += mtf_score * 0.25
            reasons.extend(mtf_reasons)
            warnings.extend(mtf_warnings)
            
            # 3. ML Prediction Validation (25% weight)
            ml_score, ml_reasons, ml_warnings = self._evaluate_ml_prediction(ml_prediction, direction)
            confluence_score += ml_score * 0.25
            reasons.extend(ml_reasons)
            warnings.extend(ml_warnings)
            
            # 4. Market Condition Assessment (20% weight)
            market_score, market_reasons, market_warnings = self._evaluate_market_condition(market_condition, atr_value)
            confluence_score += market_score * 0.2
            reasons.extend(market_reasons)
            warnings.extend(market_warnings)
            
            # 5. Risk-Reward Calculation
            if wave_signal.price_target and wave_signal.stop_loss:
                risk_reward_ratio = self._calculate_risk_reward(
                    current_price, wave_signal.price_target, wave_signal.stop_loss, direction
                )
            
            # 6. Position Size Calculation
            position_size_multiplier = self._calculate_position_size_multiplier(
                confluence_score, ml_prediction.probability, atr_value, market_condition
            )
            
            # 7. Final Decision
            entry_allowed = self._make_final_decision(
                confluence_score, risk_reward_ratio, market_condition, reasons, warnings
            )
            
            if entry_allowed:
                self.approved_signals += 1
                logger.info(f"Signal APPROVED: {direction.name}, Score={confluence_score:.2f}, RR={risk_reward_ratio:.2f}")
            else:
                self.rejected_signals += 1
                rejection_reason = self._get_primary_rejection_reason(confluence_score, risk_reward_ratio, market_condition)
                self.rejection_reasons[rejection_reason] = self.rejection_reasons.get(rejection_reason, 0) + 1
                logger.debug(f"Signal REJECTED: {rejection_reason}")
            
            return ConfluenceResult(
                entry_allowed=entry_allowed,
                confluence_score=confluence_score,
                direction=direction,
                risk_reward_ratio=risk_reward_ratio,
                position_size_multiplier=position_size_multiplier,
                reasons=reasons,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error in confluence evaluation: {e}")
            return ConfluenceResult(
                entry_allowed=False,
                confluence_score=0.0,
                direction=TrendDirection.NEUTRAL,
                risk_reward_ratio=0.0,
                position_size_multiplier=0.0,
                reasons=[f"Error in evaluation: {str(e)}"],
                warnings=["System error occurred during confluence evaluation"]
            )
    
    def _evaluate_wave_signal(self, wave_signal: WaveSignal) -> Tuple[float, List[str], List[str]]:
        """Evaluate Elliott Wave signal quality"""
        score = 0.0
        reasons = []
        warnings = []
        
        if not wave_signal.is_valid():
            warnings.append("Elliott Wave signal is not valid for trading")
            return score, reasons, warnings
        
        # Base score from confidence
        score = wave_signal.confidence
        reasons.append(f"Elliott Wave {wave_signal.wave_position} detected with {wave_signal.confidence:.1%} confidence")
        
        # Bonus for high-probability waves
        if wave_signal.wave_position in ["wave_3", "wave_c"]:
            score += 0.1
            reasons.append(f"High-probability {wave_signal.wave_position} provides additional confidence")
        
        # Bonus for impulse waves
        if wave_signal.wave_type == "impulse":
            score += 0.05
            reasons.append("Impulse wave structure provides strong directional bias")
        
        # Warning for low confidence
        if wave_signal.confidence < 0.7:
            warnings.append(f"Elliott Wave confidence is moderate ({wave_signal.confidence:.1%})")
        
        return min(score, 1.0), reasons, warnings
    
    def _evaluate_mtf_signals(self, mtf_signals: Dict[str, MultiTimeframeSignal], 
                            primary_direction: TrendDirection) -> Tuple[float, List[str], List[str]]:
        """Evaluate multi-timeframe signal alignment"""
        score = 0.0
        reasons = []
        warnings = []
        
        if not mtf_signals:
            warnings.append("No multi-timeframe signals available")
            return score, reasons, warnings
        
        # Check alignment across timeframes
        aligned_signals = 0
        total_signals = len(mtf_signals)
        confluence_scores = []
        
        for tf, signal in mtf_signals.items():
            confluence_scores.append(signal.confluence_score)
            
            if signal.overall_signal == primary_direction:
                aligned_signals += 1
                reasons.append(f"{tf} timeframe confirms {primary_direction.name} direction")
            elif signal.overall_signal == TrendDirection.NEUTRAL:
                warnings.append(f"{tf} timeframe shows neutral signals")
            else:
                warnings.append(f"{tf} timeframe conflicts with primary direction")
        
        # Calculate alignment score
        alignment_ratio = aligned_signals / total_signals if total_signals > 0 else 0
        avg_confluence = sum(confluence_scores) / len(confluence_scores) if confluence_scores else 0
        
        score = (alignment_ratio * 0.6) + (avg_confluence * 0.4)
        
        if alignment_ratio >= 0.8:
            reasons.append(f"Strong multi-timeframe alignment ({alignment_ratio:.1%})")
        elif alignment_ratio >= 0.6:
            reasons.append(f"Moderate multi-timeframe alignment ({alignment_ratio:.1%})")
        else:
            warnings.append(f"Weak multi-timeframe alignment ({alignment_ratio:.1%})")
        
        return score, reasons, warnings
    
    def _evaluate_ml_prediction(self, ml_prediction: MLPrediction, 
                              primary_direction: TrendDirection) -> Tuple[float, List[str], List[str]]:
        """Evaluate machine learning prediction quality"""
        score = 0.0
        reasons = []
        warnings = []
        
        # Check if ML prediction meets threshold
        if not ml_prediction.meets_threshold(self.min_ml_threshold):
            warnings.append(f"ML prediction below threshold ({ml_prediction.probability:.1%} < {self.min_ml_threshold:.1%})")
            return score, reasons, warnings
        
        # Base score from probability
        score = ml_prediction.probability
        reasons.append(f"ML model predicts {ml_prediction.direction.name} with {ml_prediction.probability:.1%} probability")
        
        # Check direction alignment
        if ml_prediction.direction == primary_direction:
            score += 0.1
            reasons.append("ML prediction aligns with Elliott Wave direction")
        elif ml_prediction.direction != TrendDirection.NEUTRAL:
            score -= 0.2
            warnings.append("ML prediction conflicts with Elliott Wave direction")
        
        # Bonus for high confidence
        if ml_prediction.confidence_level in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
            score += 0.05
            reasons.append(f"ML prediction has {ml_prediction.confidence_level.name} confidence")
        
        return min(score, 1.0), reasons, warnings
    
    def _evaluate_market_condition(self, market_condition: MarketCondition, 
                                 atr_value: float) -> Tuple[float, List[str], List[str]]:
        """Evaluate current market conditions"""
        score = 0.0
        reasons = []
        warnings = []
        
        # Check if market conditions are favorable
        if not market_condition.is_favorable():
            warnings.append("Market conditions are not favorable for trading")
            return score, reasons, warnings
        
        # Base score for favorable conditions
        score = 0.7
        reasons.append("Market conditions are favorable for trading")
        
        # Spread condition
        if market_condition.spread_normal:
            score += 0.1
            reasons.append("Spread is within normal range")
        else:
            score -= 0.3
            warnings.append("Spread is wider than normal")
        
        # News impact
        if not market_condition.news_impact:
            score += 0.1
            reasons.append("No high-impact news expected")
        else:
            score -= 0.4
            warnings.append("High-impact news may affect trading")
        
        # Liquidity assessment
        if market_condition.liquidity_level == SignalStrength.VERY_STRONG:
            score += 0.1
            reasons.append("Very high liquidity conditions")
        elif market_condition.liquidity_level == SignalStrength.STRONG:
            score += 0.05
            reasons.append("High liquidity conditions")
        elif market_condition.liquidity_level == SignalStrength.WEAK:
            score -= 0.1
            warnings.append("Low liquidity conditions")
        
        return min(max(score, 0.0), 1.0), reasons, warnings
    
    def _calculate_risk_reward(self, current_price: float, target_price: float, 
                             stop_loss: float, direction: TrendDirection) -> float:
        """Calculate risk-reward ratio"""
        try:
            if direction == TrendDirection.BULLISH:
                risk = abs(current_price - stop_loss)
                reward = abs(target_price - current_price)
            else:  # BEARISH
                risk = abs(stop_loss - current_price)
                reward = abs(current_price - target_price)
            
            if risk > 0:
                return reward / risk
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating risk-reward: {e}")
            return 0.0
    
    def _calculate_position_size_multiplier(self, confluence_score: float, 
                                          ml_probability: float, atr_value: float,
                                          market_condition: MarketCondition) -> float:
        """Calculate position size multiplier based on signal quality"""
        try:
            # Base multiplier from confluence score
            base_multiplier = confluence_score
            
            # Adjust for ML probability
            ml_adjustment = (ml_probability - 0.5) * 0.4  # -0.2 to +0.2
            
            # Adjust for volatility (higher ATR = smaller position)
            volatility_adjustment = max(0.1, 1.0 - (atr_value * 0.1))
            
            # Adjust for market conditions
            market_adjustment = 1.0
            if not market_condition.spread_normal:
                market_adjustment *= 0.5
            if market_condition.news_impact:
                market_adjustment *= 0.3
            if market_condition.liquidity_level == SignalStrength.WEAK:
                market_adjustment *= 0.7
            
            # Calculate final multiplier
            multiplier = (base_multiplier + ml_adjustment) * volatility_adjustment * market_adjustment
            
            # Ensure multiplier is within reasonable bounds
            return max(0.1, min(1.0, multiplier))
            
        except Exception as e:
            logger.error(f"Error calculating position size multiplier: {e}")
            return 0.1  # Conservative fallback
    
    def _make_final_decision(self, confluence_score: float, risk_reward_ratio: float,
                           market_condition: MarketCondition, reasons: List[str], 
                           warnings: List[str]) -> bool:
        """Make final entry decision based on all factors"""
        
        # Check minimum confluence score
        if confluence_score < self.min_confluence_score:
            warnings.append(f"Confluence score too low ({confluence_score:.2f} < {self.min_confluence_score})")
            return False
        
        # Check minimum risk-reward ratio
        if risk_reward_ratio < self.min_risk_reward:
            warnings.append(f"Risk-reward ratio too low ({risk_reward_ratio:.2f} < {self.min_risk_reward})")
            return False
        
        # Check market conditions
        if not market_condition.is_favorable():
            warnings.append("Market conditions not favorable")
            return False
        
        # All checks passed
        return True
    
    def _get_primary_rejection_reason(self, confluence_score: float, 
                                    risk_reward_ratio: float,
                                    market_condition: MarketCondition) -> str:
        """Get the primary reason for signal rejection"""
        if confluence_score < self.min_confluence_score:
            return "Low confluence score"
        elif risk_reward_ratio < self.min_risk_reward:
            return "Poor risk-reward ratio"
        elif not market_condition.spread_normal:
            return "Wide spread"
        elif market_condition.news_impact:
            return "High-impact news"
        elif not market_condition.is_favorable():
            return "Unfavorable market conditions"
        else:
            return "Unknown reason"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get confluence filter statistics"""
        approval_rate = (self.approved_signals / self.total_signals * 100) if self.total_signals > 0 else 0
        
        return {
            'total_signals': self.total_signals,
            'approved_signals': self.approved_signals,
            'rejected_signals': self.rejected_signals,
            'approval_rate_percent': approval_rate,
            'rejection_reasons': self.rejection_reasons.copy(),
            'settings': {
                'min_confluence_score': self.min_confluence_score,
                'min_ml_threshold': self.min_ml_threshold,
                'min_risk_reward': self.min_risk_reward,
                'max_spread_multiplier': self.max_spread_multiplier
            }
        }
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.total_signals = 0
        self.approved_signals = 0
        self.rejected_signals = 0
        self.rejection_reasons.clear()
        logger.info("Confluence filter statistics reset")

# Helper functions for creating signals from existing analyzers

def create_wave_signal_from_analyzer(elliott_analyzer_result: Dict[str, Any]) -> WaveSignal:
    """Create WaveSignal from Elliott Wave analyzer result"""
    try:
        return WaveSignal(
            wave_type=elliott_analyzer_result.get('wave_type', 'unknown'),
            wave_position=elliott_analyzer_result.get('wave_position', 'unknown'),
            direction=TrendDirection.BULLISH if elliott_analyzer_result.get('direction', 0) > 0 else TrendDirection.BEARISH,
            confidence=elliott_analyzer_result.get('confidence', 0.0),
            price_target=elliott_analyzer_result.get('price_target'),
            stop_loss=elliott_analyzer_result.get('stop_loss'),
            risk_reward=elliott_analyzer_result.get('risk_reward')
        )
    except Exception as e:
        logger.error(f"Error creating wave signal: {e}")
        return WaveSignal(
            wave_type='unknown',
            wave_position='unknown',
            direction=TrendDirection.NEUTRAL,
            confidence=0.0
        )

def create_mtf_signal_from_analyzer(mtf_analyzer_result: Dict[str, Any], timeframe: str) -> MultiTimeframeSignal:
    """Create MultiTimeframeSignal from multi-timeframe analyzer result"""
    try:
        # Extract individual indicator signals
        rsi_signal = TrendDirection.BULLISH if mtf_analyzer_result.get('rsi_signal', 0) > 0 else TrendDirection.BEARISH
        macd_signal = TrendDirection.BULLISH if mtf_analyzer_result.get('macd_signal', 0) > 0 else TrendDirection.BEARISH
        ma_signal = TrendDirection.BULLISH if mtf_analyzer_result.get('ma_signal', 0) > 0 else TrendDirection.BEARISH
        fibo_signal = TrendDirection.BULLISH if mtf_analyzer_result.get('fibo_signal', 0) > 0 else TrendDirection.BEARISH
        
        # Calculate overall signal
        signals = [rsi_signal, macd_signal, ma_signal, fibo_signal]
        bullish_count = sum(1 for s in signals if s == TrendDirection.BULLISH)
        bearish_count = sum(1 for s in signals if s == TrendDirection.BEARISH)
        
        if bullish_count > bearish_count:
            overall_signal = TrendDirection.BULLISH
        elif bearish_count > bullish_count:
            overall_signal = TrendDirection.BEARISH
        else:
            overall_signal = TrendDirection.NEUTRAL
        
        # Calculate confluence score
        confluence_score = max(bullish_count, bearish_count) / len(signals)
        
        # Determine strength
        if confluence_score >= 0.75:
            strength = SignalStrength.VERY_STRONG
        elif confluence_score >= 0.6:
            strength = SignalStrength.STRONG
        elif confluence_score >= 0.5:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        return MultiTimeframeSignal(
            timeframe=timeframe,
            rsi_signal=rsi_signal,
            macd_signal=macd_signal,
            ma_signal=ma_signal,
            fibo_signal=fibo_signal,
            overall_signal=overall_signal,
            strength=strength,
            confluence_score=confluence_score
        )
        
    except Exception as e:
        logger.error(f"Error creating MTF signal: {e}")
        return MultiTimeframeSignal(
            timeframe=timeframe,
            rsi_signal=TrendDirection.NEUTRAL,
            macd_signal=TrendDirection.NEUTRAL,
            ma_signal=TrendDirection.NEUTRAL,
            fibo_signal=TrendDirection.NEUTRAL,
            overall_signal=TrendDirection.NEUTRAL,
            strength=SignalStrength.WEAK,
            confluence_score=0.0
        )

def create_ml_prediction_from_classifier(ml_result: Dict[str, Any]) -> MLPrediction:
    """Create MLPrediction from ML classifier result"""
    try:
        probability = ml_result.get('probability', 0.0)
        direction_value = ml_result.get('direction', 0)
        
        if direction_value > 0:
            direction = TrendDirection.BULLISH
        elif direction_value < 0:
            direction = TrendDirection.BEARISH
        else:
            direction = TrendDirection.NEUTRAL
        
        # Determine confidence level
        if probability >= 0.8:
            confidence_level = SignalStrength.VERY_STRONG
        elif probability >= 0.7:
            confidence_level = SignalStrength.STRONG
        elif probability >= 0.6:
            confidence_level = SignalStrength.MODERATE
        else:
            confidence_level = SignalStrength.WEAK
        
        return MLPrediction(
            probability=probability,
            direction=direction,
            confidence_level=confidence_level,
            model_name=ml_result.get('model_name', 'unknown'),
            features_used=ml_result.get('features_used', [])
        )
        
    except Exception as e:
        logger.error(f"Error creating ML prediction: {e}")
        return MLPrediction(
            probability=0.0,
            direction=TrendDirection.NEUTRAL,
            confidence_level=SignalStrength.WEAK,
            model_name='error',
            features_used=[]
        )

def create_market_condition(spread_ratio: float, volatility: float, 
                          news_expected: bool = False, session: str = "unknown") -> MarketCondition:
    """Create MarketCondition from current market data"""
    try:
        # Assess spread condition
        spread_normal = spread_ratio <= 3.0  # Normal if spread <= 3x normal
        
        # Assess volatility level
        if volatility >= 2.0:
            volatility_level = SignalStrength.VERY_STRONG
        elif volatility >= 1.5:
            volatility_level = SignalStrength.STRONG
        elif volatility >= 1.0:
            volatility_level = SignalStrength.MODERATE
        else:
            volatility_level = SignalStrength.WEAK
        
        # Assess liquidity based on session
        if session in ["london_newyork_overlap", "overlap"]:
            liquidity_level = SignalStrength.VERY_STRONG
        elif session in ["london", "newyork"]:
            liquidity_level = SignalStrength.STRONG
        elif session == "asian":
            liquidity_level = SignalStrength.MODERATE
        else:
            liquidity_level = SignalStrength.WEAK
        
        return MarketCondition(
            spread_normal=spread_normal,
            volatility_level=volatility_level,
            news_impact=news_expected,
            trading_session=session,
            liquidity_level=liquidity_level
        )
        
    except Exception as e:
        logger.error(f"Error creating market condition: {e}")
        return MarketCondition(
            spread_normal=False,
            volatility_level=SignalStrength.WEAK,
            news_impact=True,  # Conservative default
            trading_session="unknown",
            liquidity_level=SignalStrength.WEAK
        )

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create confluence filter
    confluence_filter = ConfluenceFilter(
        min_confluence_score=0.7,
        min_ml_threshold=0.65,
        min_risk_reward=2.0
    )
    
    # Example signals
    wave_signal = WaveSignal(
        wave_type="impulse",
        wave_position="wave_3",
        direction=TrendDirection.BULLISH,
        confidence=0.8,
        price_target=1.1000,
        stop_loss=1.0950
    )
    
    mtf_signals = {
        "H1": MultiTimeframeSignal(
            timeframe="H1",
            rsi_signal=TrendDirection.BULLISH,
            macd_signal=TrendDirection.BULLISH,
            ma_signal=TrendDirection.BULLISH,
            fibo_signal=TrendDirection.NEUTRAL,
            overall_signal=TrendDirection.BULLISH,
            strength=SignalStrength.STRONG,
            confluence_score=0.75
        )
    }
    
    ml_prediction = MLPrediction(
        probability=0.72,
        direction=TrendDirection.BULLISH,
        confidence_level=SignalStrength.STRONG,
        model_name="ensemble_classifier",
        features_used=["rsi", "macd", "wave_pattern"]
    )
    
    market_condition = MarketCondition(
        spread_normal=True,
        volatility_level=SignalStrength.MODERATE,
        news_impact=False,
        trading_session="london",
        liquidity_level=SignalStrength.STRONG
    )
    
    # Evaluate confluence
    result = confluence_filter.evaluate_confluence(
        wave_signal=wave_signal,
        mtf_signals=mtf_signals,
        ml_prediction=ml_prediction,
        market_condition=market_condition,
        current_price=1.0975,
        atr_value=0.0015
    )
    
    print(f"Confluence Result: {result.get_summary()}")
    print(f"Reasons: {result.reasons}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    
    # Print statistics
    stats = confluence_filter.get_statistics()
    print(f"Filter Statistics: {stats}")

