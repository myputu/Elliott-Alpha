#!/usr/bin/env python3
"""
Trading Environment - OpenAI Gym-like Environment for Self-Play RL
Phase 3 Self-Play AlphaGo - Trading Environment & Policy-Value Network

Author: Manus AI
Version: 3.0 - Self-Play RL Integration
"""

import gym
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import talib
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class TradingAction(Enum):
    """Trading actions available to the RL agent"""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3
    ADJUST_SL_UP = 4
    ADJUST_SL_DOWN = 5
    ADJUST_TP_UP = 6
    ADJUST_TP_DOWN = 7

class PositionType(Enum):
    """Position types"""
    NONE = 0
    LONG = 1
    SHORT = 2

@dataclass
class Position:
    """Trading position information"""
    position_type: PositionType = PositionType.NONE
    entry_price: float = 0.0
    volume: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    entry_time: int = 0
    unrealized_pnl: float = 0.0
    
    def is_open(self) -> bool:
        return self.position_type != PositionType.NONE
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current unrealized PnL"""
        if not self.is_open():
            return 0.0
        
        if self.position_type == PositionType.LONG:
            return (current_price - self.entry_price) * self.volume
        else:  # SHORT
            return (self.entry_price - current_price) * self.volume

@dataclass
class TradingState:
    """Complete trading state information"""
    # Market data
    ohlc_data: np.ndarray  # Multi-timeframe OHLC data
    technical_indicators: np.ndarray  # RSI, MACD, ATR, etc.
    wave_features: np.ndarray  # Elliott Wave structure features
    
    # Position and account info
    position: Position
    account_balance: float
    account_equity: float
    margin_used: float
    margin_free: float
    
    # Risk metrics
    current_drawdown: float
    max_drawdown: float
    consecutive_losses: int
    daily_pnl: float
    
    # Time and market condition
    current_time: int
    market_session: int  # 0=Asian, 1=London, 2=NY, 3=Overlap
    volatility_regime: int  # 0=Low, 1=Medium, 2=High
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for neural network input"""
        # Flatten all components
        ohlc_flat = self.ohlc_data.flatten()
        indicators_flat = self.technical_indicators.flatten()
        wave_flat = self.wave_features.flatten()
        
        # Position features
        position_features = np.array([
            float(self.position.position_type.value),
            self.position.entry_price,
            self.position.volume,
            self.position.stop_loss,
            self.position.take_profit,
            self.position.unrealized_pnl
        ])
        
        # Account features
        account_features = np.array([
            self.account_balance,
            self.account_equity,
            self.margin_used,
            self.margin_free,
            self.current_drawdown,
            self.max_drawdown,
            float(self.consecutive_losses),
            self.daily_pnl
        ])
        
        # Market condition features
        market_features = np.array([
            float(self.current_time),
            float(self.market_session),
            float(self.volatility_regime)
        ])
        
        # Concatenate all features
        return np.concatenate([
            ohlc_flat,
            indicators_flat,
            wave_flat,
            position_features,
            account_features,
            market_features
        ])

class TradingEnvironment(gym.Env):
    """
    Trading Environment for Self-Play Reinforcement Learning
    Implements OpenAI Gym interface for RL training
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 max_position_size: float = 0.1,  # 10% of balance
                 transaction_cost: float = 0.0001,  # 1 pip spread
                 max_drawdown_limit: float = 0.15,  # 15% max drawdown
                 lookback_window: int = 100,
                 timeframes: List[str] = ['M5', 'M15', 'H1', 'H4'],
                 reward_scaling: float = 1.0):
        
        super().__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.max_drawdown_limit = max_drawdown_limit
        self.lookback_window = lookback_window
        self.timeframes = timeframes
        self.reward_scaling = reward_scaling
        
        # Environment state
        self.current_step = 0
        self.max_steps = len(data) - lookback_window - 1
        self.done = False
        
        # Account state
        self.account_balance = initial_balance
        self.account_equity = initial_balance
        self.peak_equity = initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0
        
        # Position management
        self.position = Position()
        self.trade_history = []
        
        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(len(TradingAction))
        
        # Calculate observation space dimensions
        self._calculate_observation_space()
        
        # Precompute technical indicators
        self._precompute_indicators()
        
        # Elliott Wave analyzer (placeholder - will be integrated)
        self.wave_analyzer = None
        
        logger.info(f"Trading Environment initialized: {self.max_steps} steps, "
                   f"observation_space: {self.observation_space.shape}")
    
    def _calculate_observation_space(self):
        """Calculate the dimensions of observation space"""
        # OHLC data: 4 values * timeframes * lookback_window
        ohlc_dim = 4 * len(self.timeframes) * self.lookback_window
        
        # Technical indicators: RSI, MACD, ATR, etc. per timeframe
        indicators_per_tf = 10  # RSI, MACD_line, MACD_signal, MACD_hist, ATR, SMA, EMA, BB_upper, BB_lower, BB_middle
        indicators_dim = indicators_per_tf * len(self.timeframes) * self.lookback_window
        
        # Wave features: 20 features for Elliott Wave structure
        wave_dim = 20
        
        # Position features: 6 features
        position_dim = 6
        
        # Account features: 8 features
        account_dim = 8
        
        # Market condition features: 3 features
        market_dim = 3
        
        total_dim = ohlc_dim + indicators_dim + wave_dim + position_dim + account_dim + market_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )
    
    def _precompute_indicators(self):
        """Precompute technical indicators for all timeframes"""
        self.indicators = {}
        
        for tf in self.timeframes:
            # For simplicity, using the same data for all timeframes
            # In real implementation, you would resample data for each timeframe
            tf_data = self.data.copy()
            
            # Calculate technical indicators
            indicators = {}
            
            # RSI
            indicators['rsi'] = talib.RSI(tf_data['close'].values, timeperiod=14)
            
            # MACD
            macd_line, macd_signal, macd_hist = talib.MACD(tf_data['close'].values)
            indicators['macd_line'] = macd_line
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist
            
            # ATR
            indicators['atr'] = talib.ATR(tf_data['high'].values, tf_data['low'].values, tf_data['close'].values)
            
            # Moving Averages
            indicators['sma_20'] = talib.SMA(tf_data['close'].values, timeperiod=20)
            indicators['ema_20'] = talib.EMA(tf_data['close'].values, timeperiod=20)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(tf_data['close'].values)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            self.indicators[tf] = indicators
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.done = False
        
        # Reset account
        self.account_balance = self.initial_balance
        self.account_equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0
        
        # Reset position
        self.position = Position()
        self.trade_history = []
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if self.done:
            raise ValueError("Environment is done. Call reset() first.")
        
        # Convert action to enum
        trading_action = TradingAction(action)
        
        # Execute action
        reward = self._execute_action(trading_action)
        
        # Update account state
        self._update_account_state()
        
        # Check if episode is done
        self._check_done_conditions()
        
        # Move to next step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True
        
        # Get next observation
        observation = self._get_observation()
        
        # Prepare info dict
        info = self._get_info()
        
        return observation, reward, self.done, info
    
    def _execute_action(self, action: TradingAction) -> float:
        """Execute trading action and return immediate reward"""
        reward = 0.0
        current_price = self._get_current_price()
        
        if action == TradingAction.BUY and not self.position.is_open():
            reward = self._open_position(PositionType.LONG, current_price)
            
        elif action == TradingAction.SELL and not self.position.is_open():
            reward = self._open_position(PositionType.SHORT, current_price)
            
        elif action == TradingAction.CLOSE and self.position.is_open():
            reward = self._close_position(current_price)
            
        elif action == TradingAction.ADJUST_SL_UP and self.position.is_open():
            reward = self._adjust_stop_loss(current_price, direction=1)
            
        elif action == TradingAction.ADJUST_SL_DOWN and self.position.is_open():
            reward = self._adjust_stop_loss(current_price, direction=-1)
            
        elif action == TradingAction.ADJUST_TP_UP and self.position.is_open():
            reward = self._adjust_take_profit(current_price, direction=1)
            
        elif action == TradingAction.ADJUST_TP_DOWN and self.position.is_open():
            reward = self._adjust_take_profit(current_price, direction=-1)
            
        elif action == TradingAction.HOLD:
            reward = self._calculate_holding_reward()
        
        # Apply reward scaling
        return reward * self.reward_scaling
    
    def _open_position(self, position_type: PositionType, price: float) -> float:
        """Open a new trading position"""
        # Calculate position size based on account balance and risk
        atr = self._get_current_atr()
        risk_amount = self.account_balance * 0.02  # 2% risk per trade
        
        # Position size calculation
        if atr > 0:
            volume = min(risk_amount / (atr * 2), self.account_balance * self.max_position_size / price)
        else:
            volume = self.account_balance * self.max_position_size / price
        
        # Set stop loss and take profit
        if position_type == PositionType.LONG:
            stop_loss = price - (atr * 2)
            take_profit = price + (atr * 4)  # 1:2 risk-reward
        else:  # SHORT
            stop_loss = price + (atr * 2)
            take_profit = price - (atr * 4)
        
        # Apply transaction cost
        effective_price = price + (self.transaction_cost if position_type == PositionType.LONG else -self.transaction_cost)
        
        # Create position
        self.position = Position(
            position_type=position_type,
            entry_price=effective_price,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=self.current_step
        )
        
        self.total_trades += 1
        
        # Small negative reward for opening position (transaction cost)
        return -0.1
    
    def _close_position(self, price: float) -> float:
        """Close current position and calculate reward"""
        if not self.position.is_open():
            return 0.0
        
        # Apply transaction cost
        effective_price = price - (self.transaction_cost if self.position.position_type == PositionType.LONG else -self.transaction_cost)
        
        # Calculate PnL
        pnl = self.position.calculate_pnl(effective_price)
        
        # Update account balance
        self.account_balance += pnl
        self.daily_pnl += pnl
        
        # Calculate reward based on PnL and risk-reward ratio
        reward = self._calculate_trade_reward(pnl)
        
        # Update trade statistics
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
        
        # Record trade
        trade_record = {
            'entry_time': self.position.entry_time,
            'exit_time': self.current_step,
            'position_type': self.position.position_type.name,
            'entry_price': self.position.entry_price,
            'exit_price': effective_price,
            'volume': self.position.volume,
            'pnl': pnl,
            'duration': self.current_step - self.position.entry_time
        }
        self.trade_history.append(trade_record)
        
        # Reset position
        self.position = Position()
        
        return reward
    
    def _adjust_stop_loss(self, current_price: float, direction: int) -> float:
        """Adjust stop loss level"""
        if not self.position.is_open():
            return 0.0
        
        atr = self._get_current_atr()
        adjustment = atr * 0.5 * direction
        
        if self.position.position_type == PositionType.LONG:
            new_sl = self.position.stop_loss + adjustment
            # Don't allow SL above current price for long positions
            if new_sl < current_price:
                self.position.stop_loss = new_sl
                return 0.05  # Small positive reward for good risk management
        else:  # SHORT
            new_sl = self.position.stop_loss - adjustment
            # Don't allow SL below current price for short positions
            if new_sl > current_price:
                self.position.stop_loss = new_sl
                return 0.05
        
        return -0.02  # Small penalty for invalid adjustment
    
    def _adjust_take_profit(self, current_price: float, direction: int) -> float:
        """Adjust take profit level"""
        if not self.position.is_open():
            return 0.0
        
        atr = self._get_current_atr()
        adjustment = atr * 0.5 * direction
        
        if self.position.position_type == PositionType.LONG:
            new_tp = self.position.take_profit + adjustment
            # Don't allow TP below current price for long positions
            if new_tp > current_price:
                self.position.take_profit = new_tp
                return 0.02  # Small positive reward
        else:  # SHORT
            new_tp = self.position.take_profit - adjustment
            # Don't allow TP above current price for short positions
            if new_tp < current_price:
                self.position.take_profit = new_tp
                return 0.02
        
        return -0.01  # Small penalty for invalid adjustment
    
    def _calculate_holding_reward(self) -> float:
        """Calculate reward for holding current state"""
        reward = 0.0
        
        if self.position.is_open():
            current_price = self._get_current_price()
            
            # Check if stop loss or take profit is hit
            if self.position.position_type == PositionType.LONG:
                if current_price <= self.position.stop_loss:
                    return self._close_position(self.position.stop_loss)
                elif current_price >= self.position.take_profit:
                    return self._close_position(self.position.take_profit)
            else:  # SHORT
                if current_price >= self.position.stop_loss:
                    return self._close_position(self.position.stop_loss)
                elif current_price <= self.position.take_profit:
                    return self._close_position(self.position.take_profit)
            
            # Small reward/penalty based on unrealized PnL trend
            current_pnl = self.position.calculate_pnl(current_price)
            self.position.unrealized_pnl = current_pnl
            
            if current_pnl > 0:
                reward += 0.001  # Small positive reward for profitable position
            else:
                reward -= 0.001  # Small penalty for losing position
        
        return reward
    
    def _calculate_trade_reward(self, pnl: float) -> float:
        """Calculate reward based on trade PnL and risk management"""
        # Base reward from PnL
        if pnl > 0:
            # Positive reward for profitable trades
            risk_amount = self.account_balance * 0.02
            risk_reward_ratio = pnl / risk_amount if risk_amount > 0 else 0
            
            if risk_reward_ratio >= 2.0:
                reward = 2.0  # High reward for RR >= 1:2
            elif risk_reward_ratio >= 1.0:
                reward = 1.0  # Moderate reward for RR >= 1:1
            else:
                reward = 0.5  # Small reward for any profit
        else:
            # Negative reward for losing trades
            reward = -1.0
        
        # Bonus for consistency
        if self.total_trades >= 10:
            win_rate = self.winning_trades / self.total_trades
            if win_rate >= 0.6:
                reward += 0.2  # Bonus for high win rate
        
        # Penalty for consecutive losses
        if self.consecutive_losses >= 3:
            reward -= 0.5
        
        return reward
    
    def _update_account_state(self):
        """Update account equity and drawdown calculations"""
        current_price = self._get_current_price()
        
        # Calculate current equity
        unrealized_pnl = self.position.calculate_pnl(current_price) if self.position.is_open() else 0.0
        self.account_equity = self.account_balance + unrealized_pnl
        
        # Update peak equity and drawdown
        if self.account_equity > self.peak_equity:
            self.peak_equity = self.account_equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - self.account_equity) / self.peak_equity
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
    
    def _check_done_conditions(self):
        """Check if episode should end"""
        # End if max drawdown exceeded
        if self.current_drawdown >= self.max_drawdown_limit:
            self.done = True
            logger.warning(f"Episode ended due to max drawdown: {self.current_drawdown:.1%}")
        
        # End if account balance too low
        if self.account_balance <= self.initial_balance * 0.5:
            self.done = True
            logger.warning(f"Episode ended due to low balance: {self.account_balance}")
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        # Get OHLC data for lookback window
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step + 1
        
        ohlc_data = []
        indicators_data = []
        
        for tf in self.timeframes:
            # OHLC data
            tf_ohlc = self.data[['open', 'high', 'low', 'close']].iloc[start_idx:end_idx].values
            if len(tf_ohlc) < self.lookback_window:
                # Pad with first available data
                padding = np.tile(tf_ohlc[0], (self.lookback_window - len(tf_ohlc), 1))
                tf_ohlc = np.vstack([padding, tf_ohlc])
            
            ohlc_data.append(tf_ohlc)
            
            # Technical indicators
            tf_indicators = []
            for indicator_name in ['rsi', 'macd_line', 'macd_signal', 'macd_hist', 'atr', 
                                 'sma_20', 'ema_20', 'bb_upper', 'bb_middle', 'bb_lower']:
                indicator_values = self.indicators[tf][indicator_name][start_idx:end_idx]
                if len(indicator_values) < self.lookback_window:
                    # Pad with first available value
                    first_valid = indicator_values[~np.isnan(indicator_values)][0] if len(indicator_values[~np.isnan(indicator_values)]) > 0 else 0
                    padding = np.full(self.lookback_window - len(indicator_values), first_valid)
                    indicator_values = np.concatenate([padding, indicator_values])
                
                # Replace NaN with 0
                indicator_values = np.nan_to_num(indicator_values)
                tf_indicators.append(indicator_values)
            
            indicators_data.append(np.column_stack(tf_indicators))
        
        # Combine all timeframe data
        ohlc_combined = np.concatenate(ohlc_data, axis=1)
        indicators_combined = np.concatenate(indicators_data, axis=1)
        
        # Elliott Wave features (placeholder - will be integrated with actual analyzer)
        wave_features = self._get_wave_features()
        
        # Create trading state
        trading_state = TradingState(
            ohlc_data=ohlc_combined,
            technical_indicators=indicators_combined,
            wave_features=wave_features,
            position=self.position,
            account_balance=self.account_balance,
            account_equity=self.account_equity,
            margin_used=0.0,  # Simplified
            margin_free=self.account_balance,
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            consecutive_losses=self.consecutive_losses,
            daily_pnl=self.daily_pnl,
            current_time=self.current_step,
            market_session=self._get_market_session(),
            volatility_regime=self._get_volatility_regime()
        )
        
        return trading_state.to_array().astype(np.float32)
    
    def _get_wave_features(self) -> np.ndarray:
        """Get Elliott Wave features (placeholder)"""
        # This will be integrated with actual Elliott Wave analyzer
        # For now, return dummy features
        return np.zeros(20)
    
    def _get_current_price(self) -> float:
        """Get current market price"""
        return self.data['close'].iloc[self.current_step]
    
    def _get_current_atr(self) -> float:
        """Get current ATR value"""
        atr_values = self.indicators[self.timeframes[0]]['atr']
        return atr_values[self.current_step] if not np.isnan(atr_values[self.current_step]) else 0.001
    
    def _get_market_session(self) -> int:
        """Determine current market session"""
        # Simplified session detection based on step
        hour = (self.current_step % 24)
        if 0 <= hour < 8:
            return 0  # Asian
        elif 8 <= hour < 16:
            return 1  # London
        elif 16 <= hour < 24:
            return 2  # New York
        else:
            return 3  # Overlap
    
    def _get_volatility_regime(self) -> int:
        """Determine current volatility regime"""
        atr = self._get_current_atr()
        # Simple volatility classification
        if atr < 0.001:
            return 0  # Low
        elif atr < 0.002:
            return 1  # Medium
        else:
            return 2  # High
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state"""
        return {
            'account_balance': self.account_balance,
            'account_equity': self.account_equity,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'position_open': self.position.is_open(),
            'current_step': self.current_step,
            'done_reason': self._get_done_reason() if self.done else None
        }
    
    def _get_done_reason(self) -> str:
        """Get reason why episode ended"""
        if self.current_drawdown >= self.max_drawdown_limit:
            return "max_drawdown_exceeded"
        elif self.account_balance <= self.initial_balance * 0.5:
            return "low_balance"
        elif self.current_step >= self.max_steps:
            return "max_steps_reached"
        else:
            return "unknown"
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.account_balance:.2f}")
            print(f"Equity: ${self.account_equity:.2f}")
            print(f"Drawdown: {self.current_drawdown:.1%}")
            print(f"Position: {self.position.position_type.name if self.position.is_open() else 'None'}")
            if self.position.is_open():
                print(f"  Entry: {self.position.entry_price:.5f}")
                print(f"  Current PnL: ${self.position.unrealized_pnl:.2f}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Win Rate: {self.winning_trades / max(1, self.total_trades):.1%}")
            print("-" * 40)
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics"""
        if not self.trade_history:
            return {}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL statistics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else 0
        
        # Duration statistics
        avg_duration = trades_df['duration'].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_duration': avg_duration,
            'max_drawdown': self.max_drawdown,
            'final_balance': self.account_balance,
            'return_percent': (self.account_balance - self.initial_balance) / self.initial_balance * 100
        }

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create sample data
    np.random.seed(42)
    n_steps = 1000
    
    # Generate realistic OHLC data
    price = 1.1000
    data = []
    
    for i in range(n_steps):
        # Random walk with some trend
        change = np.random.normal(0, 0.0005)
        price += change
        
        # Generate OHLC
        high = price + abs(np.random.normal(0, 0.0002))
        low = price - abs(np.random.normal(0, 0.0002))
        open_price = price + np.random.normal(0, 0.0001)
        close_price = price
        
        data.append({
            'timestamp': datetime.now() + timedelta(minutes=i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': np.random.randint(100, 1000)
        })
    
    df = pd.DataFrame(data)
    
    # Create environment
    env = TradingEnvironment(
        data=df,
        initial_balance=10000.0,
        lookback_window=50
    )
    
    # Test environment
    print("Testing Trading Environment...")
    
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Run a few random steps
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"Step {step}: Action={TradingAction(action).name}, Reward={reward:.3f}, Done={done}")
            print(f"  Balance: ${info['account_balance']:.2f}, Equity: ${info['account_equity']:.2f}")
            print(f"  Drawdown: {info['current_drawdown']:.1%}, Trades: {info['total_trades']}")
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    print(f"\nTotal reward: {total_reward:.3f}")
    
    # Print final statistics
    stats = env.get_trade_statistics()
    print("\nFinal Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

