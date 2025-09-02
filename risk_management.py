"""
Advanced Risk Management Module for Elliott Wave Trading System
This module implements sophisticated risk management techniques including:
- Kelly Criterion for position sizing
- Value at Risk (VaR) calculations
- Maximum Drawdown monitoring
- Dynamic position sizing based on volatility
- Portfolio heat management
- Correlation-based risk assessment
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskManager:
    """
    Advanced risk management system for algorithmic trading.
    Implements multiple risk control mechanisms and position sizing strategies.
    """
    
    def __init__(self, initial_capital=10000, max_risk_per_trade=0.02, max_portfolio_risk=0.06):
        """
        Initialize the Risk Manager.
        
        Args:
            initial_capital (float): Initial trading capital
            max_risk_per_trade (float): Maximum risk per individual trade (as fraction of capital)
            max_portfolio_risk (float): Maximum total portfolio risk (as fraction of capital)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        
        # Risk tracking
        self.trade_history = []
        self.daily_returns = []
        self.drawdown_history = []
        self.var_history = []
        
        # Position tracking
        self.open_positions = {}
        self.position_correlations = {}
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.var_95 = 0.0
        self.var_99 = 0.0
        
    def kelly_criterion_position_size(self, win_probability, avg_win, avg_loss):
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            win_probability (float): Probability of winning trade (0-1)
            avg_win (float): Average winning trade amount
            avg_loss (float): Average losing trade amount (positive value)
            
        Returns:
            float: Optimal fraction of capital to risk (0-1)
        """
        if avg_loss <= 0 or win_probability <= 0 or win_probability >= 1:
            return 0.0
        
        # Kelly formula: f* = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_probability, q = 1-p
        b = avg_win / avg_loss
        p = win_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety margin (use 25% of Kelly for conservative approach)
        kelly_fraction = max(0, min(kelly_fraction * 0.25, self.max_risk_per_trade))
        
        logger.info(f"Kelly Criterion: Win Rate={p:.3f}, Avg Win/Loss={b:.3f}, Kelly Fraction={kelly_fraction:.3f}")
        return kelly_fraction
    
    def calculate_position_size_volatility_adjusted(self, entry_price, stop_loss, volatility):
        """
        Calculate position size adjusted for volatility using ATR-based approach.
        
        Args:
            entry_price (float): Entry price for the trade
            stop_loss (float): Stop loss price
            volatility (float): Current market volatility (e.g., ATR)
            
        Returns:
            float: Position size in units
        """
        if volatility <= 0:
            volatility = abs(entry_price - stop_loss)  # Fallback to price difference
        
        # Base risk amount
        risk_amount = self.current_capital * self.max_risk_per_trade
        
        # Volatility adjustment factor
        # Higher volatility = smaller position size
        volatility_multiplier = 1.0 / (1.0 + volatility / entry_price)
        
        # Adjusted risk amount
        adjusted_risk_amount = risk_amount * volatility_multiplier
        
        # Calculate position size
        price_risk = abs(entry_price - stop_loss)
        if price_risk > 0:
            position_size = adjusted_risk_amount / price_risk
        else:
            position_size = 0
        
        logger.info(f"Volatility-adjusted position size: {position_size:.4f} units")
        return position_size
    
    def calculate_var(self, returns, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) using historical simulation method.
        
        Args:
            returns (list): List of historical returns
            confidence_level (float): Confidence level (0.95 for 95% VaR)
            
        Returns:
            float: VaR value (positive number representing potential loss)
        """
        if len(returns) < 30:  # Need sufficient data
            return 0.0
        
        returns_array = np.array(returns)
        var = np.percentile(returns_array, (1 - confidence_level) * 100)
        
        # Convert to positive loss value
        var = abs(var) if var < 0 else 0
        
        return var
    
    def calculate_expected_shortfall(self, returns, confidence_level=0.95):
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns (list): List of historical returns
            confidence_level (float): Confidence level
            
        Returns:
            float: Expected Shortfall value
        """
        if len(returns) < 30:
            return 0.0
        
        returns_array = np.array(returns)
        var_threshold = np.percentile(returns_array, (1 - confidence_level) * 100)
        
        # Expected shortfall is the mean of returns below VaR threshold
        tail_returns = returns_array[returns_array <= var_threshold]
        
        if len(tail_returns) > 0:
            expected_shortfall = abs(np.mean(tail_returns))
        else:
            expected_shortfall = 0.0
        
        return expected_shortfall
    
    def calculate_maximum_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve (list): List of equity values over time
            
        Returns:
            tuple: (max_drawdown, current_drawdown)
        """
        if len(equity_curve) < 2:
            return 0.0, 0.0
        
        equity_array = np.array(equity_curve)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_array)
        
        # Calculate drawdown
        drawdown = (equity_array - running_max) / running_max
        
        max_drawdown = abs(np.min(drawdown))
        current_drawdown = abs(drawdown[-1])
        
        return max_drawdown, current_drawdown
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calculate Sharpe ratio from returns.
        
        Args:
            returns (list): List of returns
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            float: Sharpe ratio
        """
        if len(returns) < 30:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Convert annual risk-free rate to period rate (assuming daily returns)
        period_risk_free_rate = risk_free_rate / 252
        
        excess_returns = returns_array - period_risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        return sharpe_ratio
    
    def check_correlation_risk(self, new_position, existing_positions):
        """
        Check correlation risk when adding a new position.
        
        Args:
            new_position (dict): New position details
            existing_positions (dict): Dictionary of existing positions
            
        Returns:
            float: Correlation risk score (0-1, higher is more risky)
        """
        if not existing_positions:
            return 0.0
        
        # Simplified correlation check based on asset class, sector, etc.
        # In a real implementation, this would use historical price correlations
        
        new_symbol = new_position.get('symbol', '')
        correlation_scores = []
        
        for pos_id, position in existing_positions.items():
            existing_symbol = position.get('symbol', '')
            
            # Simple correlation heuristic based on symbol similarity
            # This should be replaced with actual correlation calculation
            if new_symbol[:3] == existing_symbol[:3]:  # Same currency pair prefix
                correlation_scores.append(0.8)
            elif new_symbol == existing_symbol:
                correlation_scores.append(1.0)
            else:
                correlation_scores.append(0.1)
        
        # Return average correlation
        return np.mean(correlation_scores) if correlation_scores else 0.0
    
    def calculate_portfolio_heat(self, positions):
        """
        Calculate current portfolio heat (total risk exposure).
        
        Args:
            positions (dict): Dictionary of open positions
            
        Returns:
            float: Portfolio heat as fraction of capital
        """
        total_risk = 0.0
        
        for pos_id, position in positions.items():
            position_risk = position.get('risk_amount', 0.0)
            total_risk += position_risk
        
        portfolio_heat = total_risk / self.current_capital
        
        return portfolio_heat
    
    def should_take_trade(self, trade_setup):
        """
        Determine if a trade should be taken based on risk management rules.
        
        Args:
            trade_setup (dict): Trade setup details including entry, stop_loss, etc.
            
        Returns:
            tuple: (should_take, position_size, reason)
        """
        # Check if we have sufficient capital
        if self.current_capital <= self.initial_capital * 0.5:  # 50% drawdown limit
            return False, 0, "Capital below 50% of initial amount"
        
        # Check maximum drawdown
        if self.current_drawdown > 0.2:  # 20% drawdown limit
            return False, 0, "Current drawdown exceeds 20%"
        
        # Check portfolio heat
        current_heat = self.calculate_portfolio_heat(self.open_positions)
        if current_heat > self.max_portfolio_risk:
            return False, 0, f"Portfolio heat ({current_heat:.2%}) exceeds maximum ({self.max_portfolio_risk:.2%})"
        
        # Check correlation risk
        correlation_risk = self.check_correlation_risk(trade_setup, self.open_positions)
        if correlation_risk > 0.7:  # High correlation threshold
            return False, 0, f"High correlation risk ({correlation_risk:.2f}) with existing positions"
        
        # Calculate position size using Kelly Criterion if we have trade history
        if len(self.trade_history) >= 20:
            wins = [t for t in self.trade_history if t['pnl'] > 0]
            losses = [t for t in self.trade_history if t['pnl'] < 0]
            
            if wins and losses:
                win_rate = len(wins) / len(self.trade_history)
                avg_win = np.mean([t['pnl'] for t in wins])
                avg_loss = abs(np.mean([t['pnl'] for t in losses]))
                
                kelly_fraction = self.kelly_criterion_position_size(win_rate, avg_win, avg_loss)
                risk_amount = self.current_capital * kelly_fraction
            else:
                risk_amount = self.current_capital * self.max_risk_per_trade
        else:
            risk_amount = self.current_capital * self.max_risk_per_trade
        
        # Calculate position size
        entry_price = trade_setup.get('entry_price', 0)
        stop_loss = trade_setup.get('stop_loss', 0)
        
        if entry_price <= 0 or stop_loss <= 0:
            return False, 0, "Invalid entry price or stop loss"
        
        price_risk = abs(entry_price - stop_loss)
        if price_risk <= 0:
            return False, 0, "No price risk (entry equals stop loss)"
        
        position_size = risk_amount / price_risk
        
        # Apply volatility adjustment if available
        volatility = trade_setup.get('volatility', 0)
        if volatility > 0:
            position_size = self.calculate_position_size_volatility_adjusted(
                entry_price, stop_loss, volatility
            )
        
        return True, position_size, "Trade approved"
    
    def update_trade_result(self, trade_result):
        """
        Update risk metrics with new trade result.
        
        Args:
            trade_result (dict): Trade result with pnl, entry_time, exit_time, etc.
        """
        self.trade_history.append(trade_result)
        
        # Update capital
        pnl = trade_result.get('pnl', 0)
        self.current_capital += pnl
        
        # Calculate return
        if self.current_capital > 0:
            daily_return = pnl / (self.current_capital - pnl)
            self.daily_returns.append(daily_return)
        
        # Update drawdown
        equity_curve = [self.initial_capital]
        running_capital = self.initial_capital
        
        for trade in self.trade_history:
            running_capital += trade.get('pnl', 0)
            equity_curve.append(running_capital)
        
        self.max_drawdown, self.current_drawdown = self.calculate_maximum_drawdown(equity_curve)
        
        # Update VaR
        if len(self.daily_returns) >= 30:
            self.var_95 = self.calculate_var(self.daily_returns, 0.95)
            self.var_99 = self.calculate_var(self.daily_returns, 0.99)
        
        # Update Sharpe ratio
        if len(self.daily_returns) >= 30:
            self.sharpe_ratio = self.calculate_sharpe_ratio(self.daily_returns)
        
        logger.info(f"Risk metrics updated: Capital={self.current_capital:.2f}, "
                   f"Drawdown={self.current_drawdown:.2%}, VaR95={self.var_95:.4f}, "
                   f"Sharpe={self.sharpe_ratio:.2f}")
    
    def get_risk_report(self):
        """
        Generate comprehensive risk report.
        
        Returns:
            dict: Risk metrics and statistics
        """
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trade_history if t.get('pnl', 0) < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum([t.get('pnl', 0) for t in self.trade_history])
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Calculate profit factor
        gross_profit = sum([t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) > 0])
        gross_loss = abs(sum([t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expected shortfall
        expected_shortfall_95 = self.calculate_expected_shortfall(self.daily_returns, 0.95)
        expected_shortfall_99 = self.calculate_expected_shortfall(self.daily_returns, 0.99)
        
        report = {
            'capital_metrics': {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'total_pnl': total_pnl,
                'total_return': total_return,
                'max_drawdown': self.max_drawdown,
                'current_drawdown': self.current_drawdown
            },
            'trade_metrics': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            },
            'risk_metrics': {
                'var_95': self.var_95,
                'var_99': self.var_99,
                'expected_shortfall_95': expected_shortfall_95,
                'expected_shortfall_99': expected_shortfall_99,
                'sharpe_ratio': self.sharpe_ratio,
                'portfolio_heat': self.calculate_portfolio_heat(self.open_positions)
            }
        }
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Initialize Risk Manager
    risk_manager = RiskManager(initial_capital=10000, max_risk_per_trade=0.02)
    
    # Example trade setup
    trade_setup = {
        'symbol': 'XAUUSD',
        'entry_price': 1900.0,
        'stop_loss': 1890.0,
        'take_profit': 1920.0,
        'volatility': 5.0
    }
    
    # Check if trade should be taken
    should_take, position_size, reason = risk_manager.should_take_trade(trade_setup)
    print(f"Should take trade: {should_take}")
    print(f"Position size: {position_size:.4f}")
    print(f"Reason: {reason}")
    
    # Simulate some trade results
    for i in range(50):
        # Random trade results for demonstration
        pnl = np.random.normal(10, 50)  # Mean profit of 10 with std dev of 50
        
        trade_result = {
            'pnl': pnl,
            'entry_time': datetime.now() - timedelta(days=50-i),
            'exit_time': datetime.now() - timedelta(days=50-i-1),
            'symbol': 'XAUUSD'
        }
        
        risk_manager.update_trade_result(trade_result)
    
    # Generate risk report
    risk_report = risk_manager.get_risk_report()
    
    print("\n=== Risk Management Report ===")
    print(f"Current Capital: ${risk_report['capital_metrics']['current_capital']:.2f}")
    print(f"Total Return: {risk_report['capital_metrics']['total_return']:.2%}")
    print(f"Max Drawdown: {risk_report['capital_metrics']['max_drawdown']:.2%}")
    print(f"Win Rate: {risk_report['trade_metrics']['win_rate']:.2%}")
    print(f"Profit Factor: {risk_report['trade_metrics']['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {risk_report['risk_metrics']['sharpe_ratio']:.2f}")
    print(f"VaR 95%: {risk_report['risk_metrics']['var_95']:.4f}")
    print(f"VaR 99%: {risk_report['risk_metrics']['var_99']:.4f}")

