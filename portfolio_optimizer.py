"""
Portfolio Optimization Module for Elliott Wave Trading System
This module implements advanced portfolio optimization techniques including:
- Modern Portfolio Theory (Markowitz optimization)
- Risk Parity allocation
- Black-Litterman model
- Monte Carlo simulation for portfolio optimization
- Multi-asset Elliott Wave portfolio construction
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import linalg
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Advanced portfolio optimization system for multi-asset Elliott Wave trading.
    Implements various optimization techniques for risk-adjusted returns.
    """
    
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize the Portfolio Optimizer.
        
        Args:
            risk_free_rate (float): Annual risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        self.assets = []
        self.returns_data = {}
        self.covariance_matrix = None
        self.expected_returns = None
        
    def add_asset(self, symbol, returns_data):
        """
        Add an asset to the portfolio optimization universe.
        
        Args:
            symbol (str): Asset symbol
            returns_data (list or pd.Series): Historical returns data
        """
        self.assets.append(symbol)
        self.returns_data[symbol] = pd.Series(returns_data) if not isinstance(returns_data, pd.Series) else returns_data
        logger.info(f"Added asset {symbol} with {len(returns_data)} return observations")
    
    def calculate_expected_returns(self, method='historical'):
        """
        Calculate expected returns for all assets.
        
        Args:
            method (str): Method for calculating expected returns ('historical', 'capm', 'exponential')
            
        Returns:
            pd.Series: Expected returns for each asset
        """
        expected_returns = {}
        
        for symbol in self.assets:
            returns = self.returns_data[symbol]
            
            if method == 'historical':
                expected_returns[symbol] = returns.mean()
            elif method == 'exponential':
                # Exponentially weighted moving average
                expected_returns[symbol] = returns.ewm(span=60).mean().iloc[-1]
            elif method == 'capm':
                # Simplified CAPM (would need market returns in practice)
                expected_returns[symbol] = returns.mean()
            else:
                expected_returns[symbol] = returns.mean()
        
        self.expected_returns = pd.Series(expected_returns)
        logger.info(f"Calculated expected returns using {method} method")
        return self.expected_returns
    
    def calculate_covariance_matrix(self, method='sample'):
        """
        Calculate covariance matrix for all assets.
        
        Args:
            method (str): Method for calculating covariance ('sample', 'shrinkage', 'exponential')
            
        Returns:
            pd.DataFrame: Covariance matrix
        """
        # Align all return series
        returns_df = pd.DataFrame(self.returns_data)
        returns_df = returns_df.dropna()
        
        if method == 'sample':
            self.covariance_matrix = returns_df.cov()
        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage estimator
            self.covariance_matrix = self._shrinkage_covariance(returns_df)
        elif method == 'exponential':
            # Exponentially weighted covariance
            self.covariance_matrix = returns_df.ewm(span=60).cov().iloc[-len(self.assets):, :]
        else:
            self.covariance_matrix = returns_df.cov()
        
        logger.info(f"Calculated covariance matrix using {method} method")
        return self.covariance_matrix
    
    def _shrinkage_covariance(self, returns_df):
        """
        Calculate shrinkage covariance matrix using Ledoit-Wolf estimator.
        
        Args:
            returns_df (pd.DataFrame): Returns data
            
        Returns:
            pd.DataFrame: Shrinkage covariance matrix
        """
        # Simple implementation of shrinkage estimator
        sample_cov = returns_df.cov()
        n_assets = len(self.assets)
        
        # Target matrix (identity scaled by average variance)
        avg_var = np.trace(sample_cov) / n_assets
        target = np.eye(n_assets) * avg_var
        target = pd.DataFrame(target, index=sample_cov.index, columns=sample_cov.columns)
        
        # Shrinkage intensity (simplified)
        shrinkage_intensity = 0.2
        
        shrinkage_cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * target
        
        return shrinkage_cov
    
    def markowitz_optimization(self, target_return=None, risk_aversion=1.0):
        """
        Perform Markowitz mean-variance optimization.
        
        Args:
            target_return (float): Target portfolio return (if None, maximize Sharpe ratio)
            risk_aversion (float): Risk aversion parameter for utility maximization
            
        Returns:
            dict: Optimal weights and portfolio metrics
        """
        if self.expected_returns is None:
            self.calculate_expected_returns()
        if self.covariance_matrix is None:
            self.calculate_covariance_matrix()
        
        n_assets = len(self.assets)
        
        # Objective function
        def objective(weights):
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
            
            if target_return is not None:
                # Minimize variance for target return
                return portfolio_variance
            else:
                # Maximize utility (return - risk_aversion * variance)
                return -(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.dot(x, self.expected_returns) - target_return
            })
        
        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.dot(optimal_weights, self.expected_returns)
            portfolio_variance = np.dot(optimal_weights, np.dot(self.covariance_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            weights_dict = {asset: weight for asset, weight in zip(self.assets, optimal_weights)}
            
            return {
                'weights': weights_dict,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True
            }
        else:
            logger.error(f"Markowitz optimization failed: {result.message}")
            return {'optimization_success': False}
    
    def risk_parity_optimization(self):
        """
        Perform risk parity optimization (equal risk contribution).
        
        Returns:
            dict: Risk parity weights and portfolio metrics
        """
        if self.covariance_matrix is None:
            self.calculate_covariance_matrix()
        
        n_assets = len(self.assets)
        
        def risk_budget_objective(weights):
            """
            Objective function for risk parity optimization.
            Minimizes the sum of squared differences between risk contributions.
            """
            portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
            marginal_contrib = np.dot(self.covariance_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_variance
            
            # Target equal risk contribution
            target_contrib = 1.0 / n_assets
            
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds (no short selling)
        bounds = [(0.01, 0.99) for _ in range(n_assets)]  # Minimum 1% allocation
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(risk_budget_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            
            # Calculate portfolio metrics
            if self.expected_returns is not None:
                portfolio_return = np.dot(optimal_weights, self.expected_returns)
            else:
                portfolio_return = 0.0
            
            portfolio_variance = np.dot(optimal_weights, np.dot(self.covariance_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            if portfolio_volatility > 0:
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            else:
                sharpe_ratio = 0.0
            
            weights_dict = {asset: weight for asset, weight in zip(self.assets, optimal_weights)}
            
            return {
                'weights': weights_dict,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True
            }
        else:
            logger.error(f"Risk parity optimization failed: {result.message}")
            return {'optimization_success': False}
    
    def black_litterman_optimization(self, views=None, view_confidences=None, tau=0.025):
        """
        Perform Black-Litterman optimization with investor views.
        
        Args:
            views (dict): Dictionary of asset views {asset: expected_return}
            view_confidences (dict): Dictionary of view confidences {asset: confidence}
            tau (float): Scaling factor for uncertainty of prior
            
        Returns:
            dict: Black-Litterman optimal weights and metrics
        """
        if self.expected_returns is None:
            self.calculate_expected_returns()
        if self.covariance_matrix is None:
            self.calculate_covariance_matrix()
        
        # Market capitalization weights (simplified - use equal weights as proxy)
        w_market = np.array([1/len(self.assets)] * len(self.assets))
        
        # Implied equilibrium returns
        risk_aversion = 3.0  # Typical value
        pi = risk_aversion * np.dot(self.covariance_matrix, w_market)
        
        if views is not None and view_confidences is not None:
            # Construct P matrix (picking matrix)
            P = np.zeros((len(views), len(self.assets)))
            Q = np.zeros(len(views))  # View returns
            Omega = np.zeros((len(views), len(views)))  # View uncertainty
            
            for i, (asset, view_return) in enumerate(views.items()):
                if asset in self.assets:
                    asset_idx = self.assets.index(asset)
                    P[i, asset_idx] = 1.0
                    Q[i] = view_return
                    
                    # View uncertainty (inverse of confidence)
                    confidence = view_confidences.get(asset, 0.5)
                    Omega[i, i] = tau * self.covariance_matrix.iloc[asset_idx, asset_idx] / confidence
            
            # Black-Litterman formula
            tau_sigma = tau * self.covariance_matrix
            
            # New expected returns
            M1 = linalg.inv(tau_sigma)
            M2 = np.dot(P.T, np.dot(linalg.inv(Omega), P))
            M3 = np.dot(linalg.inv(tau_sigma), pi)
            M4 = np.dot(P.T, np.dot(linalg.inv(Omega), Q))
            
            mu_bl = np.dot(linalg.inv(M1 + M2), M3 + M4)
            
            # New covariance matrix
            cov_bl = linalg.inv(M1 + M2)
            
        else:
            # No views - use equilibrium returns
            mu_bl = pi
            cov_bl = self.covariance_matrix
        
        # Optimize with Black-Litterman inputs
        n_assets = len(self.assets)
        
        def objective(weights):
            portfolio_return = np.dot(weights, mu_bl)
            portfolio_variance = np.dot(weights, np.dot(cov_bl, weights))
            return -(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.dot(optimal_weights, mu_bl)
            portfolio_variance = np.dot(optimal_weights, np.dot(cov_bl, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            weights_dict = {asset: weight for asset, weight in zip(self.assets, optimal_weights)}
            
            return {
                'weights': weights_dict,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True
            }
        else:
            logger.error(f"Black-Litterman optimization failed: {result.message}")
            return {'optimization_success': False}
    
    def monte_carlo_optimization(self, n_simulations=10000):
        """
        Perform Monte Carlo simulation to find optimal portfolio.
        
        Args:
            n_simulations (int): Number of random portfolio simulations
            
        Returns:
            dict: Results from Monte Carlo optimization
        """
        if self.expected_returns is None:
            self.calculate_expected_returns()
        if self.covariance_matrix is None:
            self.calculate_covariance_matrix()
        
        n_assets = len(self.assets)
        results = []
        
        for _ in range(n_simulations):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)  # Normalize to sum to 1
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            if portfolio_volatility > 0:
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            else:
                sharpe_ratio = 0
            
            results.append({
                'weights': weights.copy(),
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            })
        
        # Find portfolios with best metrics
        results_df = pd.DataFrame(results)
        
        # Best Sharpe ratio
        best_sharpe_idx = results_df['sharpe_ratio'].idxmax()
        best_sharpe_portfolio = results[best_sharpe_idx]
        
        # Minimum volatility
        min_vol_idx = results_df['volatility'].idxmin()
        min_vol_portfolio = results[min_vol_idx]
        
        # Maximum return
        max_return_idx = results_df['return'].idxmax()
        max_return_portfolio = results[max_return_idx]
        
        return {
            'best_sharpe': {
                'weights': {asset: weight for asset, weight in zip(self.assets, best_sharpe_portfolio['weights'])},
                'expected_return': best_sharpe_portfolio['return'],
                'volatility': best_sharpe_portfolio['volatility'],
                'sharpe_ratio': best_sharpe_portfolio['sharpe_ratio']
            },
            'min_volatility': {
                'weights': {asset: weight for asset, weight in zip(self.assets, min_vol_portfolio['weights'])},
                'expected_return': min_vol_portfolio['return'],
                'volatility': min_vol_portfolio['volatility'],
                'sharpe_ratio': min_vol_portfolio['sharpe_ratio']
            },
            'max_return': {
                'weights': {asset: weight for asset, weight in zip(self.assets, max_return_portfolio['weights'])},
                'expected_return': max_return_portfolio['return'],
                'volatility': max_return_portfolio['volatility'],
                'sharpe_ratio': max_return_portfolio['sharpe_ratio']
            },
            'all_results': results_df
        }
    
    def efficient_frontier(self, n_points=50):
        """
        Calculate the efficient frontier.
        
        Args:
            n_points (int): Number of points on the efficient frontier
            
        Returns:
            pd.DataFrame: Efficient frontier data
        """
        if self.expected_returns is None:
            self.calculate_expected_returns()
        if self.covariance_matrix is None:
            self.calculate_covariance_matrix()
        
        # Range of target returns
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            result = self.markowitz_optimization(target_return=target_return)
            
            if result.get('optimization_success', False):
                efficient_portfolios.append({
                    'target_return': target_return,
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'weights': result['weights']
                })
        
        return pd.DataFrame(efficient_portfolios)
    
    def get_portfolio_summary(self, optimization_results):
        """
        Generate a comprehensive portfolio summary.
        
        Args:
            optimization_results (dict): Results from any optimization method
            
        Returns:
            dict: Portfolio summary with detailed metrics
        """
        if not optimization_results.get('optimization_success', True):
            return {'error': 'Optimization failed'}
        
        weights = optimization_results['weights']
        
        # Asset allocation summary
        allocation_summary = {}
        for asset, weight in weights.items():
            allocation_summary[asset] = {
                'weight': weight,
                'weight_pct': weight * 100
            }
        
        # Risk metrics
        portfolio_return = optimization_results.get('expected_return', 0)
        portfolio_volatility = optimization_results.get('volatility', 0)
        sharpe_ratio = optimization_results.get('sharpe_ratio', 0)
        
        # Diversification ratio
        if self.covariance_matrix is not None:
            weights_array = np.array([weights[asset] for asset in self.assets])
            portfolio_variance = np.dot(weights_array, np.dot(self.covariance_matrix, weights_array))
            
            # Weighted average of individual volatilities
            individual_vols = np.sqrt(np.diag(self.covariance_matrix))
            weighted_avg_vol = np.dot(weights_array, individual_vols)
            
            diversification_ratio = weighted_avg_vol / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 1
        else:
            diversification_ratio = 1
        
        summary = {
            'allocation': allocation_summary,
            'portfolio_metrics': {
                'expected_return': portfolio_return,
                'expected_return_pct': portfolio_return * 100,
                'volatility': portfolio_volatility,
                'volatility_pct': portfolio_volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'diversification_ratio': diversification_ratio
            },
            'risk_metrics': {
                'var_95': portfolio_return - 1.645 * portfolio_volatility,  # Parametric VaR
                'var_99': portfolio_return - 2.326 * portfolio_volatility,
                'expected_shortfall_95': portfolio_return - 2.063 * portfolio_volatility  # Approximate
            }
        }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Initialize Portfolio Optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Generate sample return data for multiple assets
    np.random.seed(42)
    n_periods = 252  # One year of daily data
    
    # Asset 1: XAUUSD (Gold)
    gold_returns = np.random.normal(0.0008, 0.015, n_periods)  # 20% annual return, 23% volatility
    
    # Asset 2: EURUSD (Euro)
    euro_returns = np.random.normal(0.0002, 0.008, n_periods)  # 5% annual return, 12% volatility
    
    # Asset 3: GBPUSD (Pound)
    pound_returns = np.random.normal(0.0003, 0.010, n_periods)  # 8% annual return, 16% volatility
    
    # Add correlation between EUR and GBP
    correlation_factor = 0.6
    pound_returns = correlation_factor * euro_returns + np.sqrt(1 - correlation_factor**2) * pound_returns
    
    # Add assets to optimizer
    optimizer.add_asset('XAUUSD', gold_returns)
    optimizer.add_asset('EURUSD', euro_returns)
    optimizer.add_asset('GBPUSD', pound_returns)
    
    # Calculate expected returns and covariance matrix
    optimizer.calculate_expected_returns(method='historical')
    optimizer.calculate_covariance_matrix(method='sample')
    
    print("=== Portfolio Optimization Results ===")
    print(f"Expected Returns:")
    for asset in optimizer.assets:
        print(f"  {asset}: {optimizer.expected_returns[asset]*252:.2%} (annualized)")
    
    # Markowitz optimization
    print("\n1. Markowitz Optimization (Maximum Sharpe Ratio):")
    markowitz_result = optimizer.markowitz_optimization()
    if markowitz_result.get('optimization_success'):
        summary = optimizer.get_portfolio_summary(markowitz_result)
        print(f"  Expected Return: {summary['portfolio_metrics']['expected_return_pct']:.2f}%")
        print(f"  Volatility: {summary['portfolio_metrics']['volatility_pct']:.2f}%")
        print(f"  Sharpe Ratio: {summary['portfolio_metrics']['sharpe_ratio']:.3f}")
        print("  Allocation:")
        for asset, data in summary['allocation'].items():
            print(f"    {asset}: {data['weight_pct']:.1f}%")
    
    # Risk Parity optimization
    print("\n2. Risk Parity Optimization:")
    risk_parity_result = optimizer.risk_parity_optimization()
    if risk_parity_result.get('optimization_success'):
        summary = optimizer.get_portfolio_summary(risk_parity_result)
        print(f"  Expected Return: {summary['portfolio_metrics']['expected_return_pct']:.2f}%")
        print(f"  Volatility: {summary['portfolio_metrics']['volatility_pct']:.2f}%")
        print(f"  Sharpe Ratio: {summary['portfolio_metrics']['sharpe_ratio']:.3f}")
        print("  Allocation:")
        for asset, data in summary['allocation'].items():
            print(f"    {asset}: {data['weight_pct']:.1f}%")
    
    # Black-Litterman with views
    print("\n3. Black-Litterman Optimization (with views):")
    views = {'XAUUSD': 0.15/252, 'EURUSD': -0.05/252}  # Daily views
    view_confidences = {'XAUUSD': 0.8, 'EURUSD': 0.6}
    
    bl_result = optimizer.black_litterman_optimization(views=views, view_confidences=view_confidences)
    if bl_result.get('optimization_success'):
        summary = optimizer.get_portfolio_summary(bl_result)
        print(f"  Expected Return: {summary['portfolio_metrics']['expected_return_pct']:.2f}%")
        print(f"  Volatility: {summary['portfolio_metrics']['volatility_pct']:.2f}%")
        print(f"  Sharpe Ratio: {summary['portfolio_metrics']['sharpe_ratio']:.3f}")
        print("  Allocation:")
        for asset, data in summary['allocation'].items():
            print(f"    {asset}: {data['weight_pct']:.1f}%")
    
    # Monte Carlo optimization
    print("\n4. Monte Carlo Optimization:")
    mc_result = optimizer.monte_carlo_optimization(n_simulations=5000)
    
    best_sharpe = mc_result['best_sharpe']
    print(f"  Best Sharpe Portfolio:")
    print(f"    Expected Return: {best_sharpe['expected_return']*252:.2f}%")
    print(f"    Volatility: {best_sharpe['volatility']*np.sqrt(252):.2f}%")
    print(f"    Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f}")
    print("    Allocation:")
    for asset, weight in best_sharpe['weights'].items():
        print(f"      {asset}: {weight*100:.1f}%")

