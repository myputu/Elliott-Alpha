import pandas as pd
import numpy as np
from trading_strategy import ElliottWaveTradingStrategy
from elliott_wave_analyzer import ElliottWaveAnalyzer
from database import OHLCDatabase

class Backtester:
    def __init__(self, data, initial_balance=10000, risk_per_trade=0.02, slippage=0.0001, spread=0.0002):
        self.data = data
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.trades = []
        self.equity_curve = []
        self.slippage = slippage
        self.spread = spread

    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate position size based on risk management."""
        risk_amount = self.current_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk > 0:
            position_size = risk_amount / price_risk
        else:
            position_size = 0
        return position_size

    def execute_trade(self, trade_setup, entry_bar_index):
        """Execute a trade based on the trade setup, simulating bar-by-bar execution."""
        entry_price = trade_setup["entry"]
        stop_loss = trade_setup["stop_loss"]
        target = trade_setup.get("target", trade_setup.get("target1"))
        trade_type = trade_setup["type"] # 'buy' or 'sell'

        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        pnl = 0
        result = "open"
        exit_price = entry_price # Default exit price if not closed
        exit_time = self.data.index[entry_bar_index]

        # Apply slippage and spread at entry
        if trade_type == 'buy':
            entry_price_adjusted = entry_price + self.slippage + self.spread / 2
        else: # sell
            entry_price_adjusted = entry_price - self.slippage - self.spread / 2

        # Iterate through subsequent bars to check for SL/TP hit
        for i in range(entry_bar_index, len(self.data)):
            current_bar = self.data.iloc[i]
            bar_high = current_bar["high"]
            bar_low = current_bar["low"]
            bar_close = current_bar["close"]
            current_time = self.data.index[i]

            if trade_type == 'buy': # Long trade
                # Check if target hit
                if bar_high >= target:
                    pnl = position_size * (target - entry_price_adjusted)
                    result = "win"
                    exit_price = target
                    exit_time = current_time
                    break
                # Check if stop loss hit
                elif bar_low <= stop_loss:
                    pnl = position_size * (stop_loss - entry_price_adjusted)
                    result = "loss"
                    exit_price = stop_loss
                    exit_time = current_time
                    break
            else: # Short trade
                # Check if target hit
                if bar_low <= target:
                    pnl = position_size * (entry_price_adjusted - target)
                    result = "win"
                    exit_price = target
                    exit_time = current_time
                    break
                # Check if stop loss hit
                elif bar_high >= stop_loss:
                    pnl = position_size * (entry_price_adjusted - stop_loss)
                    result = "loss"
                    exit_price = stop_loss
                    exit_time = current_time
                    break
            
            # If it's the last bar and trade is still open, close at current bar's close price
            if i == len(self.data) - 1:
                if trade_type == 'buy':
                    pnl = position_size * (bar_close - entry_price_adjusted)
                else:
                    pnl = position_size * (entry_price_adjusted - bar_close)
                result = "closed_at_end"
                exit_price = bar_close
                exit_time = current_time

        self.current_balance += pnl
        
        trade_record = {
            "entry_time": self.data.index[entry_bar_index],
            "exit_time": exit_time,
            "entry_price": entry_price,
            "entry_price_adjusted": entry_price_adjusted,
            "exit_price": exit_price,
            "position_size": position_size,
            "pnl": pnl,
            "result": result,
            "balance": self.current_balance,
            "trade_type": trade_type
        }
        
        self.trades.append(trade_record)
        self.equity_curve.append(self.current_balance)
        
        return trade_record

    def run_backtest(self, analyzer, strategy):
        """Run the backtest using the analyzer and strategy."""
        print("Starting backtest...")
        
        analyzer.run_analysis()
        
        signals = strategy.generate_signals(analyzer.waves)
        
        for signal in signals:
            entry_time = signal["timestamp"]
            try:
                # Get the index of the entry bar
                entry_bar_index = self.data.index.get_loc(entry_time)
            except KeyError:
                # If timestamp not found, skip this signal
                continue

            trade_setup = {
                "entry": signal["price"],
                "stop_loss": signal["stop_loss"],
                "target": signal["take_profit"],
                "type": signal["type"] # 'buy' or 'sell'
            }
            
            self.execute_trade(trade_setup, entry_bar_index)

        print(f"Backtest complete. Final balance: ${self.current_balance:.2f}")
        return self.get_performance_metrics()

    def get_performance_metrics(self):
        """Calculate performance metrics."""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["result"] == "win"])
        losing_trades = len(trades_df[trades_df["result"] == "loss"])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df["pnl"].sum()
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        
        avg_win = trades_df[trades_df["result"] == "win"]["pnl"].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df["result"] == "loss"]["pnl"].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trades_df[trades_df["pnl"] > 0]["pnl"].sum() / trades_df[trades_df["pnl"] < 0]["pnl"].sum()) if trades_df[trades_df["pnl"] < 0]["pnl"].sum() != 0 else float("inf")
        
        metrics = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_return": total_return,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "final_balance": self.current_balance
        }
        
        return metrics

if __name__ == "__main__":
    # Load data from database
    db = OHLCDatabase()
    backtest_df = db.get_ohlc_data("XAUUSD")
    
    # Initialize components
    analyzer = ElliottWaveAnalyzer(backtest_df)
    strategy = ElliottWaveTradingStrategy(analyzer)
    backtester = Backtester(backtest_df)
    
    # Run backtest
    performance = backtester.run_backtest(analyzer, strategy)
    
    print("Performance Metrics:")
    for key, value in performance.items():
        print(f"{key}: {value}")


