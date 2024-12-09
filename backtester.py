import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Backtester:
    def __init__(self, logger):
        self.logger = logger
        self.results = {}
        
    def run_backtest(self, df, strategy_params, initial_capital=10000):
        """Run backtest with given parameters"""
        try:
            positions = []
            capital = initial_capital
            in_position = False
            entry_price = 0
            position_size = 0
            
            for i in range(1, len(df)):
                signals = self.generate_signals(df.iloc[:i], strategy_params)
                
                if not in_position and signals['final_signal'] in ['STRONG_BUY', 'BUY']:
                    # Calculate position size based on risk
                    risk_amount = capital * strategy_params['risk_per_trade']
                    stop_loss = df['low'].iloc[i] * (1 - strategy_params['stop_loss_pct'])
                    position_size = self.calculate_position_size(
                        capital, risk_amount, df['close'].iloc[i], stop_loss
                    )
                    
                    entry_price = df['close'].iloc[i]
                    in_position = True
                    positions.append({
                        'type': 'ENTRY',
                        'price': entry_price,
                        'size': position_size,
                        'timestamp': df.index[i],
                        'capital': capital
                    })
                
                elif in_position:
                    # Check stop loss
                    if df['low'].iloc[i] <= stop_loss:
                        capital = self.close_position(
                            capital, position_size, stop_loss, entry_price, positions, 
                            df.index[i], 'STOP_LOSS'
                        )
                        in_position = False
                    
                    # Check take profit or sell signals
                    elif (df['high'].iloc[i] >= entry_price * (1 + strategy_params['take_profit_pct']) or 
                          signals['final_signal'] in ['STRONG_SELL', 'SELL']):
                        exit_price = max(
                            df['close'].iloc[i],
                            entry_price * (1 + strategy_params['take_profit_pct'])
                        )
                        capital = self.close_position(
                            capital, position_size, exit_price, entry_price, positions, 
                            df.index[i], 'TAKE_PROFIT'
                        )
                        in_position = False
            
            return self.calculate_statistics(positions, df)
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            return None

    def calculate_position_size(self, capital, risk_amount, entry_price, stop_loss):
        """Calculate position size based on risk management rules"""
        price_risk = abs(entry_price - stop_loss)
        position_size = risk_amount / price_risk
        max_position = capital * 0.2  # Max 20% of capital per trade
        return min(position_size, max_position)

    def close_position(self, capital, size, exit_price, entry_price, positions, timestamp, reason):
        """Close a position and record the trade"""
        pnl = (exit_price - entry_price) * size
        new_capital = capital + pnl
        positions.append({
            'type': 'EXIT',
            'price': exit_price,
            'size': size,
            'timestamp': timestamp,
            'pnl': pnl,
            'capital': new_capital,
            'reason': reason
        })
        return new_capital

    def calculate_statistics(self, positions, df):
        """Calculate backtest statistics"""
        if not positions:
            return None
            
        trades = []
        current_trade = None
        
        for pos in positions:
            if pos['type'] == 'ENTRY':
                current_trade = pos
            else:
                trades.append({
                    'entry_time': current_trade['timestamp'],
                    'exit_time': pos['timestamp'],
                    'entry_price': current_trade['price'],
                    'exit_price': pos['price'],
                    'size': current_trade['size'],
                    'pnl': pos.get('pnl', 0),
                    'return': (pos['price'] - current_trade['price']) / current_trade['price'],
                    'reason': pos.get('reason', '')
                })
        
        returns = [t['return'] for t in trades]
        pnls = [t['pnl'] for t in trades]
        
        return {
            'total_trades': len(trades),
            'winning_trades': len([t for t in trades if t['pnl'] > 0]),
            'total_return': sum(returns),
            'total_pnl': sum(pnls),
            'max_drawdown': self.calculate_max_drawdown([p['capital'] for p in positions]),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'trades': trades
        }

    def calculate_max_drawdown(self, capitals):
        """Calculate maximum drawdown"""
        peak = capitals[0]
        max_dd = 0
        
        for capital in capitals:
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        if not returns:
            return 0
        excess_returns = np.array(returns) - risk_free_rate/365
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
 