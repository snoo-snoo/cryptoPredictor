from scipy import stats
import numpy as np
from datetime import datetime

class RiskAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        
    def calculate_risk_metrics(self, df, position_size=1.0, confidence_level=0.95):
        """Calculate comprehensive risk metrics"""
        try:
            returns = df['close'].pct_change().dropna()
            
            metrics = {
                'var': self.calculate_value_at_risk(returns, position_size, confidence_level),
                'expected_shortfall': self.calculate_expected_shortfall(returns, position_size, confidence_level),
                'kelly_criterion': self.calculate_kelly_criterion(returns),
                'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                'sortino_ratio': self.calculate_sortino_ratio(returns),
                'max_drawdown': self.calculate_max_drawdown(df['close']),
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return None

    def calculate_value_at_risk(self, returns, position_size=1.0, confidence_level=0.95):
        """Calculate Value at Risk"""
        try:
            var = np.percentile(returns, (1 - confidence_level) * 100) * position_size
            return {
                'var_amount': float(var),
                'var_percent': float(var * 100),
                'confidence_level': confidence_level
            }
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return None

    def calculate_expected_shortfall(self, returns, position_size=1.0, confidence_level=0.95):
        """Calculate Expected Shortfall (CVaR)"""
        try:
            var = np.percentile(returns, (1 - confidence_level) * 100)
            es = returns[returns <= var].mean() * position_size
            return {
                'es_amount': float(es),
                'es_percent': float(es * 100),
                'confidence_level': confidence_level
            }
        except Exception as e:
            self.logger.error(f"Error calculating Expected Shortfall: {e}")
            return None

    def calculate_kelly_criterion(self, returns):
        """Calculate Kelly Criterion for position sizing"""
        try:
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                return 0
            
            win_prob = len(wins) / len(returns)
            win_avg = wins.mean()
            loss_avg = abs(losses.mean())
            
            if loss_avg == 0:
                return 0
            
            kelly = win_prob - ((1 - win_prob) / (win_avg / loss_avg))
            return max(0, min(kelly, 0.5))  # Cap at 50%
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly Criterion: {e}")
            return 0

    def calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe Ratio"""
        try:
            return returns.mean() / returns.std()
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe Ratio: {e}")
            return None

    def calculate_sortino_ratio(self, returns):
        """Calculate Sortino Ratio"""
        try:
            downside_deviation = returns[returns < 0].std()
            return returns.mean() / downside_deviation
        except Exception as e:
            self.logger.error(f"Error calculating Sortino Ratio: {e}")
            return None

    def calculate_max_drawdown(self, prices):
        """Calculate Maximum Drawdown"""
        try:
            cummax = np.maximum.accumulate(prices)
            return (prices - cummax).max() / cummax.max()
        except Exception as e:
            self.logger.error(f"Error calculating Maximum Drawdown: {e}")
            return None 