class RiskManager:
    def __init__(self, logger, max_risk_per_trade=0.02):
        self.logger = logger
        self.max_risk_per_trade = max_risk_per_trade
        
    def calculate_position_size(self, capital, entry_price, stop_loss):
        """Calculate safe position size"""
        try:
            risk_amount = capital * self.max_risk_per_trade
            price_risk = abs(entry_price - stop_loss)
            
            if price_risk == 0:
                return 0
                
            position_size = risk_amount / price_risk
            max_position = capital * 0.2  # Max 20% of capital
            
            return min(position_size, max_position)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0

    def calculate_stop_loss(self, df, entry_price, direction='long'):
        """Calculate stop loss level"""
        try:
            atr = self.calculate_atr(df)
            
            if direction == 'long':
                stop_loss = entry_price - (2 * atr)
            else:
                stop_loss = entry_price + (2 * atr)
                
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return None

    def calculate_take_profit(self, entry_price, stop_loss, risk_reward_ratio=2):
        """Calculate take profit level"""
        try:
            risk = abs(entry_price - stop_loss)
            take_profit = entry_price + (risk * risk_reward_ratio)
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            return None

    def check_risk_limits(self, portfolio, new_trade):
        """Check if new trade complies with risk limits"""
        try:
            # Check maximum portfolio risk
            total_risk = sum(trade['risk_amount'] for trade in portfolio)
            if total_risk + new_trade['risk_amount'] > self.max_portfolio_risk:
                return False
                
            # Check correlation risk
            if self.check_correlation_risk(portfolio, new_trade):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False 