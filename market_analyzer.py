import pandas as pd
import numpy as np
from scipy import stats

class MarketAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        
    def analyze_market_conditions(self, df, btc_df=None):
        """Comprehensive market analysis"""
        try:
            analysis = {
                'volatility': self.analyze_volatility(df),
                'trend': self.analyze_trend(df),
                'volume': self.analyze_volume(df),
                'liquidity': self.analyze_liquidity(df),
                'correlation': self.analyze_correlation(df, btc_df) if btc_df is not None else None,
                'market_structure': self.analyze_market_structure(df)
            }
            
            # Calculate overall market score
            scores = [v['score'] for v in analysis.values() if v and 'score' in v]
            analysis['overall_score'] = np.mean(scores) if scores else 0
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return None

    def analyze_volatility(self, df):
        """Analyze price volatility"""
        try:
            # Calculate different volatility metrics
            daily_returns = df['close'].pct_change()
            
            analysis = {
                'daily_volatility': daily_returns.std(),
                'annualized_volatility': daily_returns.std() * np.sqrt(365),
                'atr': self.calculate_atr(df),
                'volatility_trend': self.analyze_volatility_trend(daily_returns),
                'is_high_volatility': daily_returns.std() > daily_returns.std().rolling(30).mean().iloc[-1]
            }
            
            # Score volatility conditions (0 to 1)
            score = self.score_volatility(analysis)
            analysis['score'] = score
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in volatility analysis: {e}")
            return None

    def analyze_trend(self, df):
        """Analyze market trend"""
        try:
            # Calculate trend indicators
            analysis = {
                'adx': self.calculate_adx(df),
                'trend_strength': self.calculate_trend_strength(df),
                'price_momentum': self.calculate_momentum(df),
                'trend_direction': self.determine_trend_direction(df)
            }
            
            # Score trend conditions
            score = self.score_trend(analysis)
            analysis['score'] = score
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {e}")
            return None

    def analyze_volume(self, df):
        """Analyze volume patterns"""
        try:
            analysis = {
                'volume_trend': self.calculate_volume_trend(df),
                'volume_price_correlation': self.calculate_volume_price_correlation(df),
                'abnormal_volume': self.detect_abnormal_volume(df),
                'buying_pressure': self.calculate_buying_pressure(df)
            }
            
            # Score volume conditions
            score = self.score_volume(analysis)
            analysis['score'] = score
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in volume analysis: {e}")
            return None

    def analyze_liquidity(self, df):
        """Analyze market liquidity"""
        try:
            analysis = {
                'spread': self.calculate_spread(df),
                'depth': self.calculate_market_depth(df),
                'turnover_ratio': self.calculate_turnover_ratio(df),
                'liquidity_score': self.calculate_liquidity_score(df)
            }
            
            # Score liquidity conditions
            score = self.score_liquidity(analysis)
            analysis['score'] = score
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in liquidity analysis: {e}")
            return None

    def analyze_correlation(self, df, btc_df):
        """Analyze correlation with Bitcoin"""
        try:
            # Calculate various correlation metrics
            price_corr = df['close'].corr(btc_df['close'])
            returns_corr = df['close'].pct_change().corr(btc_df['close'].pct_change())
            
            analysis = {
                'price_correlation': price_corr,
                'returns_correlation': returns_corr,
                'correlation_stability': self.calculate_correlation_stability(df, btc_df)
            }
            
            # Score correlation conditions
            score = self.score_correlation(analysis)
            analysis['score'] = score
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")
            return None

    def analyze_market_structure(self, df):
        """Analyze market structure"""
        try:
            analysis = {
                'support_resistance': self.identify_support_resistance(df),
                'price_patterns': self.identify_price_patterns(df),
                'market_phases': self.identify_market_phases(df),
                'fibonacci_levels': self.calculate_fibonacci_levels(df)
            }
            
            # Score market structure
            score = self.score_market_structure(analysis)
            analysis['score'] = score
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in market structure analysis: {e}")
            return None 