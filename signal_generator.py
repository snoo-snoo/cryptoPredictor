import numpy as np
import pandas as pd
from datetime import datetime

class SignalGenerator:
    def __init__(self, logger):
        self.logger = logger
        self.signal_weights = {
            'technical': 0.4,
            'ml': 0.3,
            'social': 0.2,
            'volume': 0.1
        }

    def generate_signals(self, df, ml_predictions, social_score, volume_analysis):
        """Generate comprehensive trading signals"""
        try:
            signals = {
                'technical': self.analyze_technical_signals(df),
                'ml': self.analyze_ml_signals(ml_predictions),
                'social': self.analyze_social_signals(social_score),
                'volume': self.analyze_volume_signals(df, volume_analysis)
            }
            
            # Calculate weighted signal
            weighted_signal = 0
            for signal_type, weight in self.signal_weights.items():
                weighted_signal += signals[signal_type]['score'] * weight
            
            # Generate final signal with confidence
            if weighted_signal > 0.7:
                final_signal = 'STRONG_BUY'
            elif weighted_signal > 0.5:
                final_signal = 'BUY'
            elif weighted_signal < -0.7:
                final_signal = 'STRONG_SELL'
            elif weighted_signal < -0.5:
                final_signal = 'SELL'
            else:
                final_signal = 'NEUTRAL'
            
            return {
                'final_signal': final_signal,
                'confidence': abs(weighted_signal),
                'components': signals,
                'weighted_score': weighted_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return None

    def analyze_technical_signals(self, df):
        """Analyze technical indicators"""
        try:
            signals = []
            
            # RSI signals
            if df['RSI'].iloc[-1] < 30:
                signals.append(1)  # Oversold - Buy signal
            elif df['RSI'].iloc[-1] > 70:
                signals.append(-1)  # Overbought - Sell signal
            
            # MACD signals
            if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] and \
               df['MACD'].iloc[-2] <= df['MACD_signal'].iloc[-2]:
                signals.append(1)  # Bullish crossover
            elif df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1] and \
                 df['MACD'].iloc[-2] >= df['MACD_signal'].iloc[-2]:
                signals.append(-1)  # Bearish crossover
            
            # Bollinger Bands signals
            if df['close'].iloc[-1] < df['BB_lower'].iloc[-1]:
                signals.append(1)  # Price below lower band - potential buy
            elif df['close'].iloc[-1] > df['BB_upper'].iloc[-1]:
                signals.append(-1)  # Price above upper band - potential sell
            
            # Moving Average signals
            if df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1] and \
               df['SMA20'].iloc[-2] <= df['SMA50'].iloc[-2]:
                signals.append(1)  # Golden cross
            elif df['SMA20'].iloc[-1] < df['SMA50'].iloc[-1] and \
                 df['SMA20'].iloc[-2] >= df['SMA50'].iloc[-2]:
                signals.append(-1)  # Death cross
            
            # Calculate average signal
            score = np.mean(signals) if signals else 0
            
            return {
                'score': score,
                'signals': signals,
                'strength': abs(score)
            }
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            return {'score': 0, 'signals': [], 'strength': 0}

    def analyze_ml_signals(self, predictions):
        """Analyze ML predictions"""
        try:
            signals = []
            
            # Analyze predictions for different timeframes
            for timeframe, data in predictions.items():
                avg_change = data['avg_change']
                if avg_change > 2:  # Strong bullish
                    signals.append(1)
                elif avg_change > 0.5:  # Bullish
                    signals.append(0.5)
                elif avg_change < -2:  # Strong bearish
                    signals.append(-1)
                elif avg_change < -0.5:  # Bearish
                    signals.append(-0.5)
            
            score = np.mean(signals) if signals else 0
            
            return {
                'score': score,
                'signals': signals,
                'strength': abs(score)
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML analysis: {e}")
            return {'score': 0, 'signals': [], 'strength': 0}

    def analyze_social_signals(self, social_score):
        """Analyze social sentiment"""
        try:
            score = social_score / 5  # Normalize to [-1, 1] range
            return {
                'score': score,
                'signals': [score],
                'strength': abs(score)
            }
        except Exception as e:
            self.logger.error(f"Error in social analysis: {e}")
            return {'score': 0, 'signals': [], 'strength': 0}

    def analyze_volume_signals(self, df, volume_analysis):
        """Analyze volume patterns"""
        try:
            signals = []
            
            # Volume increase
            vol_change = df['volume'].pct_change().iloc[-1]
            if vol_change > 0.5:  # 50% volume increase
                signals.append(1)
            elif vol_change < -0.5:  # 50% volume decrease
                signals.append(-1)
            
            # Volume trend
            if df['volume'].iloc[-1] > df['Volume_SMA'].iloc[-1]:
                signals.append(0.5)
            else:
                signals.append(-0.5)
            
            score = np.mean(signals) if signals else 0
            
            return {
                'score': score,
                'signals': signals,
                'strength': abs(score)
            }
            
        except Exception as e:
            self.logger.error(f"Error in volume analysis: {e}")
            return {'score': 0, 'signals': [], 'strength': 0} 