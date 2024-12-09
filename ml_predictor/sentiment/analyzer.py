from typing import Dict, Optional
from .news import NewsAnalyzer
from .social import SocialMonitor
import logging

class SentimentAnalyzer:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.news_analyzer = NewsAnalyzer(logger=self.logger)
        self.social_monitor = SocialMonitor(logger=self.logger)
        
        # Configure weights
        self.weights = {
            'news': 0.6,
            'social': 0.4
        }

    def analyze(self, symbol: str) -> Optional[Dict]:
        """Analyze overall sentiment"""
        try:
            # Get sentiment from different sources
            news_sentiment = self.news_analyzer.analyze(symbol)
            social_sentiment = self.social_monitor.get_twitter_sentiment(symbol)
            
            if not news_sentiment and not social_sentiment:
                return None
            
            # Calculate weighted sentiment
            total_weight = 0
            weighted_score = 0
            weighted_confidence = 0
            
            if news_sentiment:
                weight = self.weights['news']
                weighted_score += news_sentiment['sentiment_score'] * weight
                weighted_confidence += news_sentiment['confidence'] * weight
                total_weight += weight
            
            if social_sentiment:
                weight = self.weights['social']
                weighted_score += social_sentiment['score'] * weight
                weighted_confidence += social_sentiment['confidence'] * weight
                total_weight += weight
            
            if total_weight == 0:
                return None
                
            # Normalize by total weight
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
            
            return {
                'score': float(final_score),
                'confidence': float(final_confidence),
                'sources': {
                    'news': bool(news_sentiment),
                    'social': bool(social_sentiment)
                },
                'components': {
                    'news': news_sentiment,
                    'social': social_sentiment
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return None 