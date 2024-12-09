import tweepy
import logging
from typing import Dict, Optional
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
import config

class SocialMonitor:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize Twitter client
        try:
            auth = tweepy.OAuth1UserHandler(
                config.TWITTER_API_KEY,
                config.TWITTER_API_SECRET,
                config.TWITTER_ACCESS_TOKEN,
                config.TWITTER_ACCESS_SECRET
            )
            self.twitter_client = tweepy.API(auth)
            
            # Verify credentials
            self.twitter_client.verify_credentials()
            self.logger.info("Twitter authentication successful")
            
        except Exception as e:
            self.logger.error(f"Error initializing Twitter client: {e}")
            self.twitter_client = None

    def get_twitter_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get Twitter sentiment for a symbol"""
        try:
            if not self.twitter_client:
                return None
            
            # Search tweets
            query = f"#{symbol} OR ${symbol}"
            tweets = self.twitter_client.search_tweets(
                q=query,
                lang="en",
                count=100,
                tweet_mode="extended"
            )
            
            if not tweets:
                return None
            
            # Analyze sentiment
            sentiments = []
            for tweet in tweets:
                sentiment = self.analyze_text_sentiment(tweet.full_text)
                if sentiment:
                    sentiments.append(sentiment)
            
            if not sentiments:
                return None
            
            return {
                'score': float(np.mean([s['score'] for s in sentiments])),
                'confidence': float(np.mean([s['confidence'] for s in sentiments])),
                'count': len(sentiments)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Twitter sentiment: {e}")
            return None

    def analyze_text_sentiment(self, text: str) -> Optional[Dict]:
        """Analyze sentiment of a single text"""
        try:
            # Use TextBlob for sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Normalize score to 0-1 range
            score = (polarity + 1) / 2
            
            # Use subjectivity as confidence measure
            confidence = subjectivity
            
            return {
                'score': float(score),
                'confidence': float(confidence),
                'text': text
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing text sentiment: {e}")
            return None

    def get_sentiment_cache_key(self, symbol: str) -> str:
        """Generate cache key for sentiment data"""
        return f"sentiment_{symbol.lower()}_{datetime.now().strftime('%Y%m%d')}"

    def get_cached_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get cached sentiment data if available"""
        try:
            cache_key = self.get_sentiment_cache_key(symbol)
            # Implement caching logic here
            return None
        except Exception as e:
            self.logger.error(f"Error getting cached sentiment: {e}")
            return None

    def save_sentiment_cache(self, symbol: str, data: Dict) -> bool:
        """Save sentiment data to cache"""
        try:
            cache_key = self.get_sentiment_cache_key(symbol)
            # Implement caching logic here
            return True
        except Exception as e:
            self.logger.error(f"Error saving sentiment cache: {e}")
            return False 