import logging
from typing import Dict, Optional
import requests
import json
from datetime import datetime, timedelta
import os
import time

class NewsAnalyzer:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.cache_dir = 'news_cache'
        self.cache_duration = 3600  # 1 hour
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Initialize API keys from config
        try:
            import config
            self.cryptopanic_key = getattr(config, 'CRYPTOPANIC_API_KEY', None)
            self.newsapi_key = getattr(config, 'NEWSAPI_KEY', None)
            self.lunarcrush_key = getattr(config, 'LUNARCRUSH_API_KEY', None)
        except ImportError:
            self.logger.warning("Could not load API keys from config")
            self.cryptopanic_key = None
            self.newsapi_key = None
            self.lunarcrush_key = None

    def analyze(self, symbol: str) -> Optional[Dict]:
        """Analyze news sentiment for a symbol"""
        try:
            # Check cache first
            cache_file = os.path.join(self.cache_dir, f"{symbol.lower()}_news.json")
            current_time = datetime.now()
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        if (current_time - datetime.fromisoformat(cached_data['timestamp'])).total_seconds() < self.cache_duration:
                            return cached_data['data']
                except Exception as e:
                    self.logger.warning(f"Error reading cache for {symbol}: {e}")

            # Fetch news from different sources
            news_data = []
            
            # CryptoPanic
            if self.cryptopanic_key:
                cryptopanic_news = self.fetch_cryptopanic(symbol)
                if cryptopanic_news:
                    news_data.extend(cryptopanic_news)
            
            # NewsAPI
            if self.newsapi_key:
                newsapi_news = self.fetch_newsapi(symbol)
                if newsapi_news:
                    news_data.extend(newsapi_news)
            
            # LunarCrush
            if self.lunarcrush_key:
                lunarcrush_news = self.fetch_lunarcrush(symbol)
                if lunarcrush_news:
                    news_data.extend(lunarcrush_news)

            if not news_data:
                self.logger.warning(f"No news data found for {symbol}")
                return None

            # Analyze sentiment
            sentiment = self.calculate_sentiment(news_data)
            if sentiment is None:
                return None

            # Prepare data for caching
            cache_data = {
                'timestamp': current_time.isoformat(),
                'data': {
                    'sentiment_score': sentiment['score'],
                    'confidence': sentiment['confidence'],
                    'sources': sentiment['sources'],
                    'article_count': sentiment['article_count'],
                    'last_updated': current_time.isoformat()
                }
            }
            
            # Save to cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Error saving cache for {symbol}: {e}")
            
            return cache_data['data']

        except Exception as e:
            self.logger.error(f"Error analyzing news for {symbol}: {e}")
            return None

    def fetch_cryptopanic(self, symbol: str) -> list:
        """Fetch news from CryptoPanic"""
        try:
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_key}&currencies={symbol}"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            return [{
                'title': item['title'],
                'source': 'cryptopanic',
                'published_at': item['published_at'],
                'sentiment': item.get('sentiment', 'neutral'),
                'url': item['url']
            } for item in data['results']]
            
        except Exception as e:
            self.logger.error(f"Error fetching CryptoPanic news: {e}")
            return []

    def fetch_newsapi(self, symbol: str) -> list:
        """Fetch news from NewsAPI"""
        try:
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.newsapi_key}&language=en"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            return [{
                'title': item['title'],
                'source': 'newsapi',
                'published_at': item['publishedAt'],
                'url': item['url']
            } for item in data['articles']]
            
        except Exception as e:
            self.logger.error(f"Error fetching NewsAPI news: {e}")
            return []

    def fetch_lunarcrush(self, symbol: str) -> list:
        """Fetch news from LunarCrush"""
        try:
            url = f"https://lunarcrush.com/api3/news?symbol={symbol}&key={self.lunarcrush_key}"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            return [{
                'title': item['title'],
                'source': 'lunarcrush',
                'published_at': item['published_at'],
                'sentiment': item.get('sentiment', 0),
                'url': item['url']
            } for item in data['data']]
            
        except Exception as e:
            self.logger.error(f"Error fetching LunarCrush news: {e}")
            return []

    def calculate_sentiment(self, news_data: list) -> Dict:
        """Calculate overall sentiment from news data"""
        try:
            from textblob import TextBlob
            
            sentiments = []
            sources = set()
            
            for item in news_data:
                # Get sentiment from text
                blob = TextBlob(item['title'])
                sentiment = blob.sentiment.polarity
                
                # If source provides sentiment, combine with text sentiment
                if 'sentiment' in item:
                    if isinstance(item['sentiment'], str):
                        source_sentiment = {'positive': 1, 'negative': -1}.get(item['sentiment'], 0)
                    else:
                        source_sentiment = item['sentiment']
                    sentiment = (sentiment + source_sentiment) / 2
                
                sentiments.append(sentiment)
                sources.add(item['source'])
            
            if not sentiments:
                return None
            
            # Calculate metrics
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            return {
                'score': (avg_sentiment + 1) / 2,  # Normalize to 0-1
                'confidence': min(len(news_data) / 10, 1.0),  # More news = higher confidence
                'sources': list(sources),
                'article_count': len(news_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment: {e}")
            return None