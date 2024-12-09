import tweepy
import praw
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import json
import os
import config
import time
import logging
from typing import Dict, Optional
import numpy as np

class SocialMonitor:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize Twitter client
        try:
            import tweepy
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

        # Initialize Reddit with proper error handling
        try:
            self.reddit = praw.Reddit(
                client_id=config.REDDIT_CLIENT_ID,
                client_secret=config.REDDIT_CLIENT_SECRET,
                user_agent=config.REDDIT_USER_AGENT
            )
            self.reddit.read_only = True
        except Exception as e:
            self.logger.error(f"Error initializing Reddit API: {e}")
            self.reddit = None
        
        # Create directory for sentiment data
        if not os.path.exists('sentiment_data'):
            os.makedirs('sentiment_data')

    def get_twitter_mentions(self, symbol, hours=24):
        """Get recent Twitter mentions for a symbol with caching"""
        try:
            current_time = time.time()
            
            # Check cache first
            if symbol in self.twitter_cache:
                cache_age = current_time - self.twitter_cache[symbol]['timestamp']
                if cache_age < self.twitter_cache_duration:
                    self.logger.info(f"Using cached Twitter data for {symbol}")
                    return self.twitter_cache[symbol]['data']
            
            # Check rate limit
            time_since_last_request = current_time - self.last_twitter_request
            if time_since_last_request < 900:  # 15 minutes in seconds
                self.logger.info(f"Skipping Twitter for {symbol} due to rate limit")
                return []
            
            # If we can make a request, proceed
            base_symbol = symbol.replace('USDT', '').replace('BTC', '')
            query = f"(#{base_symbol} OR {base_symbol} crypto) -is:retweet lang:en"
            
            try:
                tweets = self.twitter_client.search_recent_tweets(
                    query=query,
                    max_results=10,
                    start_time=datetime.utcnow() - timedelta(hours=hours),
                    tweet_fields=['created_at', 'public_metrics']
                )
            except tweepy.TooManyRequests:
                self.logger.warning(f"Rate limited for {symbol}")
                return []
            except Exception as e:
                self.logger.error(f"Twitter API error for {symbol}: {e}")
                return []
            
            analyzed_tweets = []
            if tweets and tweets.data:
                for tweet in tweets.data:
                    sentiment = self.analyze_tweet_sentiment(tweet.text)
                    analyzed_tweets.append({
                        'text': tweet.text,
                        'created_at': tweet.created_at.isoformat(),
                        'metrics': tweet.public_metrics,
                        'sentiment': sentiment
                    })
            
            # Update cache and last request time
            self.twitter_cache[symbol] = {
                'timestamp': current_time,
                'data': analyzed_tweets
            }
            self.last_twitter_request = current_time
            
            self.logger.info(f"Found {len(analyzed_tweets)} tweets for {symbol}")
            return analyzed_tweets
            
        except Exception as e:
            self.logger.error(f"Error getting Twitter mentions for {symbol}: {e}")
            return []

    def get_reddit_mentions(self, symbol, hours=24):
        """Get recent Reddit mentions for a symbol"""
        try:
            subreddits = ['CryptoCurrency', 'CryptoMarkets', 'binance']
            posts = []
            base_symbol = symbol.replace('USDT', '').replace('BTC', '')
            
            for subreddit in subreddits:
                try:
                    for post in self.reddit.subreddit(subreddit).search(
                        f"({base_symbol})", sort='new', time_filter='day', limit=10):
                        
                        if (datetime.utcnow() - datetime.fromtimestamp(post.created_utc)) \
                                <= timedelta(hours=hours):
                            
                            sentiment = self.analyze_tweet_sentiment(post.title + " " + post.selftext)
                            posts.append({
                                'title': post.title,
                                'text': post.selftext,
                                'created_utc': post.created_utc,
                                'score': post.score,
                                'num_comments': post.num_comments,
                                'sentiment': sentiment
                            })
                except Exception as e:
                    self.logger.warning(f"Error fetching from r/{subreddit}: {e}")
                    continue  # Move to next subreddit on error
            
            return posts
            
        except Exception as e:
            self.logger.error(f"Error getting Reddit mentions for {symbol}: {e}")
            return []

    def analyze_tweet_sentiment(self, text):
        """Analyze sentiment of a single text"""
        try:
            blob = TextBlob(text)
            vader_sentiment = self.sentiment_analyzer.polarity_scores(text)
            
            return {
                'textblob_sentiment': blob.sentiment.polarity,
                'vader_sentiment': vader_sentiment['compound']
            }
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return {
                'textblob_sentiment': 0,
                'vader_sentiment': 0
            }

    def calculate_social_score(self, twitter_data, reddit_data):
        """Calculate overall social sentiment score with emphasis on Reddit"""
        try:
            if not reddit_data and not twitter_data:
                return 0
            
            total_sentiment = 0
            count = 0
            
            # Reddit sentiment (weighted more heavily)
            for post in reddit_data:
                sentiment = post['sentiment']
                weight = 1 + (post['score'] + post['num_comments']) / 100
                total_sentiment += (sentiment['textblob_sentiment'] + 
                                  sentiment['vader_sentiment']) / 2 * weight * 2
                count += 2
            
            # Twitter sentiment (if available)
            for tweet in twitter_data:
                sentiment = tweet['sentiment']
                weight = 1 + (tweet['metrics']['like_count'] + 
                            tweet['metrics']['retweet_count']) / 1000
                total_sentiment += (sentiment['textblob_sentiment'] + 
                                  sentiment['vader_sentiment']) / 2 * weight
                count += 1
            
            return total_sentiment / count if count > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating social score: {e}")
            return 0

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