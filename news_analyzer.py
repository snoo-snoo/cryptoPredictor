import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import os
import config
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class NewsAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Create news cache directory
        if not os.path.exists('news_cache'):
            os.makedirs('news_cache')
        
        # Initialize proxy rotator
        self.proxy_rotator = ProxyRotator(logger)
        
        # Key news sources
        self.news_sources = {
            'cointelegraph': {
                'url': 'https://cointelegraph.com/tags/',
                'type': 'major',
                'selectors': {
                    'article': 'article',
                    'title': 'h2',
                    'text': '.post-content',
                    'date': '.post-date'
                }
            },
            'coindesk': {
                'url': 'https://www.coindesk.com/search?s=',
                'type': 'major',
                'selectors': {
                    'article': '.article-card',
                    'title': '.heading',
                    'text': '.content-text',
                    'date': '.date-time'
                }
            },
            'cryptoslate': {
                'url': 'https://cryptoslate.com/search/',
                'type': 'major',
                'selectors': {
                    'article': '.post-item',
                    'title': '.entry-title',
                    'text': '.entry-excerpt',
                    'date': '.post-date'
                }
            },
            'decrypt': {
                'url': 'https://decrypt.co/search?q=',
                'type': 'major',
                'selectors': {
                    'article': 'article.article-card',
                    'title': '.heading',
                    'text': '.description',
                    'date': 'time'
                }
            },
            'theblock': {
                'url': 'https://www.theblock.co/search?q=',
                'type': 'major',
                'selectors': {
                    'article': '.post-card',
                    'title': '.post-card__title',
                    'text': '.post-card__description',
                    'date': '.post-card__date'
                }
            },
            'bitcoinist': {
                'url': 'https://bitcoinist.com/?s=',
                'type': 'major',
                'selectors': {
                    'article': 'article.post',
                    'title': '.entry-title',
                    'text': '.entry-content',
                    'date': '.entry-date'
                }
            },
            'cryptodaily': {
                'url': 'https://cryptodaily.co.uk/search?q=',
                'type': 'technical',
                'selectors': {
                    'article': '.article-item',
                    'title': '.article-title',
                    'text': '.article-excerpt',
                    'date': '.article-date'
                }
            },
            'binance_announcements': {
                'url': 'https://www.binance.com/en/support/announcement/',
                'type': 'official',
                'selectors': {
                    'article': '.css-1wr4jig',
                    'title': '.css-1cf3zsg',
                    'text': '.css-6pr2fc',
                    'date': '.css-1g5n9fr'
                }
            }
        }
        
        # News categories and their weights
        self.news_weights = {
            'major': 0.25,      # Major crypto news outlets (reduced weight due to more sources)
            'official': 0.30,   # Official project announcements (increased importance)
            'technical': 0.25,  # Technical updates/GitHub and analysis
            'regulatory': 0.20  # Regulatory news
        }
        
        # Cache settings
        self.cache_duration = 3600  # 1 hour in seconds
        
        # Add blockchain explorer APIs with error handling
        try:
            self.chain_sources = {
                'ethereum': {
                    'url': 'https://api.etherscan.io/api',
                    'api_key': config.ETHERSCAN_API_KEY if hasattr(config, 'ETHERSCAN_API_KEY') else None
                },
                'binance': {
                    'url': 'https://api.bscscan.com/api',
                    'api_key': config.BSCSCAN_API_KEY if hasattr(config, 'BSCSCAN_API_KEY') else None
                }
            }
        except Exception as e:
            self.logger.error(f"Error initializing blockchain APIs: {e}")
            self.chain_sources = {}
        
        # Add fallback news sources with error handling
        self.fallback_sources = {}
        try:
            if hasattr(config, 'CRYPTOPANIC_API_KEY'):
                self.fallback_sources['cryptopanic'] = {
                    'url': 'https://cryptopanic.com/api/v1/posts/',
                    'api_key': config.CRYPTOPANIC_API_KEY
                }
            if hasattr(config, 'LUNARCRUSH_API_KEY'):
                self.fallback_sources['lunarcrush'] = {
                    'url': 'https://api.lunarcrush.com/v2/assets/news',
                    'api_key': config.LUNARCRUSH_API_KEY
                }
            if hasattr(config, 'NEWSAPI_KEY'):
                self.fallback_sources['newsapi'] = {
                    'url': 'https://newsapi.org/v2/everything',
                    'api_key': config.NEWSAPI_KEY
                }
            if hasattr(config, 'CRYPTOCOMPARE_API_KEY'):
                self.fallback_sources['cryptocompare'] = {
                    'url': 'https://min-api.cryptocompare.com/data/v2/news/',
                    'api_key': config.CRYPTOCOMPARE_API_KEY
                }
            if hasattr(config, 'MESSARI_API_KEY'):
                self.fallback_sources['messari'] = {
                    'url': 'https://data.messari.io/api/v1/news',
                    'api_key': config.MESSARI_API_KEY
                }
        except Exception as e:
            self.logger.error(f"Error initializing fallback news sources: {e}")

    def analyze_news(self, symbol, hours=24):
        """Analyze news from all sources"""
        try:
            # Check cache first
            cache_file = f'news_cache/{symbol.lower()}_news.json'
            if self.check_cache(cache_file):
                self.logger.info(f"Using cached news data for {symbol}")
                return self.load_cache(cache_file)
            
            news_data = []
            base_symbol = symbol.replace('USDT', '').replace('BTC', '')
            failed_sources = []
            
            # Try web scraping sources first
            for source, info in self.news_sources.items():
                try:
                    articles = self.fetch_news(info['url'], base_symbol, hours, info['selectors'])
                    if articles:
                        for article in articles:
                            sentiment = self.analyze_sentiment(article['text'])
                            article.update({
                                'source': source,
                                'type': info['type'],
                                'weight': self.news_weights[info['type']],
                                'sentiment': sentiment
                            })
                            news_data.append(article)
                        self.logger.info(f"Successfully fetched {len(articles)} articles from {source}")
                    else:
                        failed_sources.append(source)
                        
                except Exception as e:
                    self.logger.warning(f"Error fetching news from {source}: {e}")
                    failed_sources.append(source)
                    continue
                    
                # No sleep, continue immediately with next source
            
            # Try API sources in parallel if we have few articles
            if len(news_data) < 5:
                self.logger.info(f"Only found {len(news_data)} articles from web scraping, trying API sources...")
                api_data = self.fetch_from_apis_parallel(symbol, hours)
                if api_data:
                    news_data.extend(api_data)
                    self.logger.info(f"Added {len(api_data)} articles from API sources")
            
            # Log summary
            if failed_sources:
                self.logger.warning(f"Failed sources: {', '.join(failed_sources)}")
            self.logger.info(f"Total articles collected: {len(news_data)}")
            
            # Calculate impact even with partial data
            impact_analysis = self.calculate_news_impact(news_data)
            
            # Cache results if we have any data
            if news_data:
                self.save_cache(cache_file, impact_analysis)
            
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"Error in news analysis for {symbol}: {e}")
            return {
                'impact_score': 0,
                'sentiment_score': 0,
                'article_count': 0,
                'major_events': [],
                'latest_update': datetime.now().isoformat()
            }

    def fetch_news(self, base_url, symbol, hours, selectors):
        """Fetch news with fallback to direct request"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive'
            }
            
            url = f"{base_url}{symbol}"
            
            # Try with proxy first
            try:
                proxy = self.proxy_rotator.get_next_proxy()
                if proxy:
                    response = requests.get(url, headers=headers, proxies=proxy, timeout=10)
                    if response.status_code == 200:
                        return self.parse_response(response, selectors, hours)
            except Exception as e:
                self.logger.warning(f"Proxy request failed: {e}, trying direct request")
            
            # Fallback to direct request
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return self.parse_response(response, selectors, hours)
            elif response.status_code == 403:
                self.logger.warning(f"Access denied to {base_url}, skipping")
                return []
            else:
                response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Error fetching news from {base_url}: {e}")
            return []

    def parse_response(self, response, selectors, hours):
        """Parse news articles from response"""
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for article in soup.select(selectors['article']):
                try:
                    pub_time = self.extract_publish_time(article, selectors['date'])
                    if pub_time and pub_time > cutoff_time:
                        articles.append({
                            'title': article.select_one(selectors['title']).text.strip(),
                            'text': article.select_one(selectors['text']).text.strip(),
                            'url': article.find('a')['href'],
                            'published': pub_time.isoformat()
                        })
                except Exception as e:
                    self.logger.debug(f"Error parsing article: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            return []

    def extract_publish_time(self, article, date_selector):
        """Extract and parse publication time with support for multiple formats"""
        try:
            date_text = article.select_one(date_selector).text.strip()
            
            # Try different date formats
            formats = [
                '%Y-%m-%d %H:%M:%S',           # 2024-12-08 09:31:55
                '%Y-%m-%dT%H:%M:%S',           # 2024-12-08T09:31:55
                '%b %d, %Y %H:%M',             # Dec 08, 2024 09:31
                '%d %b %Y %H:%M:%S %z',        # 08 Dec 2024 09:31:55 +0000
                '%Y-%m-%d',                    # 2024-12-08
                '%B %d, %Y',                   # December 08, 2024
                '%d/%m/%Y %H:%M'               # 08/12/2024 09:31
            ]
            
            for date_format in formats:
                try:
                    return datetime.strptime(date_text, date_format)
                except ValueError:
                    continue
            
            # If no format matches, try to extract using dateutil parser
            from dateutil import parser
            return parser.parse(date_text)
            
        except Exception as e:
            self.logger.error(f"Error parsing date '{date_text}': {e}")
            return None

    def analyze_sentiment(self, text):
        """Analyze sentiment of news article"""
        try:
            blob = TextBlob(text)
            vader_sentiment = self.sentiment_analyzer.polarity_scores(text)
            
            # Combine TextBlob and VADER sentiments
            sentiment = {
                'textblob_sentiment': blob.sentiment.polarity,
                'vader_sentiment': vader_sentiment['compound'],
                'subjectivity': blob.sentiment.subjectivity,
                'vader_details': {
                    'pos': vader_sentiment['pos'],
                    'neg': vader_sentiment['neg'],
                    'neu': vader_sentiment['neu']
                }
            }
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return None

    def calculate_news_impact(self, news_data):
        """Calculate overall news impact"""
        try:
            if not news_data:
                return {
                    'impact_score': 0,
                    'sentiment_score': 0,
                    'article_count': 0,
                    'major_events': [],
                    'latest_update': datetime.now().isoformat()
                }
            
            # Calculate weighted sentiment
            total_weight = 0
            weighted_sentiment = 0
            major_events = []
            
            for article in news_data:
                weight = article['weight']
                sentiment = (article['sentiment']['vader_sentiment'] + 
                           article['sentiment']['textblob_sentiment']) / 2
                
                weighted_sentiment += sentiment * weight
                total_weight += weight
                
                # Identify major events (high impact news)
                if abs(sentiment) > 0.5:
                    major_events.append({
                        'title': article['title'],
                        'sentiment': sentiment,
                        'source': article['source'],
                        'published': article['published']
                    })
            
            avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
            
            return {
                'impact_score': avg_sentiment * len(news_data) / 10,  # Scale impact by article count
                'sentiment_score': avg_sentiment,
                'article_count': len(news_data),
                'major_events': sorted(major_events, key=lambda x: abs(x['sentiment']), reverse=True),
                'latest_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating news impact: {e}")
            return None

    def check_cache(self, cache_file):
        """Check if cache is valid"""
        try:
            if not os.path.exists(cache_file):
                return False
                
            with open(cache_file, 'r') as f:
                data = json.load(f)
                last_update = datetime.fromisoformat(data['latest_update'])
                return (datetime.now() - last_update).total_seconds() < self.cache_duration
                
        except Exception as e:
            self.logger.error(f"Error checking cache: {e}")
            return False

    def load_cache(self, cache_file):
        """Load cached news data"""
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            return None

    def save_cache(self, cache_file, data):
        """Save news data to cache"""
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")

    def analyze_on_chain_metrics(self, symbol, chain='ethereum'):
        """Analyze on-chain metrics"""
        try:
            base_symbol = symbol.replace('USDT', '').replace('BTC', '')
            chain_info = self.chain_sources.get(chain)
            
            if not chain_info:
                return None
                
            metrics = {
                'active_addresses': self.get_active_addresses(base_symbol, chain_info),
                'transaction_volume': self.get_transaction_volume(base_symbol, chain_info),
                'whale_movements': self.get_whale_movements(base_symbol, chain_info),
                'network_growth': self.get_network_growth(base_symbol, chain_info),
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate composite score
            metrics['health_score'] = self.calculate_health_score(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing on-chain metrics: {e}")
            return None

    def get_active_addresses(self, symbol, chain_info):
        """Get number of active addresses"""
        try:
            params = {
                'module': 'proxy',
                'action': 'eth_getBlockByNumber',
                'apikey': chain_info['api_key']
            }
            
            response = requests.get(chain_info['url'], params=params)
            data = response.json()
            
            return {
                'daily_active': len(set(data['result']['transactions'])),
                'trend': self.calculate_address_trend(data)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting active addresses: {e}")
            return None

    def get_whale_movements(self, symbol, chain_info):
        """Track large wallet movements"""
        try:
            params = {
                'module': 'account',
                'action': 'tokentx',
                'contractaddress': self.get_token_contract(symbol),
                'apikey': chain_info['api_key']
            }
            
            response = requests.get(chain_info['url'], params=params)
            transactions = response.json()['result']
            
            whale_threshold = self.calculate_whale_threshold(transactions)
            whale_txs = [tx for tx in transactions if float(tx['value']) > whale_threshold]
            
            return {
                'count': len(whale_txs),
                'volume': sum(float(tx['value']) for tx in whale_txs),
                'direction': self.analyze_whale_direction(whale_txs)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing whale movements: {e}")
            return None

    def calculate_health_score(self, metrics):
        """Calculate overall network health score"""
        try:
            scores = []
            
            if metrics['active_addresses']:
                scores.append(min(metrics['active_addresses']['daily_active'] / 1000, 1.0))
                
            if metrics['whale_movements']:
                whale_score = metrics['whale_movements']['volume'] / 1000000  # Normalize to millions
                scores.append(min(whale_score, 1.0))
                
            if metrics['network_growth']:
                growth_score = metrics['network_growth']['new_addresses'] / 100
                scores.append(min(growth_score, 1.0))
                
            return sum(scores) / len(scores) if scores else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return 0

    def fetch_from_apis_parallel(self, symbol, hours):
        """Fetch news from API sources in parallel"""
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            news_data = []
            base_symbol = symbol.replace('USDT', '').replace('BTC', '')
            successful_apis = []
            
            def fetch_from_source(source, info):
                try:
                    data = None
                    if source == 'cryptopanic':
                        data = self.fetch_cryptopanic(base_symbol, info['api_key'])
                    elif source == 'lunarcrush':
                        data = self.fetch_lunarcrush(base_symbol, info['api_key'])
                    elif source == 'newsapi':
                        data = self.fetch_newsapi(base_symbol, info['api_key'])
                    elif source == 'cryptocompare':
                        data = self.fetch_cryptocompare(base_symbol, info['api_key'])
                    elif source == 'messari':
                        data = self.fetch_messari(base_symbol, info['api_key'])
                    return source, data
                except Exception as e:
                    self.logger.warning(f"Error fetching from {source}: {e}")
                    return source, None
            
            # Use ThreadPoolExecutor to fetch from all sources in parallel
            with ThreadPoolExecutor(max_workers=len(self.fallback_sources)) as executor:
                future_to_source = {
                    executor.submit(fetch_from_source, source, info): source 
                    for source, info in self.fallback_sources.items()
                }
                
                for future in as_completed(future_to_source):
                    source, data = future.result()
                    if data:
                        news_data.extend(data)
                        successful_apis.append(source)
                        self.logger.info(f"Successfully fetched {len(data)} articles from {source}")
            
            if successful_apis:
                self.logger.info(f"Successfully fetched from APIs: {', '.join(successful_apis)}")
            
            return news_data
            
        except Exception as e:
            self.logger.error(f"Error in parallel API fetching: {e}")
            return []

class ProxyRotator:
    def __init__(self, logger):
        self.logger = logger
        self.proxies = []
        self.current_index = 0
        self.last_update = 0
        self.update_interval = 3600  # Update proxy list every hour
        
    def get_proxy_list(self):
        """Get fresh proxy list"""
        try:
            # Add your preferred proxy sources
            sources = [
                'https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all',
                'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt',
                'https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt'
            ]
            
            proxies = set()
            for source in sources:
                response = requests.get(source, timeout=10)
                if response.status_code == 200:
                    proxy_list = response.text.strip().split('\n')
                    proxies.update(proxy_list)
            
            return list(proxies)
            
        except Exception as e:
            self.logger.error(f"Error getting proxy list: {e}")
            return []
    
    def get_next_proxy(self):
        """Get next working proxy"""
        try:
            current_time = time.time()
            
            # Update proxy list if needed
            if current_time - self.last_update > self.update_interval:
                self.proxies = self.get_proxy_list()
                self.last_update = current_time
                self.current_index = 0
            
            # If no proxies available, return None
            if not self.proxies:
                return None
            
            # Try proxies until we find a working one
            for _ in range(len(self.proxies)):
                proxy = self.proxies[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.proxies)
                
                if self.test_proxy(proxy):
                    return {
                        'http': f'http://{proxy}',
                        'https': f'http://{proxy}'
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting next proxy: {e}")
            return None
    
    def test_proxy(self, proxy):
        """Test if proxy is working"""
        try:
            proxies = {
                'http': f'http://{proxy}',
                'https': f'http://{proxy}'
            }
            response = requests.get('https://api.binance.com/api/v3/time',
                                 proxies=proxies,
                                 timeout=5)
            return response.status_code == 200
        except:
            return False
