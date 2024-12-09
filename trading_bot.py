from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
import ta
import config
from datetime import datetime, timedelta
import json
import os
from logger import setup_logger
import time
from social_monitor import SocialMonitor
from ml_predictor.base import MLPredictor
from market_communicator import MarketCommunicator
from order_flow_analyzer import OrderFlowAnalyzer
from options_analyzer import OptionsAnalyzer
from risk_metrics import RiskAnalyzer
from news_analyzer import NewsAnalyzer
import argparse
import logging
from typing import Optional, Dict

class TradingBot:
    def __init__(self, logger, min_volume_btc=100):
        self.client = Client(config.API_KEY, config.API_SECRET)
        self.logger = logger
        self.min_volume_btc = min_volume_btc  # Minimum 24h volume in BTC
        self.trading_pairs = []
        self.opportunities = []
        
        # Initialize social monitor
        self.social_monitor = SocialMonitor(logger)
        
        # Create opportunities directory if it doesn't exist
        if not os.path.exists('opportunities'):
            os.makedirs('opportunities')
            
        self.ml_predictor = MLPredictor(logger)
        self.market_comm = MarketCommunicator(logger)
        
        # Initialize new analyzers
        self.order_flow_analyzer = OrderFlowAnalyzer(logger, self.client)
        self.options_analyzer = OptionsAnalyzer(logger)
        self.risk_analyzer = RiskAnalyzer(logger)
        self.news_analyzer = NewsAnalyzer(logger)
        
    def get_viable_trading_pairs(self):
        """Get all trading pairs that meet volume requirements"""
        try:
            # Get exchange info
            exchange_info = self.client.get_exchange_info()
            
            # Get 24h ticker for all symbols
            tickers = self.client.get_ticker()
            volume_dict = {t['symbol']: float(t['volume']) * float(t['weightedAvgPrice']) 
                         for t in tickers}
            
            viable_pairs = []
            for symbol in exchange_info['symbols']:
                if (symbol['status'] == 'TRADING' and 
                    symbol['quoteAsset'] in ['BTC', 'USDT'] and
                    volume_dict.get(symbol['symbol'], 0) >= self.min_volume_btc):
                    viable_pairs.append(symbol['symbol'])
            
            self.logger.info(f"Found {len(viable_pairs)} viable trading pairs")
            return viable_pairs
            
        except Exception as e:
            self.logger.error(f"Error getting trading pairs: {e}")
            return []

    def analyze_pair(self, symbol: str) -> Optional[Dict]:
        """Main analysis workflow for a trading pair"""
        try:
            self.logger.info(f"Analyzing {symbol}...")
            
            # 1. Get Data from APIs
            try:
                # Get historical and new data
                df = self.data_fetcher.get_historical_data(symbol)
                if df is None or df.empty:
                    self.logger.error(f"No data available for {symbol}")
                    return None
                    
                # Get social and news data in parallel
                social_data = self.get_social_data(symbol)
                news_data = self.news_analyzer.analyze(symbol)
                
                # Get order flow and options data if available
                order_flow = self.get_order_flow_data(symbol)
                options_data = self.get_options_data(symbol)
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                return None

            # 2. Analyze Data and Train Models
            try:
                # Calculate risk metrics
                risk_metrics = self.risk_analyzer.analyze(df)
                
                # Get ML predictions
                predictions = self.get_ml_predictions(df, symbol)
                
                # Calculate social sentiment
                social_score = social_data.get('social_score', 0)
                if news_data and 'sentiment_score' in news_data:
                    combined_sentiment = (social_score + news_data['sentiment_score']) / 2
                else:
                    combined_sentiment = social_score
                
            except Exception as e:
                self.logger.error(f"Error in analysis for {symbol}: {e}")
                return None

            # 3. Plot Historical Charts and 4. Predict Future Trends
            try:
                # Generate all plots
                plot_data = self.market_comm.plot_prediction(
                    symbol=symbol,
                    df=df,
                    predictions=predictions,
                    order_flow=order_flow,
                    options_data=options_data
                )
                
                # Get trading signals
                signals = {
                    'Technical': {
                        'SMA': self.check_sma_crossover(df),
                        'RSI': self.check_rsi(df),
                        'MACD': self.check_macd(df),
                        'BB': self.check_bollinger_bands(df)
                    },
                    'Sentiment': {
                        'Social': 'BUY' if combined_sentiment > 0.2 else 'NEUTRAL',
                        'News': self.interpret_news(news_data) if news_data else 'NEUTRAL'
                    },
                    'Market': {
                        'OrderFlow': self.interpret_order_flow(order_flow) if order_flow else 'NEUTRAL',
                        'Options': self.interpret_options_data(options_data) if options_data else 'NEUTRAL'
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Error in visualization for {symbol}: {e}")
                return None

            # 5. Generate Tweet
            try:
                tweet = self.market_comm.generate_tweet(
                    symbol=symbol,
                    predictions=predictions,
                    order_flow=order_flow,
                    options_data=options_data,
                    news_data=news_data,
                    analysis_dir=plot_data['analysis_dir']
                )
                
            except Exception as e:
                self.logger.error(f"Error generating tweet for {symbol}: {e}")
                tweet = None

            # Compile final result
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'price': df['close'].iloc[-1],
                    'volume_24h': df['volume'].iloc[-24:].sum(),
                    'predictions': predictions,
                    'risk_metrics': risk_metrics
                },
                'sentiment': {
                    'social_score': social_score,
                    'news_data': news_data,
                    'combined_sentiment': combined_sentiment
                },
                'signals': signals,
                'plots': plot_data,
                'tweet': tweet,
                'metadata': {
                    'social_mentions': {
                        'twitter_count': social_data.get('twitter_count', 0),
                        'reddit_count': social_data.get('reddit_count', 0),
                        'news_count': news_data.get('article_count', 0) if news_data else 0
                    }
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in main analysis workflow for {symbol}: {e}")
            return None

    def check_sma_crossover(self, df):
        """Check for SMA crossover signals"""
        if df['SMA20'].iloc[-2] < df['SMA50'].iloc[-2] and \
           df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1]:
            return 'BUY'
        return 'NEUTRAL'

    def check_rsi(self, df):
        """Check for RSI signals"""
        if df['RSI'].iloc[-1] < 30:
            return 'BUY'
        return 'NEUTRAL'

    def check_macd(self, df):
        """Check for MACD signals"""
        if df['MACD'].iloc[-2] < df['MACD_signal'].iloc[-2] and \
           df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
            return 'BUY'
        return 'NEUTRAL'

    def check_bollinger_bands(self, df):
        """Check for Bollinger Bands signals"""
        if df['close'].iloc[-1] < df['BB_lower'].iloc[-1]:
            return 'BUY'
        return 'NEUTRAL'

    def backtest_all_strategies(self, df):
        """Backtest all strategies"""
        results = []
        strategies = ['SMA_crossover', 'RSI', 'MACD', 'BB']
        
        for strategy in strategies:
            profit = self.backtest_strategy(df, strategy)
            results.append({
                'strategy': strategy,
                'profit': profit
            })
        
        return results

    def scan_market(self):
        """Scan entire market for trading opportunities"""
        self.logger.info("Starting market scan...")
        
        # Get viable trading pairs
        self.trading_pairs = self.get_viable_trading_pairs()
        
        # Take only top 20 pairs by volume for analysis
        self.trading_pairs = self.trading_pairs[:20]
        
        # Analyze each pair
        opportunities = []
        analyses = []
        for symbol in self.trading_pairs:
            self.logger.info(f"Analyzing {symbol}...")
            result = self.analyze_pair(symbol)
            if result is not None:
                if result.get('is_opportunity', False):
                    opportunities.append(result)
                analyses.append(result)
        
        # Sort opportunities by average profit
        opportunities.sort(key=lambda x: x.get('average_profit', 0), reverse=True)
        
        # Save both opportunities and analyses
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'opportunities/scan_{timestamp}.json', 'w') as f:
            json.dump(opportunities, f, indent=4)
        with open(f'opportunities/analysis_{timestamp}.json', 'w') as f:
            json.dump(analyses, f, indent=4)
        
        self.logger.info(f"Found {len(opportunities)} trading opportunities")
        self.logger.info(f"Analyzed {len(analyses)} pairs")
        return opportunities, analyses

    def print_opportunities(self, opportunities):
        """Print trading opportunities in a readable format"""
        self.logger.info("\n=== Trading Opportunities ===")
        for opp in opportunities[:10]:
            self.logger.info(f"\nSymbol: {opp['symbol']}")
            self.logger.info(f"Positive Signals: {opp['positive_signals']}/6")
            self.logger.info(f"Average Backtest Profit: {opp['average_profit']:.2f}%")
            
            # Print social and news metrics
            self.logger.info(f"Social Score: {opp['social_score']:.2f}")
            self.logger.info(f"Social Mentions: Twitter: {opp['social_mentions']['twitter_count']}, "
                            f"Reddit: {opp['social_mentions']['reddit_count']}, "
                            f"News: {opp['social_mentions']['news_count']}")
            
            # Print news events if available
            if opp.get('news_data') and opp['news_data'].get('major_events'):
                self.logger.info("\nMajor News Events:")
                for event in opp['news_data']['major_events'][:3]:  # Show top 3 events
                    self.logger.info(f"  - {event['title']} (Sentiment: {event['sentiment']:.2f})")
            
            self.logger.info("Signals:")
            for strategy, signal in opp['signals'].items():
                self.logger.info(f"  {strategy}: {signal}")
            self.logger.info(f"Current Price: {opp['current_price']}")
            if opp.get('ml_predictions'):
                self.logger.info("ML Predictions:")
                self.logger.info(f"  Average Expected Change: {opp['ml_predictions']['avg_change']:.2f}%")
                self.logger.info(f"  Max Potential Change: {opp['ml_predictions']['max_change']:.2f}%")
                self.logger.info(f"  Min Potential Change: {opp['ml_predictions']['min_change']:.2f}%")
            if opp.get('suggested_tweet'):
                self.logger.info("\nSuggested Tweet:")
                self.logger.info(opp['suggested_tweet'])
            if opp.get('ta_plot_path'):
                self.logger.info(f"TA Prediction plot saved: {opp['ta_plot_path']}")
            if opp.get('pred_plot_path'):
                self.logger.info(f"Prediction plot saved: {opp['pred_plot_path']}")
            self.logger.info("---")

    def get_historical_data(self, symbol, lookback_days=30):
        """Get historical klines/candlestick data with caching"""
        try:
            # Create historical_data directory if it doesn't exist
            if not os.path.exists('historical_data'):
                os.makedirs('historical_data')
            
            # Define cache file path
            cache_file = f'historical_data/{symbol.lower()}_data.json'
            current_time = datetime.now()
            
            # Check if cache file exists and is recent
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    last_update = datetime.fromisoformat(cached_data['last_update'])
                    
                    # If cache is less than 1 hour old, use it
                    if (current_time - last_update).total_seconds() < 3600:
                        self.logger.info(f"Using cached historical data for {symbol}")
                        df = pd.DataFrame(cached_data['data'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        # Calculate indicators for cached data
                        df = self.calculate_indicators(df)
                        return df
            
            # Get new data from Binance
            klines = self.client.get_historical_klines(
                symbol,
                Client.KLINE_INTERVAL_1HOUR,
                f"{lookback_days} days ago UTC"
            )
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'close_time', 'quote_asset_volume',
                'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Save raw data to cache
            cache_data = {
                'last_update': current_time.isoformat(),
                'data': [{
                    **row,
                    'timestamp': row['timestamp'].isoformat()
                } for row in df.to_dict('records')]
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # Moving Averages
            df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['SMA50'] = ta.trend.sma_indicator(df['close'], window=50)
            
            # RSI
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_lower'] = bollinger.bollinger_lband()
            
            # Volume indicators
            df['Volume_SMA'] = ta.trend.sma_indicator(df['volume'], window=20)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return df

    def backtest_strategy(self, df, strategy):
        """Backtest a specific trading strategy"""
        try:
            signals = []
            profits = []
            current_position = None
            entry_price = 0
            
            for i in range(1, len(df)):
                signal = None
                
                if strategy == 'SMA_crossover':
                    if df['SMA20'].iloc[i-1] < df['SMA50'].iloc[i-1] and \
                       df['SMA20'].iloc[i] > df['SMA50'].iloc[i]:
                        signal = 'BUY'
                    elif df['SMA20'].iloc[i-1] > df['SMA50'].iloc[i-1] and \
                         df['SMA20'].iloc[i] < df['SMA50'].iloc[i]:
                        signal = 'SELL'
                        
                elif strategy == 'RSI':
                    if df['RSI'].iloc[i] < 30:
                        signal = 'BUY'
                    elif df['RSI'].iloc[i] > 70:
                        signal = 'SELL'
                        
                elif strategy == 'MACD':
                    if df['MACD'].iloc[i-1] < df['MACD_signal'].iloc[i-1] and \
                       df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                        signal = 'BUY'
                    elif df['MACD'].iloc[i-1] > df['MACD_signal'].iloc[i-1] and \
                         df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                        signal = 'SELL'
                        
                elif strategy == 'BB':
                    if df['close'].iloc[i] < df['BB_lower'].iloc[i]:
                        signal = 'BUY'
                    elif df['close'].iloc[i] > df['BB_upper'].iloc[i]:
                        signal = 'SELL'
                
                if signal == 'BUY' and current_position is None:
                    current_position = 'LONG'
                    entry_price = df['close'].iloc[i]
                elif signal == 'SELL' and current_position == 'LONG':
                    profit_pct = (df['close'].iloc[i] - entry_price) / entry_price * 100
                    profits.append(profit_pct)
                    current_position = None
                
                signals.append(signal)
                
            return np.mean(profits) if profits else 0
            
        except Exception as e:
            self.logger.error(f"Error in backtest for strategy {strategy}: {e}")
            return 0

    def interpret_order_flow(self, order_flow):
        """Interpret order flow signals"""
        try:
            if not order_flow:
                return 'NEUTRAL'
                
            # Check order book imbalance
            imbalance = order_flow['order_book_imbalance']
            if imbalance['interpretation'] == 'bullish' and imbalance['imbalance_ratio'] > 0.2:
                return 'BUY'
            elif imbalance['interpretation'] == 'bearish' and imbalance['imbalance_ratio'] < -0.2:
                return 'SELL'
                
            return 'NEUTRAL'
            
        except Exception as e:
            self.logger.error(f"Error interpreting order flow: {e}")
            return 'NEUTRAL'

    def interpret_options_data(self, options_data):
        """Interpret options market signals"""
        try:
            if not options_data:
                return 'NEUTRAL'
                
            # Check put/call ratio
            pc_ratio = options_data['put_call_ratio']
            if pc_ratio['sentiment'] == 'bullish' and pc_ratio['ratio'] < 0.7:
                return 'BUY'
            elif pc_ratio['sentiment'] == 'bearish' and pc_ratio['ratio'] > 1.3:
                return 'SELL'
                
            return 'NEUTRAL'
            
        except Exception as e:
            self.logger.error(f"Error interpreting options data: {e}")
            return 'NEUTRAL'

    def get_ml_predictions(self, df, symbol):
        """Get ML predictions with error handling"""
        try:
            self.ml_predictor.train_models(df, symbol)
            return self.ml_predictor.predict_future(df, symbol)
        except Exception as e:
            self.logger.error(f"Error in ML prediction for {symbol}: {e}")
            return None

    def get_social_data(self, symbol):
        """Get social data with error handling"""
        try:
            twitter_data = self.social_monitor.get_twitter_mentions(symbol)
            reddit_data = self.social_monitor.get_reddit_mentions(symbol)
            social_score = self.social_monitor.calculate_social_score(twitter_data, reddit_data)
            
            return {
                'social_score': social_score,
                'twitter_count': len(twitter_data),
                'reddit_count': len(reddit_data)
            }
        except Exception as e:
            self.logger.error(f"Error getting social data for {symbol}: {e}")
            return {'social_score': 0, 'twitter_count': 0, 'reddit_count': 0}

    def interpret_news(self, news_data):
        """Interpret news sentiment"""
        try:
            if not news_data:
                return 'NEUTRAL'
            
            # Check impact score and major events
            impact_score = news_data.get('impact_score', 0)
            major_events = news_data.get('major_events', [])
            
            if impact_score > 0.5 or (major_events and any(e['sentiment'] > 0.5 for e in major_events)):
                return 'BUY'
            elif impact_score < -0.5 or (major_events and any(e['sentiment'] < -0.5 for e in major_events)):
                return 'SELL'
            
            return 'NEUTRAL'
        
        except Exception as e:
            self.logger.error(f"Error interpreting news: {e}")
            return 'NEUTRAL'

def setup_logger():
    logger = logging.getLogger('trading_bot')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='Crypto Trading Bot')
    parser.add_argument('--pair', '-p', type=str, help='Trading pair (e.g., BTCUSDT)')
    parser.add_argument('--timeframe', '-t', type=str, default='1h', 
                       choices=['1h', '1d', '1m'], help='Trading timeframe')
    parser.add_argument('--mode', '-m', type=str, default='trade',
                       choices=['trade', 'backtest', 'analyze'],
                       help='Bot operation mode')
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger()
    
    # Initialize MLPredictor with proper configuration
    predictor = MLPredictor(logger=logger)
    
    if args.pair:
        # Run for specific trading pair
        logger.info(f"Analyzing {args.pair}...")
        predictor.analyze_symbol(args.pair, args.timeframe, args.mode)
    else:
        # Run for default pairs
        default_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        for pair in default_pairs:
            logger.info(f"Analyzing {pair}...")
            predictor.analyze_symbol(pair, args.timeframe, args.mode)

if __name__ == "__main__":
    main() 