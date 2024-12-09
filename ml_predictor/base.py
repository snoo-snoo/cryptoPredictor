import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
import logging
import json
from .config.settings import Settings
from .models.factory import ModelFactory
from .data.processor import DataProcessor
from .sentiment.analyzer import SentimentAnalyzer
from .monitoring.metrics import PerformanceMonitor
import time
from .utils.file_lock import file_lock
from .visualization.plotter import PredictionPlotter
from typing import Optional
from openai import OpenAI
from config import OPENAI_API_KEY

class MLPredictor:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = Settings()
        
        # Initialize components
        self.model_factory = ModelFactory(self.config)
        self.data_processor = DataProcessor(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config, self.logger)
        self.performance_monitor = PerformanceMonitor(self.logger)
        
        self.models = {}
        self.prediction_horizons = self.config.data_settings['timeframes']
        
        self.initialize_components()
        
        self.plotter = PredictionPlotter()
        
    def initialize_components(self):
        """Initialize all required components"""
        # Create required directories
        for directory in ['models', 'sentiment_data', 'news_cache']:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        # Initialize TensorFlow settings
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            self.logger.warning(f"Could not configure GPU memory growth: {e}")

    def predict(self, df, symbol, include_sentiment=True):
        """Main prediction method"""
        try:
            # Get base predictions
            predictions = self.predict_future(df, symbol)
            
            if include_sentiment:
                # Add sentiment analysis
                predictions = self.predict_with_sentiment(df, symbol, predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return None 

    def analyze_symbol(self, symbol: str, timeframe: str, mode: str = 'trade'):
        """Analyze specific trading pair"""
        try:
            # Get historical data
            df = self.get_historical_data(symbol, timeframe)
            if df is None or df.empty:
                self.logger.error(f"No data available for {symbol}")
                return None
            
            # Create analysis directory for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H")
            analysis_dir = os.path.join('analysis', timestamp)
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Analyze based on mode
            if mode == 'trade':
                # Get predictions
                predictions = self.predict(df, symbol)
                if not predictions:
                    return None
                
                # Generate and save plots
                plot_paths = self.plotter.plot_predictions(
                    df=df,
                    predictions=predictions,
                    symbol=symbol,
                    timeframe=timeframe
                )
                
                if plot_paths:
                    self.logger.info(f"Saved analysis files in: {plot_paths['directory']}")
                
                return {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'predictions': predictions,
                    'plots': plot_paths,
                    'directory': analysis_dir
                }
                
            elif mode == 'backtest':
                results = self.backtest(df, symbol)
                if results:
                    self.logger.info(f"Backtest results for {symbol}: {results}")
                return results
                
            elif mode == 'analyze':
                analysis = self.analyze_market(df, symbol)
                if analysis:
                    self.logger.info(f"Market analysis for {symbol}: {analysis}")
                return analysis
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get historical and new data for a symbol"""
        try:
            # First try to get new data from API
            new_data = self.fetch_market_data(symbol, timeframe)
            
            # Initialize historical data
            historical_data = None
            
            # Try to load cached historical data
            cache_file = f'historical_data/{symbol.lower()}_data.json'
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        historical_data = pd.DataFrame(cached_data['data'])
                        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
                        historical_data.set_index('timestamp', inplace=True)
                        self.logger.info(f"Loaded historical data for {symbol}")
                except Exception as e:
                    self.logger.warning(f"Error loading historical data: {e}")
            
            # Combine historical and new data
            if new_data is not None:
                if historical_data is not None:
                    # Combine and remove duplicates
                    combined_data = pd.concat([historical_data, new_data])
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    combined_data.sort_index(inplace=True)
                else:
                    combined_data = new_data
                
                # Save updated data
                try:
                    cache_data = {
                        'last_update': datetime.now().isoformat(),
                        'data': [{
                            'timestamp': idx.isoformat(),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume'])
                        } for idx, row in combined_data.iterrows()]
                    }
                    
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f, indent=2)
                    
                    return combined_data
                except Exception as e:
                    self.logger.error(f"Error saving combined data: {e}")
                    return combined_data  # Return data even if saving fails
            
            # If no new data but we have historical data, use that
            if historical_data is not None:
                return historical_data
            
            self.logger.error(f"No data available for {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {e}")
            return None

    def fetch_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch market data from exchange with retries"""
        try:
            # Initialize Binance client
            from binance.client import Client
            client = Client(None, None)
            
            # Convert timeframe to Binance interval
            interval_map = {
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY,
                '1m': Client.KLINE_INTERVAL_1MINUTE
            }
            interval = interval_map.get(timeframe, Client.KLINE_INTERVAL_1HOUR)
            
            # Try to get data with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    klines = client.get_historical_klines(
                        symbol,
                        interval,
                        f"30 days ago UTC"
                    )
                    
                    if not klines:
                        self.logger.warning(f"No data returned from API for {symbol}")
                        return None
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_base',
                        'taker_quote', 'ignore'
                    ])
                    
                    # Process data
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Convert numeric columns
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Clean data
                    df = df[numeric_cols].dropna()
                    
                    if df.empty:
                        self.logger.warning(f"No valid data after cleaning for {symbol}")
                        return None
                    
                    return df
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    self.logger.warning(f"Retry {attempt + 1}/{max_retries} fetching data for {symbol}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return None

    def analyze_market(self, df: pd.DataFrame, symbol: str) -> dict:
        """Analyze market data"""
        try:
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'metrics': {},
                'indicators': {},
                'patterns': {},
                'sentiment': None
            }
            
            # Calculate basic metrics
            analysis['metrics'] = {
                'current_price': float(df['close'].iloc[-1]),
                'daily_change': float(df['close'].pct_change().iloc[-1] * 100),
                'volume_24h': float(df['volume'].tail(24).sum()),
                'volatility': float(df['close'].pct_change().std() * 100)
            }
            
            # Add technical indicators
            analysis['indicators'] = self.calculate_indicators(df)
            
            # Add candlestick patterns
            analysis['patterns'] = self.detect_candlestick_patterns(df)
            
            # Add sentiment analysis if available
            sentiment = self.sentiment_analyzer.analyze(symbol)
            if sentiment:
                analysis['sentiment'] = sentiment
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return None

    def backtest(self, df: pd.DataFrame, symbol: str) -> dict:
        """Perform backtesting"""
        try:
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'metrics': {},
                'trades': [],
                'performance': {}
            }
            
            # Run backtest simulation
            trades, metrics = self.run_backtest_simulation(df)
            
            results['trades'] = trades
            results['metrics'] = metrics
            
            # Calculate performance metrics
            results['performance'] = {
                'total_return': sum(trade['profit'] for trade in trades),
                'win_rate': len([t for t in trades if t['profit'] > 0]) / len(trades) if trades else 0,
                'avg_profit': sum(trade['profit'] for trade in trades) / len(trades) if trades else 0,
                'max_drawdown': self.calculate_max_drawdown(trades)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")
            return None

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            indicators = {}
            
            # Calculate moving averages
            indicators['SMA20'] = df['close'].rolling(window=20).mean()
            indicators['SMA50'] = df['close'].rolling(window=50).mean()
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            indicators['MACD'] = exp1 - exp2
            indicators['MACD_signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return {}

    def run_backtest_simulation(self, df):
        """Run backtest simulation"""
        try:
            trades = []
            metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0.0
            }
            
            # Add your backtesting logic here
            # This is a simple example
            position = None
            entry_price = 0
            
            for i in range(1, len(df)):
                if position is None and self.should_enter_trade(df, i):
                    position = 'LONG'
                    entry_price = df['close'].iloc[i]
                    trades.append({
                        'type': 'ENTRY',
                        'price': entry_price,
                        'timestamp': df.index[i]
                    })
                elif position == 'LONG' and self.should_exit_trade(df, i):
                    exit_price = df['close'].iloc[i]
                    profit = (exit_price - entry_price) / entry_price * 100
                    trades.append({
                        'type': 'EXIT',
                        'price': exit_price,
                        'profit': profit,
                        'timestamp': df.index[i]
                    })
                    position = None
                    
                    # Update metrics
                    metrics['total_trades'] += 1
                    if profit > 0:
                        metrics['winning_trades'] += 1
                    else:
                        metrics['losing_trades'] += 1
                    metrics['total_profit'] += profit
            
            return trades, metrics
            
        except Exception as e:
            self.logger.error(f"Error in backtest simulation: {e}")
            return [], {}

    def calculate_max_drawdown(self, trades):
        """Calculate maximum drawdown from trades"""
        try:
            if not trades:
                return 0
            
            equity = 100  # Start with 100 units
            peak = equity
            max_drawdown = 0
            
            for trade in trades:
                if 'profit' in trade:
                    equity *= (1 + trade['profit'] / 100)
                    if equity > peak:
                        peak = equity
                    drawdown = (peak - equity) / peak * 100
                    max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0

    def predict_future(self, df: pd.DataFrame, symbol: str) -> dict:
        """Make predictions for future price movements"""
        try:
            predictions = {}
            current_price = float(df['close'].iloc[-1])
            
            # Calculate returns for scaling
            returns = pd.DataFrame({'close': df['close'].pct_change().fillna(0)})  # Create DataFrame with 'close' column
            
            for timeframe in self.prediction_horizons:
                try:
                    # Get sequence length from config
                    seq_length = self.config.model_settings.get('sequence_length', 60)
                    
                    # Prepare data using returns DataFrame
                    X, y = self.data_processor.prepare_data(returns, seq_length)
                    if X is None or y is None or len(X) == 0:
                        self.logger.warning(f"Could not prepare data for {timeframe} predictions")
                        continue
                    
                    # Get or train model
                    model_key = f"{timeframe}_{symbol}"
                    if model_key not in self.models:
                        self.logger.info(f"Training new model for {symbol} {timeframe}")
                        model = self.model_factory.create_model('lstm', seq_length=seq_length)
                        if not self.train_model(model, X, y, timeframe):
                            continue
                        self.models[model_key] = {
                            'model': model,
                            'last_training': datetime.now().isoformat()
                        }
                    
                    # Make predictions
                    model = self.models[model_key]['model']
                    horizon = self.prediction_horizons[timeframe]
                    future_returns = self.predict_sequence(model, X[-1:], horizon)
                    
                    if future_returns is not None:
                        # Convert returns to prices
                        predicted_prices = [current_price]
                        for ret in future_returns:
                            predicted_price = predicted_prices[-1] * (1 + ret)
                            predicted_prices.append(predicted_price)
                        
                        predictions[timeframe] = {
                            'horizon': horizon,
                            'current_price': current_price,
                            'predicted_prices': predicted_prices[1:],  # Remove initial price
                            'confidence': self.calculate_prediction_confidence(model, X, y),
                            'last_updated': datetime.now().isoformat()
                        }
                        
                        # Add change metrics
                        predictions[timeframe].update({
                            'avg_change': float(np.mean([(p - current_price) / current_price * 100 
                                                        for p in predicted_prices[1:]])),
                            'max_change': float(max([(p - current_price) / current_price * 100 
                                                   for p in predicted_prices[1:]])),
                            'min_change': float(min([(p - current_price) / current_price * 100 
                                                   for p in predicted_prices[1:]]))
                        })
                    
                except Exception as e:
                    self.logger.error(f"Error predicting {timeframe} for {symbol}: {e}")
                    continue
            
            return predictions if predictions else None
            
        except Exception as e:
            self.logger.error(f"Error in predict_future: {e}")
            return None

    def train_model(self, model, X, y, timeframe):
        """Train a model with early stopping"""
        try:
            # Split data
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.model_settings.get('epochs', 100),
                batch_size=self.config.model_settings.get('batch_size', 32),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ],
                verbose=0
            )
            
            # Validate training
            val_loss = min(history.history['val_loss'])
            if val_loss > 0.1:  # Threshold for acceptable model
                self.logger.warning(f"Model validation loss too high: {val_loss}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return False

    def predict_sequence(self, model, last_sequence, horizon):
        """Predict sequence of future values"""
        try:
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(horizon):
                # Predict next value
                next_pred = model.predict(current_sequence, verbose=0)[0][0]
                predictions.append(next_pred)
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting sequence: {e}")
            return None

    def calculate_prediction_confidence(self, model, X, y, n_samples=100):
        """Calculate prediction confidence using bootstrap"""
        try:
            predictions = []
            indices = np.random.randint(0, len(X), size=(n_samples, len(X)))
            
            for idx in indices:
                X_sample, y_sample = X[idx], y[idx]
                pred = model.predict(X_sample, verbose=0)
                predictions.append(pred)
            
            # Calculate confidence as inverse of prediction variance
            variance = np.var([p.mean() for p in predictions])
            confidence = 1 / (1 + variance)
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Default confidence

    def visualize_predictions(self, df, predictions, symbol, timeframe):
        """Create and save prediction visualizations"""
        try:
            # Create prediction plot
            plot_path = self.plotter.plot_predictions(df, predictions, symbol, timeframe)
            
            # Create performance plot if model exists
            model_key = f"{timeframe}_{symbol}"
            if model_key in self.models and 'history' in self.models[model_key]:
                perf_path = self.plotter.plot_model_performance(
                    self.models[model_key]['history'],
                    symbol,
                    timeframe
                )
            
            # Create accuracy plot if we have actual values
            if len(df) > 0:
                actual = df['close'].iloc[-len(predictions[timeframe]['predicted_prices']):]
                pred = pd.Series(predictions[timeframe]['predicted_prices'], index=actual.index)
                acc_path = self.plotter.plot_prediction_accuracy(pred, actual, symbol, timeframe)
            
            return {
                'prediction_plot': plot_path,
                'performance_plot': perf_path if 'perf_path' in locals() else None,
                'accuracy_plot': acc_path if 'acc_path' in locals() else None
            }
            
        except Exception as e:
            self.logger.error(f"Error visualizing predictions: {e}")
            return None

    def predict_with_sentiment(self, df: pd.DataFrame, symbol: str, predictions: dict) -> dict:
        """Combine ML predictions with sentiment analysis"""
        try:
            if not predictions:
                return None
            
            # Get sentiment analysis
            sentiment = self.sentiment_analyzer.analyze(symbol)
            if not sentiment:
                return predictions
            
            # Adjust predictions based on sentiment
            adjusted_predictions = {}
            for timeframe, pred in predictions.items():
                # Calculate sentiment impact (0.5 neutral, >0.5 positive, <0.5 negative)
                sentiment_impact = (sentiment['score'] - 0.5) * sentiment['confidence']
                
                # Adjust predicted prices
                adjusted_prices = [
                    price * (1 + sentiment_impact * 0.1)  # Max 10% impact
                    for price in pred['predicted_prices']
                ]
                
                # Update prediction data
                adjusted_predictions[timeframe] = {
                    **pred,
                    'predicted_prices': adjusted_prices,
                    'sentiment_score': sentiment['score'],
                    'sentiment_confidence': sentiment['confidence'],
                    'sentiment_impact': sentiment_impact * 100  # Convert to percentage
                }
            
            return adjusted_predictions
            
        except Exception as e:
            self.logger.error(f"Error in sentiment-adjusted prediction: {e}")
            return predictions  # Return original predictions if adjustment fails

    def generate_trading_tweet(self, symbol: str, predictions: dict, current_price: float) -> str:
        """Generate trading-focused tweet using ChatGPT"""
        try:
            # Prepare prediction summary
            prediction_text = []
            for timeframe, data in predictions.items():
                prediction_text.append(
                    f"{timeframe.upper()}: {data['avg_change']:.2f}% "
                    f"(Range: {data['min_change']:.2f}% to {data['max_change']:.2f}%)"
                )
            
            # Create prompt for ChatGPT
            prompt = f"""
            Create a concise, professional tweet about {symbol} trading analysis:
            
            Current Price: ${current_price:.8f}
            
            Predictions:
            {chr(10).join(prediction_text)}
            
            Requirements:
            1. Use cashtags (${symbol})
            2. Include relevant hashtags (#crypto #trading)
            3. Add a brief risk disclaimer
            4. Maximum 280 characters
            5. Be informative but cautious
            """
            
            # Create client instance
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Get response from ChatGPT using new API format
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                max_tokens=100,
                temperature=0.7
            )
            
            # Extract tweet from response using new response format
            tweet = response.choices[0].message.content.strip()
            
            # Save tweet to analysis directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tweet_dir = os.path.join('analysis', f"{symbol}_{timestamp}")
            os.makedirs(tweet_dir, exist_ok=True)
            
            tweet_path = os.path.join(tweet_dir, 'tweet.txt')
            with open(tweet_path, 'w', encoding='utf-8') as f:
                f.write(tweet)
            
            self.logger.info(f"Generated and saved tweet to {tweet_path}")
            
            return tweet
            
        except Exception as e:
            self.logger.error(f"Error generating tweet: {e}")
            return None