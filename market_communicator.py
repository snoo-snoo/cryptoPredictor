import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import openai
import config
import os
import numpy as np

class MarketCommunicator:
    def __init__(self, logger):
        self.logger = logger
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Create analysis directory structure
        self.base_dir = 'analysis'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def save_analysis(self, symbol, prediction_data, plot_path=None, tweet=None):
        """Save analysis data to structured directory"""
        try:
            # Create timestamp and symbol-specific directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_dir = os.path.join(self.base_dir, f"{symbol.lower()}_{timestamp}")
            os.makedirs(analysis_dir)
            
            # Save tweet if exists
            if tweet:
                tweet_file = os.path.join(analysis_dir, "tweet.txt")
                with open(tweet_file, 'w', encoding='utf-8') as f:
                    f.write(tweet)
            
            # Save prediction data
            data_file = os.path.join(analysis_dir, "prediction_data.txt")
            with open(data_file, 'w') as f:
                f.write(f"Symbol: {symbol}\n")
                f.write(f"Timestamp: {timestamp}\n")
                
                # Write predictions for each timeframe
                for timeframe, data in prediction_data.items():
                    f.write(f"\n{timeframe.upper()} Predictions:\n")
                    f.write(f"Current Price: {data['current_price']:.8f}\n")
                    f.write(f"Horizon: {data['horizon']}\n")
                    f.write(f"Average Expected Change: {data['avg_change']:.2f}%\n")
                    f.write(f"Max Potential Change: {data['max_change']:.2f}%\n")
                    f.write(f"Min Potential Change: {data['min_change']:.2f}%\n")
            
            return analysis_dir
            
        except Exception as e:
            self.logger.error(f"Error saving analysis: {e}")
            return None

    def plot_prediction(self, symbol, df, predictions, order_flow=None, options_data=None):
        """Create and save all prediction plots"""
        try:
            plots = self.plotter.plot_predictions(df, predictions, symbol, timeframe='1h')
            
            if plots:
                self.logger.info(f"Saved prediction plots for {symbol}:")
                self.logger.info(f"HTML: {plots['html']}")
                self.logger.info(f"PNG: {plots['png']}")
                
                # Save analysis summary
                summary = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'predictions': predictions,
                    'order_flow': order_flow,
                    'options_data': options_data,
                    'plots': plots
                }
                
                summary_path = self.plotter.save_analysis_summary(symbol, summary)
                if summary_path:
                    self.logger.info(f"Saved analysis summary: {summary_path}")
                
                return plots
                
        except Exception as e:
            self.logger.error(f"Error plotting predictions: {e}")
            return None

    def plot_order_flow(self, ax, order_flow, historical_data):
        """Plot order flow metrics"""
        try:
            if not order_flow or not order_flow.get('order_book_imbalance'):
                return
            
            imbalance = order_flow['order_book_imbalance']['imbalance_ratio']
            if isinstance(imbalance, (int, float)):
                # If single value, create array of same value
                imbalance = [imbalance] * len(historical_data)
            
            ax.plot(historical_data.index, imbalance, 
                    label='Order Book Imbalance', color='#42a5f5')
            
            if 'buy_pressure' in order_flow:
                buy_pressure = order_flow['buy_pressure']
                if isinstance(buy_pressure, (int, float)):
                    buy_pressure = [buy_pressure] * len(historical_data)
                ax.plot(historical_data.index, buy_pressure,
                       label='Buy Pressure', color='#26a69a')
            
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax.fill_between(historical_data.index, imbalance, 0,
                           where=np.array(imbalance) > 0,
                           color='#26a69a', alpha=0.1)
            ax.fill_between(historical_data.index, imbalance, 0,
                           where=np.array(imbalance) < 0,
                           color='#ef5350', alpha=0.1)
            
            ax.set_ylabel('Order Flow', color='white')
            ax.legend(loc='upper left', facecolor='black', edgecolor='white')
            ax.grid(True, alpha=0.2)
            
        except Exception as e:
            self.logger.error(f"Error plotting order flow: {e}")

    def plot_options_data(self, ax, options_data, historical_data):
        """Plot options market data"""
        try:
            if not options_data or not options_data.get('put_call_ratio'):
                return
            
            # Create time series for options data
            pc_ratio = options_data['put_call_ratio']['ratio']
            if isinstance(pc_ratio, (int, float)):
                # If single value, create array
                pc_ratios = [pc_ratio] * len(historical_data)
            else:
                pc_ratios = pc_ratio
            
            # Plot put/call ratio
            ax.plot(historical_data.index, pc_ratios,
                    label='Put/Call Ratio', color='#ba68c8')
            
            # Add implied volatility if available
            if 'implied_volatility' in options_data:
                iv_data = options_data['implied_volatility']
                if iv_data and len(iv_data) > 0:
                    # Extract IV values and create time series
                    iv_values = [item['iv'] for item in iv_data]
                    if len(iv_values) == 1:
                        iv_values = [iv_values[0]] * len(historical_data)
                    ax.plot(historical_data.index[-len(iv_values):], iv_values,
                           label='Implied Volatility', color='#ffca28')
            
            ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
            ax.set_ylabel('Options Data', color='white')
            ax.legend(loc='upper left', facecolor='black', edgecolor='white')
            ax.grid(True, alpha=0.2)
            
        except Exception as e:
            self.logger.error(f"Error plotting options data: {e}")

    def create_prediction_chart(self, symbol, current_data, ml_predictions, 
                              order_flow, options_data, analysis_dir):
        """Create separate prediction chart with all signals"""
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot historical and predicted prices
            self.plot_predictions(ax, current_data, ml_predictions)
            
            # Add market signals
            if order_flow and options_data:
                self.add_signal_markers(ax, order_flow, options_data)
            
            # Add prediction summary
            self.add_prediction_summary(ax, ml_predictions, order_flow, options_data)
            
            # Save prediction chart
            pred_path = os.path.join(analysis_dir, "price_predictions.png")
            self.save_figure(fig, pred_path)
            plt.close()
            
            return pred_path
            
        except Exception as e:
            self.logger.error(f"Error creating prediction chart: {e}")
            return None

    def generate_tweet(self, symbol, ml_predictions, order_flow, options_data, news_data, analysis_dir=None):
        """Generate comprehensive tweet with all signals"""
        try:
            # Create summary of all signals
            signals_summary = []
            
            # ML Predictions
            for timeframe, data in ml_predictions.items():
                signals_summary.append(
                    f"{timeframe.upper()}: {data['avg_change']:.2f}% "
                    f"({data['min_change']:.2f}% to {data['max_change']:.2f}%)"
                )
            
            # Order Flow
            if order_flow and order_flow['order_book_imbalance']:
                imbalance = order_flow['order_book_imbalance']
                signals_summary.append(
                    f"Order Flow: {imbalance['interpretation'].title()} "
                    f"({imbalance['imbalance_ratio']:.2f})"
                )
            
            # Options Data
            if options_data and options_data['put_call_ratio']:
                pc_ratio = options_data['put_call_ratio']
                signals_summary.append(
                    f"Options: {pc_ratio['sentiment'].title()} "
                    f"(P/C: {pc_ratio['ratio']:.2f})"
                )
            
            # News Data
            if news_data and news_data.get('major_events'):
                top_event = news_data['major_events'][0]
                signals_summary.append(
                    f"News: {top_event['title'][:50]}... "
                    f"(Sentiment: {top_event['sentiment']:.2f})"
                )
            
            prompt = f"""
            Create a short, engaging tweet about {symbol} with multiple signals:
            Current price: {ml_predictions['1h']['current_price']:.8f}
            
            Signals:
            {chr(10).join(signals_summary)}
            
            Make it informative but cautious, include a disclaimer.
            Use cashtags and hashtags appropriately.
            Maximum 280 characters.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            
            tweet = response.choices[0].message.content.strip()
            
            if analysis_dir:
                self.save_analysis(symbol, ml_predictions, tweet=tweet)
            
            return tweet
            
        except Exception as e:
            self.logger.error(f"Error generating tweet: {e}")
            return None 

    def add_technical_indicators(self, ax, data):
        """Add enhanced technical indicators to price chart"""
        try:
            # Moving Averages
            ax.plot(data.index, data['SMA20'], 
                    label='SMA20', color='#42a5f5', alpha=0.7, linewidth=1)
            ax.plot(data.index, data['SMA50'], 
                    label='SMA50', color='#ffca28', alpha=0.7, linewidth=1)
            
            # Bollinger Bands with gradient fill
            ax.plot(data.index, data['BB_upper'], 
                    color='#b39ddb', linestyle=':', alpha=0.7, label='BB')
            ax.plot(data.index, data['BB_lower'], 
                    color='#b39ddb', linestyle=':', alpha=0.7)
            ax.fill_between(data.index, data['BB_upper'], data['BB_lower'], 
                           color='#7e57c2', alpha=0.05)
            
            # Add support/resistance levels
            levels = self.calculate_support_resistance(data)
            for level in levels:
                ax.axhline(y=level, color='#e1bee7', 
                          linestyle='--', alpha=0.3)
            
            # Add trend lines
            self.add_trend_lines(ax, data)
            
            # Add volume profile on the right
            self.add_volume_profile(ax, data)
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")

    def add_trend_lines(self, ax, data):
        """Add trend lines to the chart"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            dates = np.arange(len(data))
            
            # Calculate trend lines using linear regression
            z_high = np.polyfit(dates[-20:], highs[-20:], 1)
            z_low = np.polyfit(dates[-20:], lows[-20:], 1)
            
            p_high = np.poly1d(z_high)
            p_low = np.poly1d(z_low)
            
            # Plot trend lines
            ax.plot(data.index[-20:], p_high(dates[-20:]), 
                    '--', color='#26a69a', alpha=0.5, linewidth=1)
            ax.plot(data.index[-20:], p_low(dates[-20:]), 
                    '--', color='#ef5350', alpha=0.5, linewidth=1)
            
        except Exception as e:
            self.logger.error(f"Error adding trend lines: {e}")

    def add_volume_profile(self, ax, data):
        """Add volume profile to the right side of the chart"""
        try:
            # Calculate volume profile
            price_bins = np.linspace(data['low'].min(), data['high'].max(), 50)
            volumes = np.zeros_like(price_bins)
            
            for i in range(len(price_bins)-1):
                mask = (data['close'] >= price_bins[i]) & (data['close'] < price_bins[i+1])
                volumes[i] = data.loc[mask, 'volume'].sum()
            
            # Normalize volumes
            max_vol = volumes.max()
            norm_volumes = volumes / max_vol * (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.1
            
            # Plot volume profile
            ax2 = ax.twinx()
            ax2.fill_betweenx(price_bins[:-1], 
                             ax.get_xlim()[1], 
                             ax.get_xlim()[1] - norm_volumes[:-1],
                             alpha=0.2, color='#b2dfdb')
            ax2.set_ylim(ax.get_ylim())
            ax2.set_ylabel('')
            ax2.set_xticks([])
            
        except Exception as e:
            self.logger.error(f"Error adding volume profile: {e}")

    def add_signal_markers(self, ax, order_flow, options_data):
        """Add signal markers to the prediction chart"""
        try:
            last_idx = len(ax.get_lines()[0].get_xdata()) - 1
            
            # Order flow signals
            if order_flow and order_flow['order_book_imbalance']:
                imbalance = order_flow['order_book_imbalance']['imbalance_ratio']
                if abs(imbalance) > 0.2:
                    color = '#26a69a' if imbalance > 0 else '#ef5350'
                    ax.scatter(last_idx, ax.get_lines()[0].get_ydata()[-1],
                             marker='^' if imbalance > 0 else 'v',
                             color=color, s=100, label='Order Flow Signal')
            
            # Options signals
            if options_data and options_data['put_call_ratio']:
                pc_ratio = options_data['put_call_ratio']['ratio']
                if pc_ratio < 0.7 or pc_ratio > 1.3:
                    color = '#26a69a' if pc_ratio < 0.7 else '#ef5350'
                    ax.scatter(last_idx, ax.get_lines()[0].get_ydata()[-1],
                             marker='*', color=color, s=150,
                             label='Options Signal')
            
        except Exception as e:
            self.logger.error(f"Error adding signal markers: {e}")

    def add_prediction_summary(self, ax, ml_predictions, order_flow, options_data):
        """Add enhanced prediction summary to chart"""
        try:
            summary_text = "Predictions & Signals:\n\n"
            
            # ML Predictions with confidence
            for timeframe, data in ml_predictions.items():
                confidence = self.calculate_prediction_confidence(data)
                summary_text += f"{timeframe.upper()} Forecast:\n"
                summary_text += f"→ Change: {data['avg_change']:.2f}%\n"
                summary_text += f"→ Range: {data['min_change']:.2f}% to {data['max_change']:.2f}%\n"
                summary_text += f"→ Confidence: {confidence:.1f}%\n\n"
            
            # Market Signals
            signals = self.combine_signals(ml_predictions, order_flow, options_data)
            summary_text += "Market Signals:\n"
            for signal, value in signals.items():
                summary_text += f"→ {signal}: {value}\n"
            
            # Add overall sentiment
            sentiment = self.calculate_overall_sentiment(signals)
            summary_text += f"\nOverall Sentiment: {sentiment}"
            
            # Add text box
            ax.text(1.02, 0.98, summary_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    fontsize=10,
                    color='white',
                    bbox=dict(facecolor='#1a237e',
                             edgecolor='#7986cb',
                             alpha=0.7,
                             pad=10,
                             boxstyle='round'))
            
        except Exception as e:
            self.logger.error(f"Error adding prediction summary: {e}")

    def combine_signals(self, ml_predictions, order_flow, options_data):
        """Combine all signals into a unified view"""
        try:
            signals = {}
            
            # ML Signal
            ml_sentiment = 'Bullish' if ml_predictions['1h']['avg_change'] > 0 else 'Bearish'
            signals['ML Trend'] = ml_sentiment
            
            # Order Flow Signal
            if order_flow and order_flow['order_book_imbalance']:
                signals['Order Flow'] = order_flow['order_book_imbalance']['interpretation']
            
            # Options Signal
            if options_data and options_data['put_call_ratio']:
                signals['Options'] = options_data['put_call_ratio']['sentiment']
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error combining signals: {e}")
            return {}

    def calculate_prediction_confidence(self, prediction_data):
        """Calculate confidence score for predictions"""
        try:
            # Factors affecting confidence:
            # 1. Range of predictions
            range_factor = 1 - min(abs(prediction_data['max_change'] - 
                                     prediction_data['min_change']) / 20, 1)
            
            # 2. Consistency with other timeframes
            consistency = 0.8  # Base confidence
            
            # 3. Historical accuracy (could be implemented with actual tracking)
            historical_accuracy = 0.7  # Example value
            
            # Combine factors
            confidence = (range_factor * 0.4 + 
                         consistency * 0.3 + 
                         historical_accuracy * 0.3) * 100
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction confidence: {e}")
            return 50  # Default confidence

    def calculate_overall_sentiment(self, signals):
        """Calculate overall market sentiment"""
        try:
            sentiment_scores = []
            
            for signal, value in signals.items():
                if value.lower() == 'bullish':
                    sentiment_scores.append(1)
                elif value.lower() == 'bearish':
                    sentiment_scores.append(-1)
                else:
                    sentiment_scores.append(0)
            
            avg_sentiment = np.mean(sentiment_scores)
            
            if avg_sentiment > 0.5:
                return "Strongly Bullish 🚀"
            elif avg_sentiment > 0:
                return "Moderately Bullish 📈"
            elif avg_sentiment < -0.5:
                return "Strongly Bearish 🔻"
            elif avg_sentiment < 0:
                return "Moderately Bearish 📉"
            else:
                return "Neutral ↔️"
                
        except Exception as e:
            self.logger.error(f"Error calculating overall sentiment: {e}")
            return "Neutral ↔️" 

    def calculate_support_resistance(self, data):
        """Calculate support and resistance levels"""
        try:
            # Use pivot points
            high = data['high'].max()
            low = data['low'].min()
            close = data['close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            return [s2, s1, pivot, r1, r2]
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return []

    def plot_rsi(self, ax, data):
        """Plot RSI indicator"""
        try:
            ax.plot(data.index, data['RSI'], color='#ce93d8')
            ax.axhline(y=70, color='#ef5350', linestyle='--', alpha=0.5)
            ax.axhline(y=30, color='#26a69a', linestyle='--', alpha=0.5)
            ax.fill_between(data.index, 70, 30, alpha=0.1, color='#7e57c2')
            ax.set_ylabel('RSI', color='white')
            ax.grid(True, alpha=0.2)
            
        except Exception as e:
            self.logger.error(f"Error plotting RSI: {e}")

    def plot_macd(self, ax, data):
        """Plot MACD indicator"""
        try:
            ax.plot(data.index, data['MACD'], label='MACD', color='#42a5f5')
            ax.plot(data.index, data['MACD_signal'], label='Signal', color='#ffca28')
            
            # Add MACD histogram
            macd_hist = data['MACD'] - data['MACD_signal']
            ax.bar(data.index, macd_hist, 
                   color=['#ef5350' if x < 0 else '#26a69a' for x in macd_hist],
                   alpha=0.3)
                   
            ax.set_ylabel('MACD', color='white')
            ax.legend(loc='upper left', facecolor='black', edgecolor='white')
            ax.grid(True, alpha=0.2)
            
        except Exception as e:
            self.logger.error(f"Error plotting MACD: {e}")

    def plot_volume(self, ax, data):
        """Plot volume bars"""
        try:
            colors = ['#ef5350' if close < open else '#26a69a' 
                     for close, open in zip(data['close'], data['open'])]
            ax.bar(data.index, data['volume'], color=colors, alpha=0.7)
            ax.plot(data.index, data['Volume_SMA'],
                    color='#42a5f5', label='Volume SMA', linewidth=2)
            ax.set_ylabel('Volume', color='white')
            ax.legend(loc='upper left', facecolor='black', edgecolor='white')
            ax.grid(True, alpha=0.2)
            
        except Exception as e:
            self.logger.error(f"Error plotting volume: {e}") 

    def add_chart_styling(self, fig, symbol):
        """Add styling to the chart"""
        try:
            # Add title
            fig.suptitle(f'{symbol} Technical Analysis & Predictions', 
                        fontsize=16, color='white', y=0.95)
            
            # Add watermark
            fig.text(0.99, 0.01, '@CryptoAnalyzer', 
                    fontsize=10, color='#9e9e9e', 
                    ha='right', va='bottom', alpha=0.5)
            
            # Add timestamp
            fig.text(0.01, 0.01, 
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                    fontsize=8, color='#9e9e9e', 
                    ha='left', va='bottom', alpha=0.5)
            
            # Adjust layout
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
        except Exception as e:
            self.logger.error(f"Error adding chart styling: {e}") 

    def plot_predictions(self, ax, current_data, ml_predictions):
        """Plot historical and predicted prices"""
        try:
            # Plot historical prices
            historical_prices = current_data['close'].iloc[-48:]
            x_hist = range(len(historical_prices))
            ax.plot(x_hist, historical_prices, 
                    label='Historical', color='#42a5f5', linewidth=2)
            
            # Plot predictions for each timeframe
            colors = {'1h': '#26a69a', '1d': '#ffca28', '1m': '#ce93d8'}
            last_price = historical_prices.iloc[-1]
            
            for timeframe, data in ml_predictions.items():
                predictions = data['predicted_prices']
                x_pred = range(len(historical_prices)-1, 
                             len(historical_prices) + len(predictions))
                
                # Plot prediction line
                ax.plot(x_pred, [last_price] + predictions,
                       label=f'{timeframe.upper()} Prediction', 
                       color=colors[timeframe], 
                       linestyle='--')
                
                # Add confidence bands
                std_dev = (data['max_change'] - data['min_change']) / 4
                upper = [p * (1 + std_dev/100) for p in predictions]
                lower = [p * (1 - std_dev/100) for p in predictions]
                
                ax.fill_between(x_pred[1:], upper, lower,
                              color=colors[timeframe], alpha=0.1)
            
            ax.set_title(f'Price Predictions', fontsize=14, color='white')
            ax.set_xlabel('Hours', fontsize=12, color='white')
            ax.set_ylabel('Price', fontsize=12, color='white')
            ax.legend(loc='upper left', facecolor='black', edgecolor='white')
            ax.grid(True, alpha=0.2)
            
        except Exception as e:
            self.logger.error(f"Error plotting predictions: {e}") 

    def save_figure(self, fig, path, **kwargs):
        """Save figure with proper font handling"""
        try:
            # Use a font that supports emoji
            plt.rcParams['font.family'] = ['Segoe UI Emoji', 'DejaVu Sans']
            
            # Save with high quality
            fig.savefig(path, 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='black', 
                       edgecolor='none',
                       **kwargs)
        except Exception as e:
            self.logger.error(f"Error saving figure: {e}") 