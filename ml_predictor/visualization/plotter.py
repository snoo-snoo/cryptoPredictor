import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import json
import ta
import pandas as pd
from scipy import stats
from scipy import signal
from openai import OpenAI

class PredictionPlotter:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize OpenAI client
        try:
            import config
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI client: {e}")
            self.openai_client = None
        
        # Create analysis directory structure
        self.base_dir = 'analysis'
        self.subdirs = ['charts', 'predictions', 'performance', 'metrics']
        self.create_directories()
        
        # Set style using built-in matplotlib style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except Exception as e:
            self.logger.warning(f"Could not set plot style: {e}")
            plt.style.use('default')
        
        # Set color palette
        try:
            sns.set_palette("husl")
        except Exception as e:
            self.logger.warning(f"Could not set color palette: {e}")

    def create_directories(self):
        """Create hourly directory structure"""
        try:
            # Create base analysis directory
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
            
            # Create hourly directory
            self.timestamp = datetime.now().strftime("%Y%m%d_%H")
            self.current_dir = os.path.join(self.base_dir, self.timestamp)
            os.makedirs(self.current_dir, exist_ok=True)
            
            # Create subdirectories in hourly directory
            for subdir in ['charts', 'predictions', 'performance', 'metrics', 'blog']:
                path = os.path.join(self.current_dir, subdir)
                os.makedirs(path, exist_ok=True)
                
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")

    def plot_predictions(self, df, predictions, symbol, timeframe):
        """Create interactive plots with predictions for different timeframes"""
        try:
            plot_paths = {}
            
            # Create timestamp directory based on hour
            timestamp = datetime.now().strftime("%Y%m%d_%H")  # Only hour precision
            plot_dir = os.path.join(self.base_dir, timestamp)
            os.makedirs(plot_dir, exist_ok=True)
            
            # Create technical analysis plot
            tech_paths = self.create_technical_plot(
                df.tail(100),  # Last 100 candles for technical analysis
                symbol, 
                timeframe
            )
            if tech_paths:
                plot_paths['technical'] = tech_paths
            
            # Create prediction plots for each timeframe
            for tf, pred_data in predictions.items():
                pred_paths = self.create_prediction_plot(
                    df.tail(24),  # Last 24 candles for prediction plots
                    pred_data,
                    symbol,
                    tf
                )
                if pred_paths:
                    plot_paths[f'prediction_{tf}'] = pred_paths
            
            # Generate blog post
            blog_path = self.create_analysis_blog(
                df=df,
                predictions=predictions,
                symbol=symbol,
                timeframe=timeframe,
                plot_paths=plot_paths
            )
            if blog_path:
                plot_paths['blog'] = blog_path
            
            # Add directory info
            plot_paths['timestamp'] = timestamp
            plot_paths['directory'] = plot_dir
            
            # Save analysis summary
            summary = {
                'symbol': symbol,
                'timestamp': timestamp,
                'predictions': predictions,
                'plots': plot_paths
            }
            summary_path = self.save_analysis_summary(symbol, summary)
            if summary_path:
                plot_paths['summary'] = summary_path
            
            return plot_paths
            
        except Exception as e:
            self.logger.error(f"Error creating plots: {e}")
            return {
                'timestamp': datetime.now().strftime("%Y%m%d_%H"),
                'directory': self.current_dir,
                'error': str(e)
            }

    def create_technical_plot(self, df, symbol, timeframe):
        """Create technical analysis plot with enhanced technical indicators"""
        try:
            # Create figure with subplots
            fig = make_subplots(rows=6, cols=1,  # Increased to 6 rows for more indicators
                              shared_xaxes=True,
                              vertical_spacing=0.03,
                              row_heights=[0.4, 0.15, 0.15, 0.1, 0.1, 0.1],
                              subplot_titles=(
                                  f'{symbol} Price and Indicators',
                                  'RSI + Stochastic',
                                  'MACD',
                                  'ADX',
                                  'OBV',
                                  'Volume'
                              ))

            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='OHLC'
                ),
                row=1, col=1
            )

            # Add Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            fig.add_trace(
                go.Scatter(x=df.index, y=bb.bollinger_hband(), 
                          name='BB Upper', line=dict(color='rgba(250,250,250,0.5)', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=bb.bollinger_lband(), 
                          name='BB Lower', line=dict(color='rgba(250,250,250,0.5)', 
                          dash='dash'), fill='tonexty'),
                row=1, col=1
            )

            # Add Moving Averages (EMA and SMA)
            for period in [20, 50, 200]:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['close'].ewm(span=period).mean(),
                        name=f'EMA {period}',
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['close'].rolling(window=period).mean(),
                        name=f'SMA {period}',
                        line=dict(width=1, dash='dot')
                    ),
                    row=1, col=1
                )

            # Add Ichimoku Cloud
            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
            fig.add_trace(
                go.Scatter(x=df.index, y=ichimoku.ichimoku_conversion_line(), 
                          name='Conversion Line', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=ichimoku.ichimoku_base_line(), 
                          name='Base Line', line=dict(color='red')),
                row=1, col=1
            )

            # Add RSI and Stochastic
            rsi = ta.momentum.RSIIndicator(df['close']).rsi()
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            
            fig.add_trace(
                go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=stoch.stoch(), name='Stoch %K', line=dict(color='blue')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=stoch.stoch_signal(), name='Stoch %D', line=dict(color='red')),
                row=2, col=1
            )
            
            # Add overbought/oversold lines
            for level in [20, 30, 70, 80]:
                fig.add_hline(y=level, line_dash="dash", 
                             line_color="gray", row=2, col=1, opacity=0.5)

            # Add MACD with histogram
            macd = ta.trend.MACD(df['close'])
            macd_diff = macd.macd_diff()
            colors = ['red' if val < 0 else 'green' for val in macd_diff]
            
            fig.add_trace(
                go.Scatter(x=df.index, y=macd.macd(), name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=macd.macd_signal(), name='Signal', line=dict(color='orange')),
                row=3, col=1
            )
            fig.add_trace(
                go.Bar(x=df.index, y=macd_diff, name='MACD Hist', marker_color=colors),
                row=3, col=1
            )

            # Add ADX
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            fig.add_trace(
                go.Scatter(x=df.index, y=adx.adx(), name='ADX', line=dict(color='yellow')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=adx.adx_pos(), name='+DI', line=dict(color='green')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=adx.adx_neg(), name='-DI', line=dict(color='red')),
                row=4, col=1
            )

            # Add OBV (On Balance Volume)
            obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
            fig.add_trace(
                go.Scatter(x=df.index, y=obv.on_balance_volume(), name='OBV', 
                          line=dict(color='cyan')),
                row=5, col=1
            )

            # Add volume with price direction coloring
            colors = ['red' if row['open'] > row['close'] else 'green' 
                     for i, row in df.iterrows()]
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
                row=6, col=1
            )
            
            # Add volume MA
            fig.add_trace(
                go.Scatter(x=df.index, y=df['volume'].rolling(20).mean(), 
                          name='Volume MA', line=dict(color='yellow')),
                row=6, col=1
            )

            # Add trend lines
            self.add_trend_lines(df, fig)
            
            # Add support and resistance levels
            self.add_support_resistance(df, fig)
            
            # Add pivot points
            self.add_pivot_points(df, fig)
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Technical Analysis ({timeframe})',
                yaxis_title='Price',
                yaxis2_title='RSI',
                yaxis3_title='MACD',
                yaxis4_title='Volume',
                xaxis_rangeslider_visible=False,
                height=1200,
                template='plotly_dark',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                margin=dict(r=150)
            )

            # Update y-axes
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="Volume", row=4, col=1)

            # Save plots
            filename_base = f"{symbol}_{timeframe}_technical"
            html_path = os.path.join(self.current_dir, 'charts', f"{filename_base}.html")
            png_path = os.path.join(self.current_dir, 'charts', f"{filename_base}.png")
            
            fig.write_html(html_path)
            fig.write_image(png_path, height=1200, width=1600)
            
            return {
                'technical_html': html_path,
                'technical_png': png_path
            }
            
        except Exception as e:
            self.logger.error(f"Error creating technical plot: {e}")
            return None

    def create_prediction_plot(self, df, pred_data, symbol, timeframe):
        """Create focused prediction plot with essential indicators"""
        try:
            # Create main figure with 3 rows
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(
                    f'{symbol} Price Analysis ({timeframe})',
                    'Volume',
                    'RSI'
                )
            )

            # Calculate date ranges
            if timeframe == '1h':
                lookback = 48  # 2 days
                forward = 24   # 1 day
            elif timeframe == '1d':
                lookback = 30  # 30 days
                forward = 7    # 1 week
            else:  # '1m'
                lookback = 12  # 1 year
                forward = 3    # 3 months

            # Get relevant data slice
            df_slice = df.tail(lookback).copy()
            current_price = float(df_slice['close'].iloc[-1])

            # 1. Main price chart with candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=df_slice.index,
                    open=df_slice['open'],
                    high=df_slice['high'],
                    low=df_slice['low'],
                    close=df_slice['close'],
                    name='Price'
                ),
                row=1, col=1
            )

            # 2. Add EMAs
            for period in [20, 50]:
                ema = df_slice['close'].ewm(span=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df_slice.index,
                        y=ema,
                        name=f'EMA {period}',
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )

            # 3. Add prediction
            if pred_data and 'predicted_prices' in pred_data:
                # Calculate future dates
                future_dates = pd.date_range(
                    start=df_slice.index[-1],
                    periods=forward + 1,
                    freq=timeframe.replace('1', '')
                )
                
                # Prepare prediction data
                predictions = [current_price] + [float(p) for p in pred_data['predicted_prices'][:forward]]
                
                # Add prediction line
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=predictions,
                        name='Prediction',
                        line=dict(color='green', width=2, dash='dash')
                    ),
                    row=1, col=1
                )

                # Add confidence bands if available
                if 'confidence' in pred_data:
                    confidence = float(pred_data['confidence'])
                    upper = [p * (1 + confidence) for p in predictions]
                    lower = [p * (1 - confidence) for p in predictions]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=upper,
                            name='Upper Band',
                            line=dict(width=0),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=lower,
                            name='Lower Band',
                            fill='tonexty',
                            fillcolor='rgba(0,255,0,0.1)',
                            line=dict(width=0),
                            showlegend=False
                        ),
                        row=1, col=1
                    )

            # 4. Add volume bars
            colors = ['red' if o > c else 'green' 
                     for o, c in zip(df_slice['open'], df_slice['close'])]
            fig.add_trace(
                go.Bar(
                    x=df_slice.index,
                    y=df_slice['volume'],
                    marker_color=colors,
                    name='Volume'
                ),
                row=2, col=1
            )

            # 5. Add RSI
            rsi = ta.momentum.RSIIndicator(df_slice['close']).rsi()
            fig.add_trace(
                go.Scatter(x=df_slice.index, y=rsi, name='RSI'),
                row=3, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # Update layout
            fig.update_layout(
                height=900,
                width=1200,
                template='plotly_dark',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            # Update axes
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)

            # Save plot
            png_path = os.path.join(
                self.current_dir, 
                'predictions', 
                f"{symbol}_{timeframe}_prediction.png"
            )
            fig.write_image(png_path)

            return {'png': png_path}

        except Exception as e:
            self.logger.error(f"Error creating prediction plot: {e}")
            return None

    def detect_candlestick_patterns(self, df):
        """Detect candlestick patterns"""
        patterns = []
        
        # Example patterns (add more as needed)
        for i in range(2, len(df)):
            # Bullish engulfing
            if (df['close'].iloc[i-1] < df['open'].iloc[i-1] and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i] > df['open'].iloc[i-1] and
                df['open'].iloc[i] < df['close'].iloc[i-1]):
                patterns.append({
                    'date': df.index[i],
                    'price': df['high'].iloc[i],
                    'type': 'bullish',
                    'name': 'Bullish Engulfing'
                })
            
            # Bearish engulfing
            if (df['close'].iloc[i-1] > df['open'].iloc[i-1] and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i] < df['open'].iloc[i-1] and
                df['open'].iloc[i] > df['close'].iloc[i-1]):
                patterns.append({
                    'date': df.index[i],
                    'price': df['low'].iloc[i],
                    'type': 'bearish',
                    'name': 'Bearish Engulfing'
                })
        
        return patterns

    def save_static_plot(self, df, predictions, symbol, timeframe, filepath):
        """Save static plot using matplotlib as fallback"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot OHLC
            plt.subplot(2, 1, 1)
            plt.plot(df.index, df['close'], label='Close Price')
            
            # Add predictions
            if predictions and timeframe in predictions:
                pred_data = predictions[timeframe]
                future_dates = [df.index[-1] + timedelta(hours=i) 
                              for i in range(1, len(pred_data['predicted_prices'])+1)]
                plt.plot(future_dates, pred_data['predicted_prices'], 
                        '--', label='Prediction')
                
                # Add confidence intervals
                if 'confidence' in pred_data:
                    confidence = pred_data['confidence']
                    upper_bound = [p * (1 + confidence) for p in pred_data['predicted_prices']]
                    lower_bound = [p * (1 - confidence) for p in pred_data['predicted_prices']]
                    plt.fill_between(future_dates, upper_bound, lower_bound, 
                                   alpha=0.2, label='Confidence')
            
            plt.title(f'{symbol} Price Prediction ({timeframe})')
            plt.ylabel('Price')
            plt.legend()
            
            # Plot volume
            plt.subplot(2, 1, 2)
            plt.bar(df.index, df['volume'], alpha=0.7, label='Volume')
            plt.ylabel('Volume')
            
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close()
            
            self.logger.info(f"Saved static plot using matplotlib to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving static plot with matplotlib: {e}")
            raise e

    def plot_model_performance(self, model_history, symbol, timeframe):
        """Plot model training history"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot training & validation loss
            ax1.plot(model_history['loss'], label='Training Loss')
            ax1.plot(model_history['val_loss'], label='Validation Loss')
            ax1.set_title(f'Model Loss ({symbol} - {timeframe})')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Plot loss ratio
            loss_ratio = np.array(model_history['val_loss']) / np.array(model_history['loss'])
            ax2.plot(loss_ratio, label='Validation/Training Loss Ratio')
            ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            ax2.set_title('Loss Ratio')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Ratio')
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plot
            filename = f"{symbol}_{timeframe}_performance_{self.timestamp}.png"
            filepath = os.path.join(self.current_dir, filename)
            plt.savefig(filepath)
            plt.close()
            
            return filepath

        except Exception as e:
            self.logger.error(f"Error creating performance plot: {e}")
            return None

    def save_analysis_summary(self, symbol: str, data: dict):
        """Save analysis summary as JSON"""
        try:
            # Convert numpy/pandas types to Python native types
            def convert_to_native(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, dict):
                    return {key: convert_to_native(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                return obj

            # Convert data to native Python types
            native_data = convert_to_native(data)
            
            # Save to file
            filename = f"{symbol}_analysis_{self.timestamp}.json"
            filepath = os.path.join(self.current_dir, 'metrics', filename)
            
            with open(filepath, 'w') as f:
                json.dump(native_data, f, indent=2)
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving analysis summary: {e}")
            return None

    def create_analysis_blog(self, df, predictions, symbol, timeframe, plot_paths):
        """Create detailed markdown blog post of the analysis using ChatGPT"""
        try:
            if not self.openai_client:
                self.logger.error("OpenAI client not initialized")
                return None
            
            blog_dir = os.path.join(self.current_dir, 'blog')
            os.makedirs(blog_dir, exist_ok=True)
            
            # Get technical plot path
            tech_plot = plot_paths.get('technical', {}).get('technical_png', '')
            if tech_plot:
                tech_plot = os.path.basename(tech_plot)
            
            # Convert all numeric values to native Python types
            market_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': float(df['close'].iloc[-1]),
                'volume_24h': float(df['volume'].tail(24).sum()),
                'price_change_24h': float((df['close'].iloc[-1] / df['close'].iloc[-25] - 1) * 100),
                'rsi': float(ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1]),
                'macd': float(ta.trend.MACD(df['close']).macd().iloc[-1]),
                'macd_signal': float(ta.trend.MACD(df['close']).macd_signal().iloc[-1])
            }
            
            # Convert predictions separately
            converted_predictions = self.convert_predictions_to_native(predictions)
            
            # Create prompt for ChatGPT
            prompt = f"""
            Create a detailed cryptocurrency market analysis blog post with the following data:

            Symbol: {market_data['symbol']}
            Timeframe: {market_data['timeframe']}
            Current Price: ${market_data['current_price']:.8f}
            24h Volume: {market_data['volume_24h']:.2f}
            24h Change: {market_data['price_change_24h']:.2f}%
            RSI: {market_data['rsi']:.2f}
            MACD: {market_data['macd']:.2f}
            MACD Signal: {market_data['macd_signal']:.2f}

            Predictions:
            {json.dumps(converted_predictions, indent=2)}

            Requirements:
            1. Write in a professional but engaging style
            2. Include technical analysis interpretation
            3. Explain the predictions and their implications
            4. Add market sentiment analysis
            5. Include a risk disclaimer
            6. Format in Markdown
            7. Keep it concise but informative
            """
            
            # Get response from ChatGPT
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=1000,
                temperature=0.7
            )
            
            # Get blog content and add image
            blog_content = response.choices[0].message.content.strip()
            image_section = f"\n\n## Technical Analysis\n![Technical Analysis]({tech_plot})\n\n"
            blog_content = blog_content.replace("## Technical Analysis", image_section)
            
            # Save blog post
            blog_path = os.path.join(blog_dir, f"{symbol}_{timeframe}_analysis.md")
            with open(blog_path, 'w', encoding='utf-8') as f:
                f.write(blog_content)
            
            self.logger.info(f"Created analysis blog post: {blog_path}")
            return blog_path

        except Exception as e:
            self.logger.error(f"Error creating analysis blog: {e}")
            self.logger.error(f"Error details: {str(e)}")
            return None

    def add_trend_lines(self, df, fig):
        """Add trend lines to the chart"""
        try:
            # Calculate highs and lows
            highs = df[df['high'] == df['high'].rolling(10).max()]['high']
            lows = df[df['low'] == df['low'].rolling(10).min()]['low']
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Upper trend line
                z_high = np.polyfit(range(len(highs)), highs.values, 1)
                p_high = np.poly1d(z_high)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=p_high(range(len(df))),
                        name='Upper Trend',
                        line=dict(color='rgba(255,255,255,0.5)', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                
                # Lower trend line
                z_low = np.polyfit(range(len(lows)), lows.values, 1)
                p_low = np.poly1d(z_low)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=p_low(range(len(df))),
                        name='Lower Trend',
                        line=dict(color='rgba(255,255,255,0.5)', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
        except Exception as e:
            self.logger.error(f"Error adding trend lines: {e}")

    def add_support_resistance(self, df, fig):
        """Add support and resistance levels"""
        try:
            # Calculate potential levels using price clusters
            price_clusters = pd.concat([df['high'], df['low']])
            kde = stats.gaussian_kde(price_clusters)
            x_range = np.linspace(min(price_clusters), max(price_clusters), 100)
            density = kde(x_range)
            
            # Find local maxima in density (price levels with high activity)
            peaks = signal.find_peaks(density, distance=10)[0]
            levels = x_range[peaks]
            
            # Add levels to chart
            for level in levels[:6]:  # Show top 6 levels
                fig.add_hline(
                    y=float(level),  # Convert to float to avoid numpy type issues
                    line_dash="dot",
                    line_color="rgba(255,255,255,0.3)",
                    row=1, col=1,
                    annotation=dict(
                        text=f"S/R {float(level):.2f}",  # Convert to float
                        xref="paper",
                        x=1.02,
                        showarrow=False
                    )
                )
        except Exception as e:
            self.logger.error(f"Error adding support/resistance: {e}")

    def add_pivot_points(self, df, fig):
        """Add pivot points"""
        try:
            # Calculate classic pivot points
            high = float(df['high'].iloc[-1])  # Convert to float
            low = float(df['low'].iloc[-1])    # Convert to float
            close = float(df['close'].iloc[-1]) # Convert to float
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            # Add pivot lines
            pivots = {
                'R3': r3, 'R2': r2, 'R1': r1,
                'P': pivot,
                'S1': s1, 'S2': s2, 'S3': s3
            }
            
            colors = {
                'R3': 'red', 'R2': 'orange', 'R1': 'yellow',
                'P': 'white',
                'S1': 'yellow', 'S2': 'orange', 'S3': 'red'
            }
            
            for name, level in pivots.items():
                fig.add_hline(
                    y=float(level),  # Convert to float
                    line_dash="dot",
                    line_color=colors[name],
                    line_width=1,
                    opacity=0.3,
                    row=1, col=1,
                    annotation=dict(
                        text=f"{name} {float(level):.2f}",  # Convert to float
                        xref="paper",
                        x=1.02,
                        showarrow=False,
                        font=dict(color=colors[name])
                    )
                )
        except Exception as e:
            self.logger.error(f"Error adding pivot points: {e}")

    def convert_predictions_to_native(self, predictions):
        """Convert prediction values to native Python types"""
        try:
            def convert_value(value):
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    return int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                    return float(value)
                elif isinstance(value, np.ndarray):
                    return [convert_value(v) for v in value.tolist()]
                elif isinstance(value, list):
                    return [convert_value(v) for v in value]
                elif isinstance(value, dict):
                    return {k: convert_value(v) for k, v in value.items()}
                elif isinstance(value, pd.Series):
                    return convert_value(value.tolist())
                elif isinstance(value, pd.DataFrame):
                    return {col: convert_value(values.tolist()) 
                           for col, values in value.items()}
                return value

            # First convert predictions to native types
            converted = {}
            for timeframe, data in predictions.items():
                converted[timeframe] = {
                    key: convert_value(value)
                    for key, value in data.items()
                }

            # Verify conversion by testing JSON serialization
            try:
                json.dumps(converted)  # Test if serializable
                return converted
            except TypeError as e:
                self.logger.error(f"Conversion verification failed: {e}")
                # If verification fails, do a more aggressive conversion
                return json.loads(json.dumps(converted, default=str))
            
        except Exception as e:
            self.logger.error(f"Error converting predictions: {e}")
            # Last resort: convert everything to strings
            return {k: str(v) for k, v in predictions.items()}