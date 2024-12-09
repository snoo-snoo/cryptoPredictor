import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

class OrderFlowAnalyzer:
    def __init__(self, logger, client):
        self.logger = logger
        self.client = client
        
    def analyze_order_flow(self, symbol, timeframe='1h'):
        """Analyze order flow patterns"""
        try:
            # Get order book data
            order_book = self.client.get_order_book(symbol=symbol, limit=1000)
            # Get recent trades
            trades = self.client.get_recent_trades(symbol=symbol, limit=1000)
            # Get aggregated trades
            agg_trades = self.client.get_aggregate_trades(symbol=symbol)
            
            analysis = {
                'order_book_imbalance': self.calculate_book_imbalance(order_book),
                'trade_flow': self.analyze_trade_flow(trades),
                'buy_sell_ratio': self.calculate_buy_sell_ratio(agg_trades),
                'large_orders': self.detect_large_orders(order_book),
                'price_impact': self.calculate_price_impact(order_book),
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing order flow for {symbol}: {e}")
            return None

    def calculate_book_imbalance(self, order_book):
        """Calculate order book imbalance"""
        try:
            bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity'])
            asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity'])
            
            bid_value = (bids['price'].astype(float) * bids['quantity'].astype(float)).sum()
            ask_value = (asks['price'].astype(float) * asks['quantity'].astype(float)).sum()
            
            imbalance = (bid_value - ask_value) / (bid_value + ask_value)
            
            return {
                'imbalance_ratio': float(imbalance),
                'bid_value': float(bid_value),
                'ask_value': float(ask_value),
                'interpretation': 'bullish' if imbalance > 0.1 else 'bearish' if imbalance < -0.1 else 'neutral'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating book imbalance: {e}")
            return None

    def analyze_trade_flow(self, trades):
        """Analyze recent trade flow"""
        try:
            df = pd.DataFrame(trades)
            df['price'] = pd.to_numeric(df['price'])
            df['qty'] = pd.to_numeric(df['qty'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            
            # Calculate trade flow metrics
            metrics = {
                'avg_trade_size': float(df['qty'].mean()),
                'max_trade_size': float(df['qty'].max()),
                'trade_count': len(df),
                'buy_pressure': self.calculate_buy_pressure(df),
                'recent_trend': self.detect_trade_trend(df)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing trade flow: {e}")
            return None

    def detect_large_orders(self, order_book, threshold=0.1):
        """Detect large orders in the order book"""
        try:
            bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity'])
            asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity'])
            
            total_bid_volume = bids['quantity'].astype(float).sum()
            total_ask_volume = asks['quantity'].astype(float).sum()
            
            large_bids = bids[bids['quantity'].astype(float) > total_bid_volume * threshold]
            large_asks = asks[asks['quantity'].astype(float) > total_ask_volume * threshold]
            
            return {
                'large_bids': large_bids.to_dict('records'),
                'large_asks': large_asks.to_dict('records'),
                'large_order_count': len(large_bids) + len(large_asks)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting large orders: {e}")
            return None

    def calculate_buy_pressure(self, df):
        """Calculate buying pressure from trades"""
        try:
            # Calculate buy vs sell volume
            buy_volume = df[df['isBuyerMaker'] == False]['qty'].sum()
            sell_volume = df[df['isBuyerMaker'] == True]['qty'].sum()
            
            total_volume = buy_volume + sell_volume
            if total_volume == 0:
                return 0
            
            return (buy_volume - sell_volume) / total_volume
            
        except Exception as e:
            self.logger.error(f"Error calculating buy pressure: {e}")
            return 0

    def calculate_buy_sell_ratio(self, trades):
        """Calculate buy/sell ratio from trades"""
        try:
            # Handle both list and DataFrame inputs
            if isinstance(trades, pd.DataFrame):
                buy_trades = trades[trades['isBuyerMaker'] == False]
                sell_trades = trades[trades['isBuyerMaker'] == True]
                
                buy_volume = buy_trades['qty'].sum()
                sell_volume = sell_trades['qty'].sum()
            else:
                buy_trades = [t for t in trades if not t.get('isBuyerMaker', False)]
                sell_trades = [t for t in trades if t.get('isBuyerMaker', False)]
                
                buy_volume = sum(float(t.get('qty', 0)) for t in buy_trades)
                sell_volume = sum(float(t.get('qty', 0)) for t in sell_trades)
            
            if sell_volume == 0:
                return float('inf')
            
            return buy_volume / sell_volume
            
        except Exception as e:
            self.logger.error(f"Error calculating buy/sell ratio: {e}")
            return 1.0

    def detect_trade_trend(self, df):
        """Detect recent trade trend"""
        try:
            # Calculate price changes
            df['price_change'] = df['price'].diff()
            
            # Calculate weighted price change
            recent_trades = df.iloc[-20:]  # Look at last 20 trades
            weighted_change = (recent_trades['price_change'] * 
                             recent_trades['qty']).sum() / recent_trades['qty'].sum()
            
            if weighted_change > 0:
                return 'uptrend'
            elif weighted_change < 0:
                return 'downtrend'
            return 'neutral'
            
        except Exception as e:
            self.logger.error(f"Error detecting trade trend: {e}")
            return 'neutral'

    def calculate_price_impact(self, order_book):
        """Calculate price impact of orders"""
        try:
            bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity'])
            asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity'])
            
            # Calculate impact for standard order sizes
            sizes = [1, 5, 10, 20]  # BTC or ETH
            impacts = {}
            
            for size in sizes:
                # Buy impact
                buy_impact = self.calculate_slippage(asks, size, 'buy')
                # Sell impact
                sell_impact = self.calculate_slippage(bids, size, 'sell')
                
                impacts[str(size)] = {
                    'buy_impact': buy_impact,
                    'sell_impact': sell_impact
                }
            
            return impacts
            
        except Exception as e:
            self.logger.error(f"Error calculating price impact: {e}")
            return None

    def calculate_slippage(self, orders, size, side):
        """Calculate slippage for given order size"""
        try:
            orders = orders.copy()
            orders['quantity'] = orders['quantity'].astype(float)
            orders['price'] = orders['price'].astype(float)
            
            cumulative_qty = orders['quantity'].cumsum()
            required_rows = cumulative_qty[cumulative_qty >= size].index[0] + 1
            
            weighted_price = (orders.iloc[:required_rows]['price'] * 
                            orders.iloc[:required_rows]['quantity']).sum() / \
                            orders.iloc[:required_rows]['quantity'].sum()
            
            base_price = orders.iloc[0]['price']
            slippage = (weighted_price - base_price) / base_price * 100
            
            return float(slippage if side == 'buy' else -slippage)
            
        except Exception as e:
            self.logger.error(f"Error calculating slippage: {e}")
            return 0