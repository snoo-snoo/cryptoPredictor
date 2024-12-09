from datetime import datetime  # Add at the top

class OptionsAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        
    def analyze_options_flow(self, symbol):
        """Analyze options market data"""
        try:
            # Get options data from various sources
            deribit_data = self.get_deribit_options(symbol)
            binance_data = self.get_binance_options(symbol)
            
            analysis = {
                'put_call_ratio': self.calculate_put_call_ratio(deribit_data, binance_data),
                'implied_volatility': self.calculate_iv_surface(deribit_data),
                'options_volume': self.analyze_options_volume(deribit_data, binance_data),
                'open_interest': self.analyze_open_interest(deribit_data),
                'gamma_exposure': self.calculate_gamma_exposure(deribit_data),
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing options flow: {e}")
            return None

    def calculate_put_call_ratio(self, *data_sources):
        """Calculate put/call ratio"""
        try:
            total_call_volume = 0
            total_put_volume = 0
            
            for data in data_sources:
                if isinstance(data, dict) and data:
                    if data.get('type') == 'call':
                        total_call_volume += float(data.get('volume', 0))
                    elif data.get('type') == 'put':
                        total_put_volume += float(data.get('volume', 0))
            
            ratio = total_put_volume / total_call_volume if total_call_volume > 0 else float('inf')
            
            return {
                'ratio': ratio,
                'call_volume': total_call_volume,
                'put_volume': total_put_volume,
                'sentiment': 'bearish' if ratio > 1 else 'bullish'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating put/call ratio: {e}")
            return None

    def calculate_iv_surface(self, options_data):
        """Calculate implied volatility surface"""
        try:
            if not isinstance(options_data, dict):
                return None
            
            iv_data = []
            if 'price' in options_data and 'strike' in options_data:
                # Single option case
                iv = self.calculate_implied_volatility(
                    float(options_data['price']),
                    float(options_data['strike']),
                    options_data.get('time_to_expiry', 30),
                    options_data.get('type', 'call')
                )
                iv_data.append({
                    'strike': float(options_data['strike']),
                    'expiry': options_data.get('expiry', ''),
                    'iv': iv,
                    'type': options_data.get('type', 'call')
                })
            
            return iv_data
            
        except Exception as e:
            self.logger.error(f"Error calculating IV surface: {e}")
            return None

    def get_deribit_options(self, symbol):
        """Get options data from Deribit"""
        try:
            # Example implementation - replace with actual API call
            return {
                'type': 'call',
                'strike': 100,
                'expiry': '2024-12-31',
                'price': 1.0,
                'volume': 100
            }
        except Exception as e:
            self.logger.error(f"Error getting Deribit options: {e}")
            return None

    def get_binance_options(self, symbol):
        """Get options data from Binance"""
        try:
            # Example implementation - replace with actual API call
            return {
                'type': 'put',
                'strike': 90,
                'expiry': '2024-12-31',
                'price': 0.5,
                'volume': 50
            }
        except Exception as e:
            self.logger.error(f"Error getting Binance options: {e}")
            return None 

    def calculate_implied_volatility(self, price, strike, time_to_expiry, option_type):
        """Calculate implied volatility using Black-Scholes"""
        try:
            # Simplified IV calculation
            return 0.5  # Default to 50% volatility
        except Exception as e:
            self.logger.error(f"Error calculating implied volatility: {e}")
            return None

    def analyze_options_volume(self, *data_sources):
        """Analyze options trading volume"""
        try:
            volume_data = {
                'total_volume': 0,
                'call_volume': 0,
                'put_volume': 0,
                'volume_trend': 'neutral'
            }
            
            for data in data_sources:
                if data:
                    volume_data['total_volume'] += data.get('volume', 0)
                    if data.get('type') == 'call':
                        volume_data['call_volume'] += data.get('volume', 0)
                    else:
                        volume_data['put_volume'] += data.get('volume', 0)
            
            return volume_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing options volume: {e}")
            return None

    def analyze_open_interest(self, options_data):
        """Analyze open interest"""
        try:
            return {
                'total_oi': 1000,  # Example values
                'call_oi': 600,
                'put_oi': 400,
                'oi_trend': 'increasing'
            }
        except Exception as e:
            self.logger.error(f"Error analyzing open interest: {e}")
            return None

    def calculate_gamma_exposure(self, options_data):
        """Calculate gamma exposure"""
        try:
            return {
                'total_gamma': 100,  # Example values
                'gamma_level': 'moderate',
                'risk_level': 'medium'
            }
        except Exception as e:
            self.logger.error(f"Error calculating gamma exposure: {e}")
            return None