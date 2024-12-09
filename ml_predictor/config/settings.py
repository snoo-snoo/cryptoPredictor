import os
import json

class Settings:
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self):
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        except ImportError:
            # Fallback to default configuration if yaml is not installed
            return self.get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Default configuration if YAML file cannot be loaded"""
        return {
            'models': {
                'lstm': {
                    'base_units': 50,
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                },
                'ensemble': {
                    'models': ['lstm', 'gru', 'cnn_lstm', 'transformer'],
                    'weights': [0.4, 0.3, 0.2, 0.1]
                }
            },
            'sentiment': {
                'cache_duration': 3600,
                'sources': {
                    'cryptopanic': 0.7,
                    'lunarcrush': 0.8,
                    'newsapi': 0.6,
                    'coindesk': 0.9,
                    'cointelegraph': 0.8
                }
            },
            'data': {
                'timeframes': {
                    '1h': 24,
                    '1d': 7,
                    '1m': 30
                },
                'min_periods': {
                    '1h': 168,
                    '1d': 30,
                    '1m': 12
                }
            },
            'trading': {
                'default_pairs': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
                'timeframes': ['1h', '1d', '1m'],
                'modes': ['trade', 'backtest', 'analyze']
            }
        }
            
    @property
    def model_settings(self):
        return self.config.get('models', {})
        
    @property
    def sentiment_settings(self):
        return self.config.get('sentiment', {})
        
    @property
    def data_settings(self):
        return self.config.get('data', {})
        
    @property
    def trading_settings(self):
        return self.config.get('trading', {}) 