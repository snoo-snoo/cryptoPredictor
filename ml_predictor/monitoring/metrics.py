from typing import Dict, List
import numpy as np
from datetime import datetime

class PerformanceMonitor:
    def __init__(self, logger):
        self.logger = logger
        self.metrics_history = {}
    
    def track_prediction(self, symbol: str, predictions: Dict, actual_values: List[float]):
        """Track prediction performance"""
        try:
            metrics = self.calculate_metrics(predictions, actual_values)
            
            if symbol not in self.metrics_history:
                self.metrics_history[symbol] = []
                
            self.metrics_history[symbol].append({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error tracking prediction: {e}")
            return None
    
    def calculate_metrics(self, predictions: Dict, actual_values: List[float]) -> Dict:
        """Calculate performance metrics"""
        try:
            metrics = {}
            for timeframe, pred in predictions.items():
                metrics[timeframe] = {
                    'mse': np.mean((np.array(pred['predicted_prices']) - np.array(actual_values)) ** 2),
                    'direction_accuracy': self.calculate_direction_accuracy(pred['predicted_prices'], actual_values),
                    'profit_factor': self.calculate_profit_factor(pred['predicted_prices'], actual_values)
                }
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {} 