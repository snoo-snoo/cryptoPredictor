from typing import Dict
from pydantic import BaseModel, validator

class ModelConfig(BaseModel):
    base_units: int
    dropout_rate: float
    learning_rate: float
    batch_size: int
    epochs: int

    @validator('dropout_rate')
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Dropout rate must be between 0 and 1')
        return v

class ConfigValidator:
    def __init__(self):
        self.required_keys = {
            'models': ['lstm', 'ensemble'],
            'sentiment': ['cache_duration', 'sources'],
            'data': ['timeframes', 'min_periods']
        }

    def validate_config(self, config: Dict) -> bool:
        try:
            # Validate structure
            for section, keys in self.required_keys.items():
                if section not in config:
                    raise ValueError(f"Missing section: {section}")
                for key in keys:
                    if key not in config[section]:
                        raise ValueError(f"Missing key: {key} in section {section}")

            # Validate model config
            ModelConfig(**config['models']['lstm'])

            return True
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")

class ConfigurationError(Exception):
    pass 