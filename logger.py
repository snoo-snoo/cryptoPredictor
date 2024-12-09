import logging
from datetime import datetime
import os
import sys

def setup_logger():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Set up logging
    logger = logging.getLogger('trading_bot')
    logger.setLevel(logging.INFO)
    
    # Create file handler with UTF-8 encoding
    fh = logging.FileHandler(
        f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log',
        encoding='utf-8'
    )
    fh.setLevel(logging.INFO)
    
    # Create console handler with UTF-8 encoding
    ch = logging.StreamHandler(sys.stdout)  # Use stdout instead of stderr
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger 