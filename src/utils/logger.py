"""
Logging configuration for the trading system
"""

import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(name: str = "trading_system", 
                level: int = logging.INFO,
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "trading_system") -> logging.Logger:
    """Get existing logger or create new one"""
    return logging.getLogger(name)

# Setup default logger
default_log_file = f"logs/trading_system_{datetime.now().strftime('%Y%m%d')}.log"
setup_logger("trading_system", logging.INFO, default_log_file)