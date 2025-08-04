"""
Configuration management for the trading system
"""

import os
import yaml
from typing import Any, Dict
from dotenv import load_dotenv

class Config:
    """Configuration manager for the trading system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._config = {}
        self.load_config()
        self.load_env_vars()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            # Handle both relative and absolute paths
            if not os.path.isabs(self.config_path):
                # Get the project root directory
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                config_file = os.path.join(project_root, self.config_path)
            else:
                config_file = self.config_path
            
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
            else:
                # Fallback default configuration
                self._config = self._get_default_config()
                print(f"Config file not found at {config_file}, using defaults")
                
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            self._config = self._get_default_config()
    
    def load_env_vars(self):
        """Load environment variables"""
        # Try to load from .env file
        env_files = ['config/api_keys.env', '.env', 'api_keys.env']
        for env_file in env_files:
            if os.path.exists(env_file):
                load_dotenv(env_file)
                break
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'api': {
                'provider': 'yfinance',
                'rate_limit': 5,
                'timeout': 30
            },
            'stocks': {
                'symbols': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS'],
                'market_open': '09:15',
                'market_close': '15:30'
            },
            'strategy': {
                'name': 'RSI_MA_Strategy',
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ma_short': 20,
                'ma_long': 50,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            },
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'max_daily_loss': 0.05
            },
            'ml': {
                'model_type': 'decision_tree',
                'features': ['rsi', 'ma_20', 'ma_50', 'macd', 'volume_sma'],
                'train_test_split': 0.8,
                'lookback_days': 30,
                'retrain_frequency': 7
            },
            'sheets': {
                'url': 'https://docs.google.com/spreadsheets/d/1XzTBEYUoFrmCD8DRkJZ0fSF7Ut1wyCeLx6HFCORgZME/edit',
                'credentials_path': 'credentials.json',
                'trade_log_realtime': True,
                'portfolio_summary_daily': True,
                'performance_metrics_weekly': True
            },
            'logging': {
                'level': 'INFO',
                'file_path': 'logs/trading.log',
                'max_file_size': '10MB',
                'backup_count': 5,
                'console_output': True
            },
            'execution': {
                'mode': 'paper',
                'initial_capital': 100000,
                'check_interval_minutes': 60,
                'market_data_refresh_minutes': 5
            },
            'backtesting': {
                'start_date': '2023-06-01',
                'end_date': '2024-01-01',
                'initial_capital': 100000,
                'commission': 0.001
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable"""
        return os.getenv(key, default)
    
    def save_config(self, path: str = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)