
# ==========================================
# src/data/data_processor.py
# ==========================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

class DataProcessor:
    """Data cleaning and preprocessing for trading data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean raw stock data"""
        try:
            # Remove duplicates
            data = data.drop_duplicates()
            
            # Handle missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Remove outliers (prices that changed more than 20% in a day)
            data['price_change'] = data['Close'].pct_change()
            data = data[abs(data['price_change']) < 0.20]
            
            # Ensure proper data types
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            if 'Volume' in data.columns:
                data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
            
            self.logger.info(f"Data cleaned: {len(data)} records")
            return data.drop('price_change', axis=1)
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            return data
    
    def add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for ML"""
        try:
            # Price features
            data['price_change'] = data['Close'].pct_change()
            data['high_low_ratio'] = data['High'] / data['Low']
            data['open_close_ratio'] = data['Open'] / data['Close']
            
            # Volume features
            if 'Volume' in data.columns:
                data['volume_change'] = data['Volume'].pct_change()
                data['volume_price_trend'] = data['Volume'] * data['price_change']
            
            # Volatility features
            data['volatility_5d'] = data['price_change'].rolling(5).std()
            data['volatility_20d'] = data['price_change'].rolling(20).std()
            
            # Gap features
            data['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
            
            self.logger.info("Derived features added")
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding derived features: {str(e)}")
            return data
    
    def resample_data(self, data: pd.DataFrame, timeframe: str = '1D') -> pd.DataFrame:
        """Resample data to different timeframes"""
        try:
            if 'Date' in data.columns:
                data.set_index('Date', inplace=True)
            
            resampled = data.resample(timeframe).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            return resampled.reset_index()
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {str(e)}")
            return data