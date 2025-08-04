"""
Market data fetcher using yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import Optional

class DataFetcher:
    """Fetches market data using yfinance"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.rate_limit = config.get('api.rate_limit', 5)
        self.timeout = config.get('api.timeout', 30)
        self.last_request_time = 0
    
    def _rate_limit_wait(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60 / self.rate_limit  # seconds between requests
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def fetch_daily_data(self, symbol: str, days: int = 200) -> Optional[pd.DataFrame]:
        """
        Fetch daily stock data
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self._rate_limit_wait()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            self.logger.info(f"Fetching {days} days of data for {symbol}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch data
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if data.empty:
                self.logger.warning(f"No data available for {symbol}")
                return None
            
            # Clean and standardize data
            data = self._clean_data(data, symbol)
            
            self.logger.info(f"Successfully fetched {len(data)} days of data for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_intraday_data(self, symbol: str, period: str = "1d", interval: str = "5m") -> Optional[pd.DataFrame]:
        """
        Fetch intraday stock data
        
        Args:
            symbol: Stock symbol
            period: Time period ('1d', '5d', '1mo', etc.)
            interval: Data interval ('1m', '5m', '15m', '1h')
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        try:
            self._rate_limit_wait()
            
            self.logger.info(f"Fetching {period} intraday data for {symbol} at {interval} intervals")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No intraday data available for {symbol}")
                return None
            
            data = self._clean_data(data, symbol)
            
            self.logger.info(f"Successfully fetched {len(data)} intraday data points for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None
    
    def fetch_multiple_symbols(self, symbols: list, days: int = 200) -> dict:
        """
        Fetch data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            days: Number of days of historical data
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_daily_data(symbol, days)
                if data is not None and not data.empty:
                    data_dict[symbol] = data
                else:
                    self.logger.warning(f"Skipping {symbol} - no data available")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
        
        self.logger.info(f"Successfully fetched data for {len(data_dict)} out of {len(symbols)} symbols")
        return data_dict
    
    def _clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and standardize the fetched data
        
        Args:
            data: Raw data from yfinance
            symbol: Stock symbol for logging
            
        Returns:
            Cleaned DataFrame
        """
        try:
            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            # Remove any rows with NaN values
            initial_len = len(data)
            data = data.dropna()
            
            if len(data) < initial_len:
                self.logger.info(f"Removed {initial_len - len(data)} rows with NaN values for {symbol}")
            
            # Sort by date (index)
            data = data.sort_index()
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns for {symbol}: {missing_columns}")
                return pd.DataFrame()
            
            # Round numerical values
            numeric_columns = ['open', 'high', 'low', 'close']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = data[col].round(2)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error cleaning data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest close price or None if error
        """
        try:
            data = self.fetch_daily_data(symbol, days=1)
            if data is not None and not data.empty:
                return float(data['close'].iloc[-1])
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and has data
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            data = self.fetch_daily_data(symbol, days=5)
            return data is not None and not data.empty
        except Exception as e:
            self.logger.error(f"Error validating symbol {symbol}: {e}")
            return False