import pandas as pd
import numpy as np
from src.data.indicators import TechnicalIndicators
import logging

logger = logging.getLogger("FeatureEngineer")

class FeatureEngineer:
    """Robust feature engineering with enhanced validation"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.indicators = TechnicalIndicators()
        self.required_columns = ['close', 'volume']  # Minimum required
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features with enhanced validation and error handling"""
        if data is None or data.empty:
            logger.warning("Data is empty or None")
            return pd.DataFrame()
        
        # Check for required columns
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        try:
            df = data.copy()
            close = df['close']
            volume = df['volume']
            
            # Calculate technical indicators
            df['RSI'] = self.indicators.calculate_rsi(close)
            df['MA_20'] = self.indicators.calculate_moving_average(close, 20)
            df['MA_50'] = self.indicators.calculate_moving_average(close, 50)
            
            # MACD
            macd, signal = self.indicators.calculate_macd(close)
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Hist'] = macd - signal
            
            # Volume change
            df['Volume_Change'] = volume.pct_change()
            
            # Price change
            df['Price_Change'] = close.pct_change()
            
            # Volatility
            df['Volatility'] = df['Price_Change'].rolling(10).std()
            
            # Lagged features
            df['Close_Lag1'] = close.shift(1)
            df['Close_Lag2'] = close.shift(2)
            
            # Target: next day's price movement (1 if up, 0 if down)
            df['Target'] = (close.shift(-1) > close).astype(int)
            
            # Drop rows with NaN values
            df = df.dropna()
            
            # Check if we have enough data
            if len(df) < 20:
                logger.warning(f"Insufficient data after processing: {len(df)} rows")
                return pd.DataFrame()
                
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering error: {str(e)}")
            return pd.DataFrame()