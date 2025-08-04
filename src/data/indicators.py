"""
Technical indicators for trading strategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

class TechnicalIndicators:
    """Calculate various technical indicators"""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Series of close prices
            period: RSI period (default 14)
            
        Returns:
            Series with RSI values
        """
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Fill NaN with neutral RSI value
            
        except Exception as e:
            logging.error(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, data=50)  # Return neutral RSI on error
    
    @staticmethod
    def moving_average(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA)
        
        Args:
            prices: Series of prices
            period: Moving average period
            
        Returns:
            Series with moving average values
        """
        try:
            return prices.rolling(window=period, min_periods=1).mean()
        except Exception as e:
            logging.error(f"Error calculating moving average: {e}")
            return prices.copy()
    
    @staticmethod
    def exponential_moving_average(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA)
        
        Args:
            prices: Series of prices
            period: EMA period
            
        Returns:
            Series with EMA values
        """
        try:
            return prices.ewm(span=period, min_periods=1).mean()
        except Exception as e:
            logging.error(f"Error calculating EMA: {e}")
            return prices.copy()
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Series of close prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        try:
            ema_fast = TechnicalIndicators.exponential_moving_average(prices, fast)
            ema_slow = TechnicalIndicators.exponential_moving_average(prices, slow)
            
            macd_line = ema_fast - ema_slow
            signal_line = TechnicalIndicators.exponential_moving_average(macd_line, signal)
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            logging.error(f"Error calculating MACD: {e}")
            # Return zero series on error
            zero_series = pd.Series(index=prices.index, data=0)
            return zero_series, zero_series, zero_series
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Series of close prices
            period: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        try:
            middle_band = TechnicalIndicators.moving_average(prices, period)
            std = prices.rolling(window=period, min_periods=1).std()
            
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {e}")
            # Return price series on error
            return prices.copy(), prices.copy(), prices.copy()
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K, %D)
        """
        try:
            lowest_low = low.rolling(window=k_period, min_periods=1).min()
            highest_high = high.rolling(window=k_period, min_periods=1).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
            
            return k_percent.fillna(50), d_percent.fillna(50)
            
        except Exception as e:
            logging.error(f"Error calculating Stochastic Oscillator: {e}")
            neutral_series = pd.Series(index=close.index, data=50)
            return neutral_series, neutral_series
    
    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Volume Simple Moving Average
        
        Args:
            volume: Series of volume data
            period: Moving average period
            
        Returns:
            Series with volume SMA
        """
        try:
            return volume.rolling(window=period, min_periods=1).mean()
        except Exception as e:
            logging.error(f"Error calculating volume SMA: {e}")
            return volume.copy()
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        try:
            prev_close = close.shift(1)
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period, min_periods=1).mean()
            
            return atr.fillna(0)
            
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            return pd.Series(index=close.index, data=0)

def add_all_indicators(data: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    Add all technical indicators to the dataframe
    
    Args:
        data: DataFrame with OHLCV data
        config: Configuration object (optional)
        
    Returns:
        DataFrame with added indicators
    """
    try:
        df = data.copy()
        
        # Get parameters from config or use defaults
        if config:
            rsi_period = config.get('strategy.rsi_period', 14)
            ma_short = config.get('strategy.ma_short', 20)
            ma_long = config.get('strategy.ma_long', 50)
            macd_fast = config.get('strategy.macd_fast', 12)
            macd_slow = config.get('strategy.macd_slow', 26)
            macd_signal = config.get('strategy.macd_signal', 9)
        else:
            rsi_period = 14
            ma_short = 20
            ma_long = 50
            macd_fast = 12
            macd_slow = 26
            macd_signal = 9
        
        # RSI
        df['rsi'] = TechnicalIndicators.rsi(df['close'], rsi_period)
        
        # Moving Averages
        df['ma_20'] = TechnicalIndicators.moving_average(df['close'], ma_short)
        df['ma_50'] = TechnicalIndicators.moving_average(df['close'], ma_long)
        
        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            df['close'], macd_fast, macd_slow, macd_signal
        )
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # Volume indicators
        df['volume_sma'] = TechnicalIndicators.volume_sma(df['volume'])
        
        # ATR
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        # Stochastic Oscillator
        stoch_k, stoch_d = TechnicalIndicators.stochastic_oscillator(
            df['high'], df['low'], df['close']
        )
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        logging.info(f"Added technical indicators to {len(df)} data points")
        return df
        
    except Exception as e:
        logging.error(f"Error adding indicators: {e}")
        return data