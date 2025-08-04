"""
RSI + Moving Average Trading Strategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from src.data.indicators import add_all_indicators

class RSIMAStrategy:
    """
    Trading strategy using RSI and Moving Average crossover
    
    Buy Signals:
    - RSI < 30 (oversold)
    - 20-day MA crossing above 50-day MA
    
    Sell Signals:
    - RSI > 70 (overbought)
    - 20-day MA crossing below 50-day MA
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.rsi_oversold = config.get('strategy.rsi_oversold', 30)
        self.rsi_overbought = config.get('strategy.rsi_overbought', 70)
        self.ma_short = config.get('strategy.ma_short', 20)
        self.ma_long = config.get('strategy.ma_long', 50)
        
        # Risk management
        self.stop_loss_pct = config.get('risk_management.stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('risk_management.take_profit_pct', 0.04)
        
        self.logger.info(f"Initialized RSI-MA Strategy with RSI({self.rsi_oversold}, {self.rsi_overbought}) and MA({self.ma_short}, {self.ma_long})")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI and MA crossover
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals and indicator values
        """
        try:
            # Add technical indicators
            df = add_all_indicators(data, self.config)
            
            # Initialize signal column
            df['signal'] = 'hold'
            df['signal_strength'] = 0.0
            df['entry_reason'] = ''
            
            # Calculate MA crossover
            df['ma_cross'] = np.where(df['ma_20'] > df['ma_50'], 1, -1)
            df['ma_cross_signal'] = df['ma_cross'].diff()
            
            # Generate buy signals
            buy_condition = (
                (df['rsi'] < self.rsi_oversold) &  # RSI oversold
                (df['ma_cross_signal'] == 2)        # MA crossover (20 above 50)
            )
            
            # Alternative buy signal (either condition can trigger)
            buy_condition_alt = (
                (df['rsi'] < self.rsi_oversold) |   # RSI oversold OR
                (df['ma_cross_signal'] == 2)        # MA crossover
            )
            
            # Use alternative condition for more signals
            df.loc[buy_condition_alt, 'signal'] = 'buy'
            
            # Calculate signal strength for buy signals
            buy_mask = df['signal'] == 'buy'
            df.loc[buy_mask, 'signal_strength'] = (
                (self.rsi_oversold - df.loc[buy_mask, 'rsi']) / self.rsi_oversold * 0.5 +  # RSI component
                (df.loc[buy_mask, 'ma_cross_signal'] == 2).astype(float) * 0.5             # MA crossover component
            )
            
            # Add entry reasons for buy signals
            df.loc[buy_mask & (df['rsi'] < self.rsi_oversold), 'entry_reason'] += 'RSI_OVERSOLD '
            df.loc[buy_mask & (df['ma_cross_signal'] == 2), 'entry_reason'] += 'MA_CROSSOVER_UP '
            
            # Generate sell signals
            sell_condition = (
                (df['rsi'] > self.rsi_overbought) |  # RSI overbought OR
                (df['ma_cross_signal'] == -2)        # MA crossover down (20 below 50)
            )
            
            df.loc[sell_condition, 'signal'] = 'sell'
            
            # Calculate signal strength for sell signals
            sell_mask = df['signal'] == 'sell'
            df.loc[sell_mask, 'signal_strength'] = (
                (df.loc[sell_mask, 'rsi'] - self.rsi_overbought) / (100 - self.rsi_overbought) * 0.5 +  # RSI component
                (df.loc[sell_mask, 'ma_cross_signal'] == -2).astype(float) * 0.5                        # MA crossover component
            )
            
            # Add entry reasons for sell signals
            df.loc[sell_mask & (df['rsi'] > self.rsi_overbought), 'entry_reason'] += 'RSI_OVERBOUGHT '
            df.loc[sell_mask & (df['ma_cross_signal'] == -2), 'entry_reason'] += 'MA_CROSSOVER_DOWN '
            
            # Additional confirmation signals
            df = self._add_confirmation_signals(df)
            
            # Clean up entry reasons
            df['entry_reason'] = df['entry_reason'].str.strip()
            
            self.logger.info(f"Generated {len(df[df['signal'] != 'hold'])} signals out of {len(df)} data points")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return data.copy()
    
    def _add_confirmation_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional confirmation signals"""
        try:
            # Volume confirmation
            df['volume_avg'] = df['volume'].rolling(window=20).mean()
            df['volume_spike'] = df['volume'] > (df['volume_avg'] * 1.5)
            
            # Trend confirmation using MACD
            df['macd_bullish'] = df['macd'] > df['macd_signal']
            df['macd_bearish'] = df['macd'] < df['macd_signal']
            
            # Enhance signal strength with confirmations
            buy_mask = df['signal'] == 'buy'
            sell_mask = df['signal'] == 'sell'
            
            # Add volume confirmation bonus
            df.loc[buy_mask & df['volume_spike'], 'signal_strength'] += 0.1
            df.loc[sell_mask & df['volume_spike'], 'signal_strength'] += 0.1
            
            # Add MACD confirmation
            df.loc[buy_mask & df['macd_bullish'], 'signal_strength'] += 0.1
            df.loc[sell_mask & df['macd_bearish'], 'signal_strength'] += 0.1
            
            # Cap signal strength at 1.0
            df['signal_strength'] = df['signal_strength'].clip(upper=1.0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding confirmation signals: {e}")
            return df
    
    def calculate_position_size(self, capital: float, price: float, risk_pct: float = 0.02) -> int:
        """
        Calculate position size based on risk management
        
        Args:
            capital: Available capital
            price: Stock price
            risk_pct: Risk percentage per trade
            
        Returns:
            Number of shares to buy/sell
        """
        try:
            max_risk_amount = capital * risk_pct
            stop_loss_amount = price * self.stop_loss_pct
            
            if stop_loss_amount > 0:
                position_size = int(max_risk_amount / stop_loss_amount)
            else:
                position_size = int(capital * 0.1 / price)  # Fallback: 10% of capital
            
            return max(1, position_size)  # Minimum 1 share
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1
    
    def get_stop_loss_price(self, entry_price: float, signal_type: str) -> float:
        """Calculate stop loss price"""
        if signal_type == 'buy':
            return entry_price * (1 - self.stop_loss_pct)
        else:  # sell/short
            return entry_price * (1 + self.stop_loss_pct)
    
    def get_take_profit_price(self, entry_price: float, signal_type: str) -> float:
        """Calculate take profit price"""
        if signal_type == 'buy':
            return entry_price * (1 + self.take_profit_pct)
        else:  # sell/short
            return entry_price * (1 - self.take_profit_pct)
    
    def validate_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Validate if a signal meets minimum criteria
        
        Args:
            signal_data: Dictionary containing signal information
            
        Returns:
            True if signal is valid
        """
        try:
            # Minimum signal strength threshold
            min_strength = 0.3
            
            if signal_data.get('signal_strength', 0) < min_strength:
                return False
            
            # Check if indicators are available
            required_indicators = ['rsi', 'ma_20', 'ma_50']
            for indicator in required_indicators:
                if signal_data.get(indicator) is None:
                    return False
            
            # Additional validation rules can be added here
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get strategy configuration summary"""
        return {
            'name': 'RSI-MA Strategy',
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'ma_short': self.ma_short,
            'ma_long': self.ma_long,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'description': 'Buy when RSI < 30 or MA(20) crosses above MA(50). Sell when RSI > 70 or MA(20) crosses below MA(50).'
        }