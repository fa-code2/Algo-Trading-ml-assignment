"""
Backtesting engine for trading strategies
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from src.data.indicators import add_all_indicators

class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, config, strategy=None, symbol=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.strategy = strategy
        self.symbol = symbol 
        
        # Backtesting parameters
        self.initial_capital = config.get('backtesting.initial_capital', 100000)
        self.commission = config.get('backtesting.commission', 0.001)  # 0.1%
        self.slippage = config.get('backtesting.slippage', 0.001)  # 0.1%
        
        # Log initialization
        self.logger.info(f"Initialized backtester for {self.symbol or 'no symbol'}")
    
    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest using the instance-level strategy and symbol
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with backtest results
        """
        if not self.symbol:
            self.logger.error("Symbol not set for backtesting")
            return self._empty_results()
            
        if not self.strategy:
            self.logger.error("Strategy not set for backtesting")
            return self._empty_results()

        try:
            self.logger.info(f"Starting backtest for {self.symbol}")

            # Add technical indicators to data
            data_with_indicators = add_all_indicators(data, self.config)
            
            if data_with_indicators.empty:
                self.logger.error("No data available after adding indicators")
                return self._empty_results()
            
            # Generate trading signals using INSTANCE strategy
            signals = self.strategy.generate_signals(data_with_indicators)
            
            if signals.empty:
                self.logger.warning("No trading signals generated")
                return self._empty_results()
            
            # Initialize tracking variables
            capital = self.initial_capital
            position = 0  # Number of shares held
            cash = capital
            portfolio_value = []
            trades = []
            
            # Track positions
            in_position = False
            entry_price = 0
            entry_date = None
            
            for i, (date, row) in enumerate(signals.iterrows()):
                current_price = row['close']
                signal = row.get('signal', 'hold')
                
                # Calculate current portfolio value
                current_portfolio_value = cash + (position * current_price)
                portfolio_value.append({
                    'date': date,
                    'value': current_portfolio_value,
                    'cash': cash,
                    'position': position,
                    'price': current_price
                })
                
                # Execute trades based on signals
                if signal == 'buy' and not in_position and cash > current_price:
                    # Buy signal - enter long position
                    shares_to_buy = int(cash * 0.95 / current_price)  # Use 95% of cash
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price * (1 + self.commission + self.slippage)
                        if cost <= cash:
                            position += shares_to_buy
                            cash -= cost
                            in_position = True
                            entry_price = current_price
                            entry_date = date
                            
                            trades.append({
                                'date': date,
                                'type': 'buy',
                                'price': current_price,
                                'shares': shares_to_buy,
                                'value': cost
                            })
                
                elif signal == 'sell' and in_position and position > 0:
                    # Sell signal - exit position
                    sell_value = position * current_price * (1 - self.commission - self.slippage)
                    cash += sell_value
                    
                    # Calculate trade profit/loss
                    trade_pnl = (current_price - entry_price) * position
                    trade_return = (current_price - entry_price) / entry_price if entry_price > 0 else 0
                    
                    trades.append({
                        'date': date,
                        'type': 'sell',
                        'price': current_price,
                        'shares': position,
                        'value': sell_value,
                        'pnl': trade_pnl,
                        'return': trade_return,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'hold_days': (date - entry_date).days if entry_date else 0
                    })
                    
                    position = 0
                    in_position = False
                    entry_price = 0
                    entry_date = None
            
            # Calculate final portfolio value
            final_price = signals.iloc[-1]['close']
            final_portfolio_value = cash + (position * final_price)
            
            # Calculate performance metrics
            results = self._calculate_metrics(
                initial_capital=self.initial_capital,
                final_value=final_portfolio_value,
                portfolio_values=portfolio_value,
                trades=trades,
                data=data_with_indicators
            )
            
            # Use INSTANCE symbol
            results['symbol'] = self.symbol or 'Unknown'
            results['data_points'] = len(data_with_indicators)
            results['signal_distribution'] = dict(signals['signal'].value_counts()) if 'signal' in signals.columns else {}
            
            self.logger.info(f"Backtest completed for {self.symbol}: {results['total_return']:.2%} return, {results['total_trades']} trades")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            return self._empty_results()
    
    def _calculate_metrics(self, initial_capital: float, final_value: float, 
                          portfolio_values: list, trades: list, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        # Basic returns
        total_return = (final_value - initial_capital) / initial_capital
        
        # Trade statistics
        total_trades = len([t for t in trades if t['type'] == 'sell'])
        winning_trades = len([t for t in trades if t['type'] == 'sell' and t.get('pnl', 0) > 0])
        losing_trades = len([t for t in trades if t['type'] == 'sell' and t.get('pnl', 0) <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average returns
        sell_trades = [t for t in trades if t['type'] == 'sell']
        avg_return = np.mean([t.get('return', 0) for t in sell_trades]) if sell_trades else 0
        avg_winning_return = np.mean([t['return'] for t in sell_trades if t.get('pnl', 0) > 0]) if winning_trades > 0 else 0
        avg_losing_return = np.mean([t['return'] for t in sell_trades if t.get('pnl', 0) <= 0]) if losing_trades > 0 else 0
        
        # Portfolio value series for volatility calculations
        if portfolio_values:
            portfolio_series = pd.Series([pv['value'] for pv in portfolio_values])
            portfolio_returns = portfolio_series.pct_change().dropna()
            
            # Volatility (annualized)
            volatility = portfolio_returns.std() * np.sqrt(252) if len(portfolio_returns) > 1 else 0
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = (total_return - 0) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak
            max_drawdown = drawdown.min()
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Buy and hold comparison
        if len(data) > 1:
            buy_hold_return = (data.iloc[-1]['close'] - data.iloc[0]['close']) / data.iloc[0]['close']
        else:
            buy_hold_return = 0
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_winning_return': avg_winning_return,
            'avg_losing_return': avg_losing_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'trades_detail': trades,
            'portfolio_values': portfolio_values
        }
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'symbol': self.symbol or 'Unknown',
            'total_return': 0.0,
            'final_value': self.initial_capital,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'avg_winning_return': 0.0,
            'avg_losing_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'buy_hold_return': 0.0,
            'excess_return': 0.0,
            'trades_detail': [],
            'portfolio_values': []
        }
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a formatted backtest report"""
        
        report = f"""
BACKTEST REPORT - {results.get('symbol', 'Unknown')}
{'='*50}

PERFORMANCE SUMMARY:
Total Return:           {results['total_return']:.2%}
Buy & Hold Return:      {results['buy_hold_return']:.2%}
Excess Return:          {results['excess_return']:.2%}
Final Portfolio Value:  ${results['final_value']:,.2f}

RISK METRICS:
Volatility (Annual):    {results['volatility']:.2%}
Sharpe Ratio:          {results['sharpe_ratio']:.2f}
Maximum Drawdown:      {results['max_drawdown']:.2%}

TRADING STATISTICS:
Total Trades:          {results['total_trades']}
Winning Trades:        {results['winning_trades']}
Losing Trades:         {results['losing_trades']}
Win Rate:             {results['win_rate']:.1%}

AVERAGE RETURNS:
Average Return/Trade:  {results['avg_return']:.2%}
Average Winning Trade: {results['avg_winning_return']:.2%}
Average Losing Trade:  {results['avg_losing_return']:.2%}

DATA INFO:
Data Points:          {results.get('data_points', 'N/A')}
Signal Distribution:  {results.get('signal_distribution', {})}
"""
        return report
    