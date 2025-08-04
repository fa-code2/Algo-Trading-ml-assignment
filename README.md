# Algorithmic Trading System

A complete algorithmic trading system with RSI + Moving Average strategy, ML predictions, and Google Sheets automation.

## Features

1. **Data Ingestion**: Fetch stock data using yfinance
2. **Trading Strategy**: RSI + Moving Average crossover strategy
3. **Backtesting**: Complete backtesting engine with performance metrics
4. **Machine Learning**: Predict next-day movements using Decision Tree
5. **Google Sheets Integration**: Automated logging of trades and performance
6. **Comprehensive Logging**: Detailed logging and monitoring

## Quick Start

1. **Setup Environment**:
   ```bash
   python setup.py
   ```

2. **Run the System**:
   ```bash
   python main.py
   ```

3. **Schedule Continuous Execution**:
   ```bash
   python main.py --mode schedule
   ```

## Configuration

- Edit `config/config.yaml` to customize strategy parameters
- Add API keys to `config/api_keys.env` (copy from `api_keys.env.example`)
- For Google Sheets integration, add `credentials.json` file

## Strategy Details

- **Buy Signal**: RSI < 30 (oversold) OR 20-MA crosses above 50-MA
- **Sell Signal**: RSI > 70 (overbought) OR 20-MA crosses below 50-MA
- **Risk Management**: 2% stop loss, 4% take profit

## Output

The system provides:
- Backtest results with performance metrics
- ML prediction accuracy
- Current trading signals
- Google Sheets with detailed logs

## Requirements

See `requirements.txt` for all dependencies.
