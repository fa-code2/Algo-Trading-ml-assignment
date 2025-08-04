# test_backtester.py
from src.utils.config import Config
from src.data.data_fetcher import DataFetcher
from src.strategy.rsi_ma_strategy import RSIMAStrategy
from src.strategy.backtester import Backtester

config = Config()
data_fetcher = DataFetcher(config)
strategy = RSIMAStrategy(config)

# Test one symbol
symbol = "RELIANCE.NS"
data = data_fetcher.fetch_daily_data(symbol, days=100)

# Initialize backtester with strategy AND symbol
backtester = Backtester(
    config=config,
    strategy=strategy,
    symbol=symbol  # THIS IS CRITICAL
)

# Run backtest with only data parameter
results = backtester.run_backtest(data)

print(f"Backtest results for {symbol}:")
print(f"Total return: {results['total_return']:.2%}")
print(f"Win rate: {results['win_rate']:.1%}")

