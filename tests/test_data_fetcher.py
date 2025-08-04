# test_fetcher.py
from src.utils.config import Config
from src.data.data_fetcher import DataFetcher

config = Config()
fetcher = DataFetcher(config)

# Test single symbol
data = fetcher.fetch_intraday_data("RELIANCE.NS", period="7d", interval="1d")
print(f"Fetched {len(data)} rows")
print(data.tail())

# Test multiple symbols
symbols = config.get('stocks.symbols')
for symbol in symbols:
    data = fetcher.fetch_stock_data(symbol)
    print(f"{symbol}: {len(data)} rows")