#!/usr/bin/env python3
"""
Test script to verify the trading system components
"""

import os
import sys
import logging
import inspect
from datetime import datetime

def test_imports():
    """Test if all modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test external packages
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier
        print("✓ External packages imported successfully")

        # Test internal modules
        from src.utils.config import Config
        from src.utils.logger import setup_logger
        from src.data.data_fetcher import DataFetcher
        from src.data.indicators import TechnicalIndicators, add_all_indicators
        from src.strategy.rsi_ma_strategy import RSIMAStrategy
        from src.strategy.backtester import Backtester
        from src.ml.model_trainer import ModelTrainer
        from src.automation.sheets_manager import SheetsManager
       

        print("✓ All modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_fetching():
    """Test data fetching functionality"""
    try:
        print("\nTesting data fetching...")
        
        from src.utils.config import Config
        from src.data.data_fetcher import DataFetcher
        
        config = Config()
        data_fetcher = DataFetcher(config)
        
        # Test fetching data for one symbol
        symbol = "RELIANCE.NS"
        data = data_fetcher.fetch_daily_data(symbol, days=30)
        
        if data is not None and not data.empty:
            print(f" Successfully fetched {len(data)} days of data for {symbol}")
            print(f"  Columns: {list(data.columns)}")
            print(f"  Date range: {data.index[0]} to {data.index[-1]}")
            return True
        else:
            print(f" Failed to fetch data for {symbol}")
            return False
            
    except Exception as e:
        print(f" Data fetching error: {e}")
        return False

def test_indicators():
    """Test technical indicators"""
    try:
        print("\nTesting technical indicators...")
        
        from src.utils.config import Config
        from src.data.data_fetcher import DataFetcher
        from src.data.indicators import add_all_indicators
        
        config = Config()
        data_fetcher = DataFetcher(config)
        
        # Get sample data
        data = data_fetcher.fetch_daily_data("RELIANCE.NS", days=100)
        if data is None or data.empty:
            print("✗ No data available for indicator testing")
            return False
        
        # Add indicators
        data_with_indicators = add_all_indicators(data, config)
        
        required_indicators = ['rsi', 'ma_20', 'ma_50', 'macd']
        missing_indicators = [ind for ind in required_indicators if ind not in data_with_indicators.columns]
        
        if not missing_indicators:
            print("✓ All technical indicators calculated successfully")
            print(f"  Indicators: {[col for col in data_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol']]}")
            return True
        else:
            print(f"✗ Missing indicators: {missing_indicators}")
            return False
            
    except Exception as e:
        print(f"✗ Indicators error: {e}")
        return False

def test_strategy():
    """Test trading strategy"""
    try:
        print("\nTesting trading strategy...")
        
        from src.utils.config import Config
        from src.data.data_fetcher import DataFetcher
        from src.strategy.rsi_ma_strategy import RSIMAStrategy
        
        config = Config()
        data_fetcher = DataFetcher(config)
        strategy = RSIMAStrategy(config)
        
        # Get sample data
        data = data_fetcher.fetch_daily_data("RELIANCE.NS", days=100)
        if data is None or data.empty:
            print(" No data available for strategy testing")
            return False
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        if not signals.empty:
            signal_counts = signals['signal'].value_counts()
            print("Strategy signals generated successfully")
            print(f"  Signal distribution: {dict(signal_counts)}")
            return True
        else:
            print(" No signals generated")
            return False
            
    except Exception as e:
        print(f"Strategy error: {e}")
        return False

def test_backtesting():
    """Test backtesting functionality with improved error handling"""

    try:
        print("\nTesting backtesting...")
        
        from src.utils.config import Config
        from src.data.data_fetcher import DataFetcher
        from src.strategy.rsi_ma_strategy import RSIMAStrategy
        from src.strategy.backtester import Backtester
        
        config = Config()
        data_fetcher = DataFetcher(config)
        strategy = RSIMAStrategy(config)
        backtester = Backtester(config)
        
        # Get sample data
        data = data_fetcher.fetch_daily_data("RELIANCE.NS", days=150)
        if data is None or data.empty:
            print(" No data available for backtesting")
            return False
        
       

        # Check the method signature of run_backtest
        sig = inspect.signature(backtester.run_backtest)
        param_names = list(sig.parameters.keys())
        param_count = len([p for p in sig.parameters.values() if p.default == inspect.Parameter.empty]) - 1  # Exclude 'self'
        
        print(f"  Backtester method signature: {param_names}")
        print(f"  Required parameters (excluding self): {param_count}")
        
         # Create backtester with strategy and symbol
        backtester = Backtester(
            config=config,
            strategy=strategy,
            symbol="RELIANCE.NS"
        )
        
         # Run backtest with only data parameter
        print("  Trying: run_backtest(data)")
        results = backtester.run_backtest(data)

        
        
        if results and isinstance(results, dict) and results.get('total_return') is not None:
            print("✓ Backtesting completed successfully")
            print(f"  Total return: {results['total_return']:.2%}")
            print(f"  Total trades: {results.get('total_trades', 0)}")
            print(f"  Win rate: {results.get('win_rate', 0):.1%}")
            return True
        else:
            print("✗ Backtesting failed - no valid results returned")
            print(f"  Results type: {type(results)}")
            print(f"  Results content: {results}")
            return False
            
    except TypeError as e:
        print(f"✗ Backtesting method signature error: {e}")
        print("  SOLUTION: Check your backtester.py file and ensure the run_backtest method signature matches the test")
        print("  Expected signature: run_backtest(self, strategy, data, symbol=None)")
        return False
    except Exception as e:
        print(f"✗ Backtesting error: {e}")
        import traceback
        print(f"  Full traceback: {traceback.format_exc()}")
        return False

def test_ml_model():
    """Test ML model functionality with better data requirements"""
    try:
        print("\nTesting ML model...")
        
        from src.utils.config import Config
        from src.data.data_fetcher import DataFetcher
        from src.ml.model_trainer import ModelTrainer
        
        config = Config()
        data_fetcher = DataFetcher(config)
        ml_trainer = ModelTrainer(config)
        
        # Get more data for ML training (ML needs more historical data)
        print("  Fetching more data for ML training...")
        data = data_fetcher.fetch_daily_data("RELIANCE.NS", days=300)
        
        if data is None or data.empty:
            print("✗ No data available for ML testing")
            return False
        
        print(f"  Got {len(data)} days of data for ML training")
        
        if len(data) < 100:
            print(f"✗ Insufficient data for ML training: {len(data)} days (minimum 100 required)")
            return False
        
        # Train model and get predictions
        accuracy, predictions = ml_trainer.train_and_predict(data, "RELIANCE.NS")
        
        if accuracy > 0 and len(predictions) > 0:
            print("✓ ML model trained successfully")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Predictions generated: {len(predictions)}")
            return True
        else:
            print("✗ ML model training failed")
            print(f"  Accuracy: {accuracy}")
            print(f"  Predictions count: {len(predictions) if predictions else 0}")
            return False
            
    except Exception as e:
        print(f"✗ ML model error: {e}")
        import traceback
        print(f"  Full traceback: {traceback.format_exc()}")
        return False

def test_configuration():
    """Test configuration management"""
    try:
        print("\nTesting configuration...")
        
        from src.utils.config import Config
        
        config = Config()
        
        # Test getting configuration values
        rsi_period = config.get('strategy.rsi_period', 14)
        symbols = config.get('stocks.symbols', [])
        
        print(" Configuration loaded successfully")
        print(f"  RSI period: {rsi_period}")
        print(f"  Symbols configured: {len(symbols)}")
        return True
        
    except Exception as e:
        print(f" Configuration error: {e}")
        return False

def run_full_test():
    """Run complete system test"""
    print("ALGORITHMIC TRADING SYSTEM - FULL TEST")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Data Fetching", test_data_fetching),
        ("Technical Indicators", test_indicators),
        ("Trading Strategy", test_strategy),
        ("Backtesting", test_backtesting),
        ("ML Model", test_ml_model)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print(" All tests passed! System is ready to use.")
        print("\nTo run the trading system:")
        print("python main.py")
    else:
        print("  Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("1. For backtesting: Check run_backtest method signature in backtester.py")
        print("2. For ML model: Ensure you have enough historical data (300+ days recommended)")
        print("3. Check import paths match your project structure")
    
    return passed == total

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during testing
    
    success = run_full_test()
    sys.exit(0 if success else 1)

