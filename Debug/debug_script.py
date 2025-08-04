#!/usr/bin/env python3
"""
Debug script to identify specific import and strategy issues
"""

import sys
import traceback

def debug_imports():
    """Debug import issues one by one"""
    print("DEBUGGING IMPORTS")
    print("="*50)
    
    imports_to_test = [
        ("yfinance", "import yfinance as yf"),
        ("pandas", "import pandas as pd"),
        ("numpy", "import numpy as np"),
        ("sklearn", "from sklearn.tree import DecisionTreeClassifier"),
        ("src.utils.config", "from src.utils.config import Config"),
        ("src.utils.logger", "from src.utils.logger import setup_logger"),
        ("src.data.data_fetcher", "from src.data.data_fetcher import DataFetcher"),
        ("src.data.indicators", "from src.data.indicators import TechnicalIndicators, add_all_indicators"),
        ("src.strategy.rsi_ma_strategy", "from src.strategy.rsi_ma_strategy import RSIMAStrategy"),
        ("src.strategy.backtester", "from src.strategy.backtester import Backtester"),
        ("src.ml.model_trainer", "from src.ml.model_trainer import ModelTrainer"),
        ("src.automation.sheets_manager", "from src.automation.sheets_manager import SheetsManager"),
    ]
    
    failed_imports = []
    
    for name, import_statement in imports_to_test:
        try:
            exec(import_statement)
            print(f"✓ {name}")
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed_imports.append((name, str(e)))
    
    return failed_imports

def debug_strategy():
    """Debug strategy issues"""
    print("\nDEBUGGING STRATEGY")
    print("="*50)
    
    try:
        # Test config first
        print("Testing Config...")
        from src.utils.config import Config
        config = Config()
        print("✓ Config loaded")
        
        # Test data fetcher
        print("Testing DataFetcher...")
        from src.data.data_fetcher import DataFetcher
        data_fetcher = DataFetcher(config)
        print("✓ DataFetcher created")
        
        # Test strategy import
        print("Testing RSIMAStrategy import...")
        from src.strategy.rsi_ma_strategy import RSIMAStrategy
        print("✓ RSIMAStrategy imported")
        
        # Test strategy creation
        print("Testing RSIMAStrategy creation...")
        strategy = RSIMAStrategy(config)
        print("✓ RSIMAStrategy created")
        
        # Test data fetching for strategy
        print("Testing data fetch...")
        data = data_fetcher.fetch_daily_data("RELIANCE.NS", days=100)
        if data is None or data.empty:
            print("✗ No data available")
            return False
        print(f"✓ Data fetched: {len(data)} rows")
        
        # Test signal generation
        print("Testing signal generation...")
        print(f"Data columns: {list(data.columns)}")
        
        # Check if strategy has generate_signals method
        if not hasattr(strategy, 'generate_signals'):
            print("✗ Strategy missing generate_signals method")
            return False
        
        signals = strategy.generate_signals(data)
        print(f"✓ Signals generated: {len(signals)} rows")
        
        if 'signal' in signals.columns:
            signal_counts = signals['signal'].value_counts()
            print(f"Signal distribution: {dict(signal_counts)}")
        else:
            print("✗ No 'signal' column in output")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Strategy error: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

def debug_backtesting():
    """Debug backtesting issues"""
    print("\nDEBUGGING BACKTESTING")
    print("="*50)
    
    try:
        # Import required modules
        from src.utils.config import Config
        from src.data.data_fetcher import DataFetcher
        from src.strategy.rsi_ma_strategy import RSIMAStrategy
        from src.strategy.backtester import Backtester
        import inspect
        
        config = Config()
        data_fetcher = DataFetcher(config)
        strategy = RSIMAStrategy(config)
        backtester = Backtester(config)
        
        print("✓ All modules imported for backtesting")
        
        # Check backtester method signature
        sig = inspect.signature(backtester.run_backtest)
        print(f"Backtester signature: {sig}")
        
        # Get parameters
        params = list(sig.parameters.keys())
        print(f"Parameters: {params}")
        
        # Check if file exists
        import os
        backtester_path = "src/strategy/backtester.py"
        if os.path.exists(backtester_path):
            print(f"✓ Backtester file exists: {backtester_path}")
        else:
            print(f"✗ Backtester file missing: {backtester_path}")
            
        return True
        
    except Exception as e:
        print(f"✗ Backtesting debug error: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

def main():
    print("TRADING SYSTEM DEBUG ANALYSIS")
    print("="*60)
    
    # Debug imports
    failed_imports = debug_imports()
    
    # Debug strategy
    strategy_ok = debug_strategy()
    
    # Debug backtesting  
    backtesting_ok = debug_backtesting()
    
    print("\n" + "="*60)
    print("DEBUG SUMMARY")
    print("="*60)
    
    if failed_imports:
        print("FAILED IMPORTS:")
        for name, error in failed_imports:
            print(f"  {name}: {error}")
    else:
        print("✓ All imports working")
    
    print(f"Strategy test: {'✓ PASS' if strategy_ok else '✗ FAIL'}")
    print(f"Backtesting test: {'✓ PASS' if backtesting_ok else '✗ FAIL'}")
    
    # Suggestions
    print("\nSUGGEST)ED FIXES:")
    if failed_imports:
        print("1. Fix missing imports - check file paths and missing files")
    if not strategy_ok:
        print("2. Check RSIMAStrategy class and generate_signals method")
    if not backtesting_ok:
        print("3. Check Backtester class and run_backtest method signature")

if __name__ == "__main__":
    main()