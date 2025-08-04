
"""
Main entry point for the algorithmic trading system
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import schedule
import time


from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data.data_fetcher import DataFetcher
from src.strategy.rsi_ma_strategy import RSIMAStrategy
from src.strategy.backtester import Backtester
from src.ml.model_trainer import ModelTrainer
from src.automation.sheets_manager import SheetsManager


def run_trading_system():
    """Main function to run the complete trading system"""
    logger = logging.getLogger(__name__)
    logger.info("Starting algorithmic trading system...")
    
    try:
        # Initialize components
        config = Config()
        data_fetcher = DataFetcher(config)
        strategy = RSIMAStrategy(config)
        sheets_manager = SheetsManager(config)
        
        # Fetch data for configured stocks
        logger.info("Fetching market data...")
        all_data = {}
        for symbol in config.get('stocks.symbols')[:3]:  # Use first 3 stocks as requested
            try:
                data = data_fetcher.fetch_daily_data(symbol, days=200)
                if data is not None and not data.empty:
                    all_data[symbol] = data
                    logger.info(f"Fetched {len(data)} days of data for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        if not all_data:
            logger.error("No market data available. Exiting.")
            return
        
        # Run backtesting
        logger.info("Running backtesting...")
        backtester = Backtester(config)
        
        backtest_results = {}
        for symbol, data in all_data.items():
            try:
                 # Initialize backtester with strategy and symbol
                backtester = Backtester(
                    config=config,
                    strategy=strategy,
                    symbol=symbol  # THIS IS CRITICAL
                )

                # Run backtest with data
                result = backtester.run_backtest(data)
                if result:
                    backtest_results[symbol] = result
                    logger.info(f"Backtest completed for {symbol}")
            except Exception as e:
                logger.error(f"Error in backtesting {symbol}: {e}")
        
        # Run ML predictions (bonus feature)
        logger.info("Training ML model and making predictions...")
        try:
            ml_trainer = ModelTrainer(config)
            ml_results = {}
            
            for symbol, data in all_data.items():
                try:
                    accuracy, predictions = ml_trainer.train_and_predict(data, symbol)
                    ml_results[symbol] = {
                        'accuracy': accuracy,
                        'latest_prediction': predictions[-1] if predictions else None
                    }
                    logger.info(f"ML accuracy for {symbol}: {accuracy:.2%}")
                except Exception as e:
                    logger.error(f"Error in ML training for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error in ML module: {e}")
            ml_results = {}
        
        # Generate current signals
        logger.info("Generating current trading signals...")
        current_signals = {}
        for symbol, data in all_data.items():
            try:
                signals = strategy.generate_signals(data)
                if not signals.empty:

                    latest_signal = signals.iloc[-1]
                    signal_date = signals.index[-1]  # This gets the datetime index

                    if latest_signal['signal'] != 'hold':
                        current_signals[symbol] = {
                            'signal': latest_signal['signal'],
                            'price': latest_signal['close'],
                            'rsi': latest_signal['rsi'],
                            'ma_20': latest_signal['ma_20'],
                            'ma_50': latest_signal['ma_50'],
                            'date': latest_signal.name
                        }
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")
        
        # Update Google Sheets
        logger.info("Updating Google Sheets...")
        try:
            sheets_manager.update_trading_log(backtest_results)
            sheets_manager.update_performance_summary(backtest_results)
            sheets_manager.update_current_signals(current_signals)
            if ml_results:
                sheets_manager.update_ml_predictions(ml_results)
            logger.info("Google Sheets updated successfully")
        except Exception as e:
            logger.error(f"Error updating Google Sheets: {e}")
        
        # Print summary
        print("\n" + "="*60)
        print("TRADING SYSTEM EXECUTION SUMMARY")
        print("="*60)
        
        print(f"\n Data Fetched for {len(all_data)} stocks:")
        for symbol in all_data.keys():
            print(f"   • {symbol}")
        
        print(f"\n Backtest Results:")
        for symbol, result in backtest_results.items():
            print(f"   • {symbol}: {result['total_return']:.2%} return, {result['win_rate']:.1%} win rate")
        
        if ml_results:
            print(f"\n ML Predictions (Accuracy):")
            for symbol, result in ml_results.items():
                print(f"   • {symbol}: {result['accuracy']:.2%}")
        
        if current_signals:
            print(f"\n Current Signals:")
            for symbol, signal in current_signals.items():
                print(f"   • {symbol}: {signal['signal'].upper()} at ₹{signal['price']:.2f}")
        else:
            print(f"\n Current Signals: No active signals")
        
        print(f"\n Google Sheets updated with all results")
        print("="*60)
        
        logger.info("Trading system execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main trading system: {e}")
        raise

def schedule_trading_system():
    """Schedule the trading system to run automatically"""
    logger = logging.getLogger(__name__)
    
    # Schedule to run every hour during market hours
    schedule.every().hour.do(run_trading_system)
    
    # Schedule daily summary at market close
    schedule.every().day.at("16:00").do(run_trading_system)
    
    logger.info("Trading system scheduled. Running initial execution...")
    run_trading_system()
    
    logger.info("Waiting for scheduled executions...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    # Setup logging
    logger = setup_logger()
    
    import argparse
    parser = argparse.ArgumentParser(description='Algorithmic Trading System')
    parser.add_argument('--mode', choices=['run', 'schedule'], default='run',
                       help='Run once or schedule continuous execution')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'schedule':
            schedule_trading_system()
        else:
            run_trading_system()
    except KeyboardInterrupt:
        logger.info("Trading system stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)