# src/automation/sheets_manager.py
import gspread
import logging
from datetime import datetime
from typing import Dict, Any, List
from google.oauth2.service_account import Credentials

class SheetsManager:
    """Manages Google Sheets integration for trade logging"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.sheets_url = config.get('sheets.url', '')
        self.credentials_path = config.get('sheets.credentials_path', 'config/credentials.json')
        
        self.client = None
        self.spreadsheet = None
        self.initialized = False
        
        # Sheet names
        self.trading_log_sheet = "Trading_Log"
        self.performance_sheet = "Performance_Summary"
        self.signals_sheet = "Current_Signals"
        self.ml_sheet = "ML_Predictions"
    
    def initialize(self):
        """Initialize Google Sheets connection"""
        if self.initialized:
            return True
            
        try:
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = Credentials.from_service_account_file(
                self.credentials_path, 
                scopes=scopes
            )
            self.client = gspread.authorize(creds)
            
            # Open spreadsheet
            if self.sheets_url:
                self.spreadsheet = self.client.open_by_url(self.sheets_url)
                self.logger.info("Successfully connected to Google Sheets")
                self.initialized = True
                return True
            else:
                self.logger.error("No Google Sheets URL provided in configuration")
                return False
                
        except FileNotFoundError:
            self.logger.error(f"Credentials file not found: {self.credentials_path}")
        except Exception as e:
            self.logger.error(f"Error initializing Google Sheets: {e}")
        
        return False
    
    def get_worksheet(self, sheet_name):
        """Get or create a worksheet"""
        if not self.initialize():
            return None
            
        try:
            return self.spreadsheet.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            try:
                self.logger.info(f"Creating new worksheet: {sheet_name}")
                return self.spreadsheet.add_worksheet(title=sheet_name, rows=100, cols=20)
            except Exception as e:
                self.logger.error(f"Error creating worksheet: {e}")
                return None
    
    def update_trading_log(self, backtest_results: Dict[str, Any]):
        """Update the trading log worksheet"""
        if not self.initialize():
            return
            
        try:
            worksheet = self.get_worksheet(self.trading_log_sheet)
            if not worksheet:
                return
                
            # Clear existing data except headers
            if worksheet.row_count > 1:
                worksheet.delete_rows(2, worksheet.row_count)
            
            # Prepare data for all symbols
            all_trades = []
            for symbol, results in backtest_results.items():
              # Handle both portfolio-level and symbol-level results
              trades = results.get('trades', []) if isinstance(results, dict) else []
        
            if not trades and 'portfolio' in results:
               trades = results['portfolio'].get('trades', [])
            
            for trade in trades:
               # Ensure trade has required fields
               trade_row = [
                 trade.get('entry_time', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                 symbol,
                 trade.get('type', 'BUY').upper(),
                 trade.get('size', 0),
                 trade.get('entry_price', 0),
                 # Add default values for missing fields
                 trade.get('value', trade.get('size', 0) * trade.get('entry_price', 0)),
                 trade.get('commission', 0),
                 trade.get('signal_strength', 0.0),
                 trade.get('reason', ''),
                 trade.get('indicators', {}).get('rsi', 0),
                 trade.get('indicators', {}).get('ma_20', 0),
                 trade.get('indicators', {}).get('ma_50', 0),
                 trade.get('pnl', 0),
                'BACKTEST',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
               ]
               all_trades.append(trade_row)

            
             # Add trades to worksheet
            if all_trades:
                worksheet.append_rows(all_trades)
                self.logger.info(f"Added {len(all_trades)} trades to Google Sheets")
            
        except Exception as e:
            self.logger.error(f"Error updating trading log: {e}")
    
    def update_performance_summary(self, backtest_results: Dict[str, Any]):
        """Update the performance summary worksheet"""
        if not self.initialize():
            return
            
        try:
            worksheet = self.get_worksheet(self.performance_sheet)
            if not worksheet:
                return
                
            # Prepare data
            data = []
            headers = [
                'Symbol', 'Start_Date', 'End_Date', 'Initial_Capital',
                'Final_Value', 'Total_Return', 'Total_Trades', 'Win_Rate',
                'Max_Drawdown', 'Sharpe_Ratio'
            ]
            data.append(headers)
            
            for symbol, results in backtest_results.items():
                data.append([
                    results.get('symbol', symbol),
                    results.get('start_date', '').strftime('%Y-%m-%d') if hasattr(results.get('start_date', ''), 'strftime') else str(results.get('start_date', '')),
                    results.get('end_date', '').strftime('%Y-%m-%d') if hasattr(results.get('end_date', ''), 'strftime') else str(results.get('end_date', '')),
                    results.get('initial_capital', 0),
                    results.get('final_value', 0),
                    results.get('total_return', 0),
                    results.get('total_trades', 0),
                    results.get('win_rate', 0),
                    results.get('max_drawdown', 0),
                    results.get('sharpe_ratio', 0)
                ])
            
            # Update the entire sheet
            worksheet.clear()
            worksheet.update('A1', data)
            self.logger.info(f"Updated performance summary for {len(backtest_results)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error updating performance summary: {e}")
    
    def update_current_signals(self, current_signals: Dict[str, Any]):
        """Update current trading signals worksheet"""
        if not self.initialize():
            return
            
        try:
            worksheet = self.get_worksheet(self.signals_sheet)
            if not worksheet:
                return
                
            # Prepare data
            data = []
            headers = [
                'Symbol', 'Signal', 'Price', 'RSI', 'MA_20', 'MA_50',
                'Date', 'Updated'
            ]
            data.append(headers)
            
            for symbol, signal_data in current_signals.items():
                data.append([
                    symbol,
                    signal_data.get('signal', '').upper(),
                    signal_data.get('price', 0),
                    signal_data.get('rsi', 0),
                    signal_data.get('ma_20', 0),
                    signal_data.get('ma_50', 0),
                    signal_data.get('date', '').strftime('%Y-%m-%d') if hasattr(signal_data.get('date', ''), 'strftime') else str(signal_data.get('date', '')),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ])
            
            # Update the entire sheet
            worksheet.clear()
            worksheet.update('A1', data)
            self.logger.info(f"Updated {len(current_signals)} current signals")
            
        except Exception as e:
            self.logger.error(f"Error updating current signals: {e}")
    
    def update_ml_predictions(self, ml_results: Dict[str, Any]):
        """Update ML predictions worksheet"""
        if not self.initialize():
            return
            
        try:
            worksheet = self.get_worksheet(self.ml_sheet)
            if not worksheet:
                return
                
            # Prepare data
            data = []
            headers = [
                'Symbol', 'Model_Accuracy', 'Latest_Prediction',
                'Confidence', 'Date', 'Updated'
            ]
            data.append(headers)
            
            for symbol, ml_data in ml_results.items():
                data.append([
                    symbol,
                    ml_data.get('accuracy', 0),
                    ml_data.get('latest_prediction', ''),
                    ml_data.get('confidence', 0),
                    datetime.now().strftime('%Y-%m-%d'),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ])
            
            # Update the entire sheet
            worksheet.clear()
            worksheet.update('A1', data)
            self.logger.info(f"Updated ML predictions for {len(ml_results)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error updating ML predictions: {e}")