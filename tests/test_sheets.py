# test_sheets.py
from src.utils.config import Config
from src.automation.sheets_manager import SheetsManager

config = Config()
sheets = SheetsManager(config)

if sheets.initialize():
    print(" Google Sheets connection successful!")
    sheets.update_performance_summary({"TEST": {"symbol": "TEST", "total_return": 0.1}})
else:
    print(" Connection failed")