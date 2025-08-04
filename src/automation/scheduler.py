import schedule
import time
from src.utils.logger import setup_logger

logger = setup_logger("Scheduler")

class TradingScheduler:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.jobs = []
    
    def schedule_daily_trading(self, time_str="09:30"):
        self.jobs.append(schedule.every().day.at(time_str).do(
            self.trading_system.run_trading_cycle
        ))
        logger.info(f"Scheduled daily trading at {time_str}")
    
    def run_pending(self):
        while True:
            schedule.run_pending()
            time.sleep(60)