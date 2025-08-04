import pandas as pd
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, name, config):
        self.name = name
        self.config = config
    
    @abstractmethod
    def generate_signal(self, data):
        pass
    
    @abstractmethod
    def backtest(self, data):
        pass