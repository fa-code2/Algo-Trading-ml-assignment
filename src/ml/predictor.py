import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from src.ml.feature_engineer import FeatureEngineer

class MLPredictor:
    """Machine learning prediction engine"""
    
    def __init__(self, model: BaseEstimator):
        self.model = model
        self.feature_engineer = FeatureEngineer()
        
    def predict(self, data: pd.DataFrame) -> dict:
        """
        Make prediction using trained model
        Returns: {
            'direction': 0/1 (down/up),
            'confidence': 0.0-1.0,
            'features': [...]  # For interpretability
        }
        """
        # Feature engineering
        features = self.feature_engineer.transform(data)
        
        if features is None or len(features) == 0:
            return {'direction': 0, 'confidence': 0.5}
        
        # Get latest features
        latest = features.iloc[-1].values.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(latest)[0]
        probability = self.model.predict_proba(latest)[0][1]
        
        return {
            'direction': prediction,
            'confidence': probability,
            'features': latest.tolist()[0]
        }