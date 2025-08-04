"""
Machine Learning model trainer for predicting stock movements
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Fixed import path
from src.data.indicators import add_all_indicators

class ModelTrainer:
    """Machine Learning model trainer for stock prediction"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ML configuration
        self.model_type = config.get('ml.model_type', 'decision_tree')
        self.features = config.get('ml.features', ['rsi', 'ma_20', 'ma_50', 'macd', 'volume_sma'])
        self.train_test_split_ratio = config.get('ml.train_test_split', 0.8)
        self.lookback_days = config.get('ml.lookback_days', 5)
        
        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        self.logger.info(f"Initialized ML trainer with {self.model_type} model")
    
    def _create_model(self):
        """Create ML model based on configuration"""
        if self.model_type == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            )
        else:
            self.logger.warning(f"Unknown model type {self.model_type}, using decision tree")
            return DecisionTreeClassifier(max_depth=10, random_state=42)
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable for ML training
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            Tuple of (features_df, target_series)
        """
        try:
            # Check if data has minimum required length
            if len(data) < 100:
                self.logger.warning(f"Insufficient data for ML training: {len(data)} samples (minimum 100 required)")
                return pd.DataFrame(), pd.Series()
            
            # Add all technical indicators
            df = add_all_indicators(data, self.config)
            
            # Check if indicators were added successfully
            if df.empty or len(df) < 50:
                self.logger.warning("Failed to add technical indicators or insufficient data after adding indicators")
                return pd.DataFrame(), pd.Series()
            
            # Create target variable (1 if next day price goes up, 0 otherwise)
            df['next_close'] = df['close'].shift(-1)
            df['target'] = (df['next_close'] > df['close']).astype(int)
            
            # Create additional features
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            
            # Safe ratio calculations with fallback
            if 'ma_20' in df.columns:
                df['close_ma20_ratio'] = df['close'] / df['ma_20'].replace(0, np.nan)
            if 'ma_50' in df.columns:
                df['close_ma50_ratio'] = df['close'] / df['ma_50'].replace(0, np.nan)
            
            # Create lagged features
            for lag in range(1, min(self.lookback_days + 1, 6)):  # Limit to prevent too many features
                if 'rsi' in df.columns:
                    df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
                df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
                df[f'volume_change_lag_{lag}'] = df['volume_change'].shift(lag)
            
            # Select feature columns - only use what's available
            feature_cols = []
            
            # Add basic features that should exist
            basic_features = ['rsi', 'ma_20', 'ma_50', 'macd', 'volume_sma']
            for feature in basic_features:
                if feature in df.columns:
                    feature_cols.append(feature)
            
            # Add additional features if they exist
            additional_features = [
                'price_change', 'volume_change', 'high_low_ratio',
                'close_ma20_ratio', 'close_ma50_ratio', 'stoch_k', 'stoch_d',
                'bb_upper', 'bb_lower', 'atr', 'macd_signal', 'macd_histogram'
            ]
            
            for feature in additional_features:
                if feature in df.columns:
                    feature_cols.append(feature)
            
            # Add lagged features that exist
            for lag in range(1, min(self.lookback_days + 1, 6)):
                for base_feature in ['rsi', 'price_change', 'volume_change']:
                    lag_feature = f'{base_feature}_lag_{lag}'
                    if lag_feature in df.columns:
                        feature_cols.append(lag_feature)
            
            # Remove duplicates
            feature_cols = list(set(feature_cols))
            
            if not feature_cols:
                self.logger.warning("No valid features found for ML training")
                return pd.DataFrame(), pd.Series()
            
            self.feature_columns = feature_cols
            
            # Create feature matrix
            features_df = df[feature_cols].copy()
            target_series = df['target'].copy()
            
            # Replace infinite values with NaN
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            
            # Remove rows with NaN values
            valid_mask = ~(features_df.isnull().any(axis=1) | target_series.isnull())
            features_df = features_df[valid_mask]
            target_series = target_series[valid_mask]
            
            if len(features_df) < 50:
                self.logger.warning(f"Insufficient clean data for ML training: {len(features_df)} samples (minimum 50 required)")
                return pd.DataFrame(), pd.Series()
            
            self.logger.info(f"Prepared {len(features_df)} samples with {len(feature_cols)} features")
            self.logger.info(f"Features used: {feature_cols}")
            
            return features_df, target_series
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame(), pd.Series()
    
    def train_and_predict(self, data: pd.DataFrame, symbol: str) -> Tuple[float, List[int]]:
        """
        Train model and make predictions
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            Tuple of (accuracy, predictions_list)
        """
        try:
            self.logger.info(f"Training ML model for {symbol}")
            
            # Prepare features and target
            features_df, target_series = self.prepare_features(data)
            
            if features_df.empty or len(features_df) < 50:
                self.logger.warning(f"Insufficient data for ML training: {len(features_df)} samples")
                return 0.5, []
            
            # Split data
            split_idx = int(len(features_df) * self.train_test_split_ratio)
            
            X_train = features_df.iloc[:split_idx]
            X_test = features_df.iloc[split_idx:]
            y_train = target_series.iloc[:split_idx]
            y_test = target_series.iloc[split_idx:]
            
            if len(X_test) < 5:
                self.logger.warning("Insufficient test data, using smaller train/test split")
                X_train, X_test, y_train, y_test = train_test_split(
                    features_df, target_series, test_size=0.3, random_state=42, stratify=target_series
                )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train model
            self.model = self._create_model()
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Generate predictions for all data (for latest predictions)
            all_predictions = self.model.predict(self.scaler.transform(features_df))
            
            # Get feature importance (if available)
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                self.logger.info(f"Top 5 features for {symbol}: {top_features}")
            
            self.logger.info(f"ML model trained for {symbol} with accuracy: {accuracy:.2%}")
            
            return accuracy, all_predictions.tolist()
            
        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {e}")
            return 0.5, []
    
    def predict_next_day(self, data: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict next day movement for the latest data
        
        Args:
            data: DataFrame with latest OHLCV data
            
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            if self.model is None:
                self.logger.error("Model not trained yet")
                return 0, 0.5
            
            # Prepare features
            features_df, _ = self.prepare_features(data)
            
            if features_df.empty:
                return 0, 0.5
            
            # Get latest features
            latest_features = features_df.iloc[-1:][self.feature_columns]
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Make prediction
            prediction = self.model.predict(latest_features_scaled)[0]
            
            # Get prediction probability (confidence)
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(latest_features_scaled)[0]
                confidence = max(proba)
            else:
                confidence = 0.6  # Default confidence for non-probabilistic models
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return 0, 0.5
    
    def evaluate_model(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Evaluate model performance with detailed metrics
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Prepare features and target
            features_df, target_series = self.prepare_features(data)
            
            if features_df.empty:
                return {'accuracy': 0.5, 'error': 'No data available'}
            
            # Split data
            split_idx = int(len(features_df) * self.train_test_split_ratio)
            
            X_test = features_df.iloc[split_idx:]
            y_test = target_series.iloc[split_idx:]
            
            if self.model is None:
                return {'accuracy': 0.5, 'error': 'Model not trained'}
            
            # Scale test features
            X_test_scaled = self.scaler.transform(X_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            evaluation = {
                'symbol': symbol,
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'confusion_matrix': cm.tolist(),
                'total_samples': len(y_test),
                'positive_samples': sum(y_test),
                'negative_samples': len(y_test) - sum(y_test)
            }
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating model for {symbol}: {e}")
            return {'accuracy': 0.5, 'error': str(e)}
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        try:
            import joblib
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'config': {
                    'model_type': self.model_type,
                    'features': self.features,
                    'lookback_days': self.lookback_days
                }
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        try:
            import joblib
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")