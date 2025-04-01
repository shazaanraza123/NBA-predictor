import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NBAModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = [
            'home_ppg', 'away_ppg', 'home_fg_pct', 'away_fg_pct',
            'home_3pt_pct', 'away_3pt_pct', 'home_ft_pct', 'away_ft_pct',
            'home_reb', 'away_reb', 'home_ast', 'away_ast',
            'home_tov', 'away_tov', 'home_recent_win_pct', 'away_recent_win_pct',
            'home_injuries', 'away_injuries'
        ]

    def preprocess_data(self, data):
        """Preprocess the input data."""
        try:
            # Convert data to DataFrame if it's not already
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Ensure all required features are present
            missing_features = set(self.feature_columns) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Handle missing values
            data = data.fillna(data.mean())
            
            # Scale features
            scaled_data = self.scaler.fit_transform(data[self.feature_columns])
            return pd.DataFrame(scaled_data, columns=self.feature_columns)
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def train_model(self, X, y, model_type='xgboost'):
        """Train the model using the provided data."""
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Preprocess the data
            X_train_scaled = self.preprocess_data(X_train)
            X_test_scaled = self.preprocess_data(X_test)
            
            # Initialize and train the model
            if model_type == 'xgboost':
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            else:  # Random Forest
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate the model
            y_pred = self.model.predict(X_test_scaled)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            logger.info(f"Model performance metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict_game(self, game_features):
        """Predict the winner of a game using the trained model."""
        try:
            if self.model is None:
                # If no trained model exists, use a simple probability based on recent win percentages
                home_win_pct = game_features.get('home_recent_win_pct', 0.5)
                away_win_pct = game_features.get('away_recent_win_pct', 0.5)
                
                # Add home court advantage (typically around 60% in NBA)
                home_advantage = 0.1
                home_prob = (home_win_pct + home_advantage) / (home_win_pct + away_win_pct + home_advantage)
                away_prob = 1 - home_prob
                
                return {
                    'home_win_probability': float(home_prob),
                    'away_win_probability': float(away_prob),
                    'predicted_winner': 'Home' if home_prob > away_prob else 'Away',
                    'confidence': abs(home_prob - away_prob) * 100  # Confidence as percentage difference
                }
                
            # If model exists, use it for prediction
            processed_features = self.preprocess_data(game_features)
            prediction = self.model.predict_proba(processed_features)[0]
            
            return {
                'home_win_probability': float(prediction[1]),
                'away_win_probability': float(prediction[0]),
                'predicted_winner': 'Home' if prediction[1] > prediction[0] else 'Away',
                'confidence': abs(prediction[1] - prediction[0]) * 100  # Confidence as percentage difference
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def save_model(self, model_path='nba_predictor.joblib'):
        """Save the trained model and scaler."""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path='nba_predictor.joblib'):
        """Load a trained model and scaler."""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 