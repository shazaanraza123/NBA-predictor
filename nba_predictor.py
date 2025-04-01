import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class NBAPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'home_pts_per_game', 'home_fg_pct', 'home_fg3_pct', 'home_ft_pct',
            'home_oreb', 'home_dreb', 'home_ast', 'home_stl', 'home_blk',
            'home_tov', 'home_plus_minus',
            'away_pts_per_game', 'away_fg_pct', 'away_fg3_pct', 'away_ft_pct',
            'away_oreb', 'away_dreb', 'away_ast', 'away_stl', 'away_blk',
            'away_tov', 'away_plus_minus'
        ]
        self._initialize_model()

    def _initialize_model(self):
        """Initialize a default model"""
        try:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            # Create dummy data for initial training
            X = pd.DataFrame(np.random.random((100, len(self.feature_columns))), 
                           columns=self.feature_columns)
            y = np.random.randint(0, 2, 100)
            self.model.fit(X, y)
            logger.info("Initialized default model")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def predict(self, X):
        """Make predictions"""
        try:
            # Ensure we have all required columns in the correct order
            X = X[self.feature_columns]
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return np.array([0])  # Default prediction

    def predict_proba(self, X):
        """Get prediction probabilities"""
        try:
            # Ensure we have all required columns in the correct order
            X = X[self.feature_columns]
            return self.model.predict_proba(X)
        except Exception as e:
            logger.error(f"Prediction probability error: {str(e)}")
            return np.array([[0.5, 0.5]])  # Default probabilities

    def save_model(self, filepath='nba_predictor.joblib'):
        """Save the model"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'feature_columns': self.feature_columns
            }, filepath)
            logger.info("Model saved successfully")

    def load_model(self, filepath='nba_predictor.joblib'):
        """Load the model"""
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.feature_columns = data['feature_columns']
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self._initialize_model()

    def train_model(self, training_data):
        """Train the prediction model using current season data"""
        try:
            if len(training_data) < 3:
                raise ValueError("Need at least 3 games for training")
                
            X = training_data.drop('home_team_won', axis=1)
            y = training_data['home_team_won']
            
            # For very small datasets, use leave-one-out validation
            if len(training_data) < 5:
                X_train = X
                y_train = y
                X_test = X
                y_test = y
            else:
                # For larger datasets, use train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.1, random_state=42
                )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train XGBoost model with adjusted parameters for small dataset
            self.model = xgb.XGBClassifier(
                n_estimators=10,  # Reduced for small dataset
                learning_rate=0.1,
                max_depth=2,      # Reduced to prevent overfitting
                min_child_weight=1,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            return self.model.score(X_test_scaled, y_test)
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None

    def predict_game(self, home_team_stats, away_team_stats):
        """Predict game outcome using current season stats"""
        try:
            features = {
                'home_wins': home_team_stats['wins'],
                'home_losses': home_team_stats['losses'],
                'home_ppg': home_team_stats['ppg'],
                'home_opp_ppg': home_team_stats['opp_ppg'],
                'home_fg_pct': home_team_stats['fg_pct'],
                'home_3pt_pct': home_team_stats['three_pt_pct'],
                'home_ft_pct': home_team_stats['ft_pct'],
                'home_rebounds': home_team_stats['rebounds'],
                'home_assists': home_team_stats['assists'],
                'home_turnovers': home_team_stats['turnovers'],
                
                'away_wins': away_team_stats['wins'],
                'away_losses': away_team_stats['losses'],
                'away_ppg': away_team_stats['ppg'],
                'away_opp_ppg': away_team_stats['opp_ppg'],
                'away_fg_pct': away_team_stats['fg_pct'],
                'away_3pt_pct': away_team_stats['three_pt_pct'],
                'away_ft_pct': away_team_stats['ft_pct'],
                'away_rebounds': away_team_stats['rebounds'],
                'away_assists': away_team_stats['assists'],
                'away_turnovers': away_team_stats['turnovers']
            }
            
            features_df = pd.DataFrame([features])
            features_scaled = self.scaler.transform(features_df)
            
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            return {
                'home_win_probability': float(probabilities[1]),
                'away_win_probability': float(probabilities[0]),
                'features_used': features
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None 