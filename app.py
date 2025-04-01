from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from nba_predictor import NBAPredictor
from data_collector import NBADataCollector
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize predictor and collector
try:
    predictor = NBAPredictor()
    collector = NBADataCollector()
    predictor.load_model()  # Load the model if it exists
    logger.info("Successfully initialized predictor and collector")
except Exception as e:
    logger.error(f"Error initializing predictor: {str(e)}")
    predictor = None
    collector = None

@app.route('/')
def home():
    try:
        team_names = [team['full_name'] for team in collector.teams_dict]
        return render_template('index.html', teams=team_names)
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not predictor:
            return jsonify({"error": "Predictor not initialized"}), 500
        
        logger.info(f"Received prediction request: {data}")
        
        home_team_name = data.get('home_team')
        away_team_name = data.get('away_team')
        
        if not home_team_name or not away_team_name:
            return jsonify({'error': 'Please select both teams'}), 400

        # Get team IDs and stats
        home_team_id = collector.get_team_id(home_team_name)
        away_team_id = collector.get_team_id(away_team_name)
        
        if not home_team_id or not away_team_id:
            return jsonify({'error': 'Invalid team name(s)'}), 400

        # Get team stats
        home_stats = collector.get_team_stats(home_team_id)
        away_stats = collector.get_team_stats(away_team_id)
        
        if not home_stats or not away_stats:
            return jsonify({'error': 'Could not fetch team statistics'}), 500

        # Convert numpy values to Python native types
        home_stats = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                     for k, v in home_stats.items()}
        away_stats = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                     for k, v in away_stats.items()}

        # Create feature dictionary
        features = pd.DataFrame([{
            'home_pts_per_game': float(home_stats['pts_per_game']),
            'home_fg_pct': float(home_stats['fg_pct']),
            'home_fg3_pct': float(home_stats['fg3_pct']),
            'home_ft_pct': float(home_stats['ft_pct']),
            'home_oreb': float(home_stats['oreb']),
            'home_dreb': float(home_stats['dreb']),
            'home_ast': float(home_stats['ast']),
            'home_stl': float(home_stats['stl']),
            'home_blk': float(home_stats['blk']),
            'home_tov': float(home_stats['tov']),
            'home_plus_minus': float(home_stats['plus_minus']),
            
            'away_pts_per_game': float(away_stats['pts_per_game']),
            'away_fg_pct': float(away_stats['fg_pct']),
            'away_fg3_pct': float(away_stats['fg3_pct']),
            'away_ft_pct': float(away_stats['ft_pct']),
            'away_oreb': float(away_stats['oreb']),
            'away_dreb': float(away_stats['dreb']),
            'away_ast': float(away_stats['ast']),
            'away_stl': float(away_stats['stl']),
            'away_blk': float(away_stats['blk']),
            'away_tov': float(away_stats['tov']),
            'away_plus_minus': float(away_stats['plus_minus'])
        }])

        # Make prediction
        prediction = predictor.predict(features)
        probabilities = predictor.predict_proba(features)

        # Create response with native Python types
        response = {
            'winner': home_team_name if prediction[0] == 1 else away_team_name,
            'win_probability': float(probabilities[0][1] if prediction[0] == 1 else probabilities[0][0]) * 100,
            'home_stats': home_stats,
            'away_stats': away_stats,
            'key_factors': {
                'home_advantage': bool(home_stats['home_win_pct'] > away_stats['away_win_pct']),
                'better_scoring': bool(home_stats['pts_per_game'] > away_stats['pts_per_game']),
                'better_defense': bool(home_stats['plus_minus'] > away_stats['plus_minus']),
                'better_form': bool(home_stats['recent_form'] > away_stats['recent_form'])
            }
        }

        logger.info(f"Prediction response: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({"error": "Error making prediction"}), 500

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port) 