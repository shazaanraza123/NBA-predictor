from flask import Flask, render_template, request, jsonify
from nba_predictor import NBAPredictor
from data_collector import NBADataCollector
import pandas as pd
import logging
import os
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize once at module level
try:
    predictor = NBAPredictor()
    collector = NBADataCollector()
    logger.info("Successfully initialized predictor and collector")
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    raise

@app.route('/')
def home():
    team_names = [team['full_name'] for team in collector.teams_dict]
    return render_template('index.html', teams=team_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
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
        logger.exception("Error in prediction route")
        return jsonify({'error': str(e)}), 500

def find_available_port(start_port=5002, max_attempts=100):
    """Find an available port"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError("Could not find an available port")

if __name__ == '__main__':
    # Kill existing Flask processes
    os.system('pkill -f flask')
    
    # Find an available port
    port = find_available_port()
    logger.info(f"Starting server on port {port}")
    app.run(debug=True, port=port) 