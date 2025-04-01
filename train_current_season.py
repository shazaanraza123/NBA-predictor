from data_collector import NBADataCollector
from nba_predictor import NBAPredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize data collector
    collector = NBADataCollector()
    
    # Get training data
    logger.info("Collecting and preparing training data...")
    training_data = collector.prepare_game_data()
    
    if training_data is not None and not training_data.empty:
        logger.info(f"Collected {len(training_data)} games for training")
        
        # Initialize and train predictor
        predictor = NBAPredictor()
        accuracy = predictor.train_model(training_data)
        
        if accuracy:
            logger.info(f"Model trained successfully with accuracy: {accuracy:.2f}")
            predictor.save_model()
            logger.info("Model saved successfully")
        else:
            logger.error("Failed to train model")
    else:
        logger.error("Failed to collect training data")

if __name__ == "__main__":
    main() 