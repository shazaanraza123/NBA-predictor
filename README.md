# NBA Game Winner Predictor

A machine learning-based web application that predicts the winner of NBA games using historical and real-time data.

## Features

- Real-time data collection from multiple sources
- Machine learning model using XGBoost
- Modern web interface with interactive visualizations
- Support for all NBA teams
- Probability-based predictions

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nba-predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Select the home and away teams from the dropdown menus and click "Predict Winner" to get the prediction.

## Project Structure

- `app.py`: Flask web application
- `data_collector.py`: Data collection and preprocessing
- `model_trainer.py`: Machine learning model training and prediction
- `templates/index.html`: Web interface template
- `requirements.txt`: Python package dependencies

## Data Sources

The application collects data from:
- Basketball Reference
- ESPN
- NBA API

## Model Features

The prediction model uses the following features:
- Team points per game (PPG)
- Field goal percentage
- 3-point percentage
- Free throw percentage
- Rebounds
- Assists
- Turnovers
- Recent win percentage
- Injury data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 