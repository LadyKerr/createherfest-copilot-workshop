import pickle
import warnings
import logging
import pandas as pd
from typing import Tuple, Dict, List
from http import HTTPStatus
from flask import Flask, jsonify, request

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model from pickle file
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Let's build an api!"

def validate_inputs(day: int, airport: int) -> Tuple[bool, str]:
    """Validate input parameters."""
    if not (1 <= day <= 7):
        return False, "Day must be between 1 and 7"
    if not (10000 <= airport <= 99999):
        return False, "Invalid airport ID"
    return True, ""

@app.route('/predict', methods=['GET'])
def predict() -> Dict:
    """
    Predict flight delay probability.
    
    Parameters:
        day (int): Day of week (1-7)
        airport (int): Airport ID
    
    Returns:
        JSON response with prediction results
    """
    try:
        # Get and validate parameters
        day = request.args.get('day', type=int)
        airport = request.args.get('airport', type=int)
        
        if not all([day, airport]):
            return jsonify({
                'error': 'Missing required parameters'
            }), HTTPStatus.BAD_REQUEST
            
        is_valid, error_msg = validate_inputs(day, airport)
        if not is_valid:
            return jsonify({'error': error_msg}), HTTPStatus.BAD_REQUEST

        # Make prediction
        logger.info(f"Making prediction for day: {day}, airport: {airport}")
        prediction = model.predict_proba([[day, airport]])[0]
        
        # Process prediction values
        values = str(prediction).split(' ')
        certainty = float(values[0][1:])
        delayed = float(values[1][:-1])
        
        response = {
            'prediction': {
                'certainty': round(certainty, 4),
                'delayed_percentage': round(delayed, 4)
            },
            'input': {
                'day': day,
                'airport': airport
            },
            'status': 'success'
        }
        
        logger.info(f"Prediction successful: {response}")
        return jsonify(response), HTTPStatus.OK
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), HTTPStatus.INTERNAL_SERVER_ERROR

@app.route('/airports', methods=['GET'])
def get_airports() -> List[Dict]:
    """
    Get list of all airports.
    
    Returns:
        JSON response with list of airports
    """
    try:
        # Read airports CSV file
        airports_df = pd.read_csv('origin_airport.csv')
        airports = airports_df.to_dict('records')
        
        return jsonify({
            'airports': airports,
            'count': len(airports),
            'status': 'success'
        }), HTTPStatus.OK
        
    except Exception as e:
        logger.error(f"Failed to get airports: {str(e)}")
        return jsonify({
            'error': 'Failed to get airports',
            'message': str(e)
        }), HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    app.run(debug=True)