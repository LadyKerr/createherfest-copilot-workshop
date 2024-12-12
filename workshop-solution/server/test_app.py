import pytest
from app import app
import pandas as pd
from unittest.mock import patch, mock_open

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Test the home route."""
    response = client.get('/')
    assert response.status_code == 200
    assert response.data.decode() == "Let's build an api!"

def test_predict_route_success(client):
    """Test successful prediction."""
    response = client.get('/predict?day=1&airport=14683')
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    assert 'input' in data
    assert data['status'] == 'success'

def test_predict_route_invalid_params(client):
    """Test prediction with invalid parameters."""
    # Test missing parameters
    response = client.get('/predict')
    assert response.status_code == 400
    assert 'error' in response.get_json()

    # Test invalid day
    response = client.get('/predict?day=8&airport=14683')
    assert response.status_code == 400
    assert 'error' in response.get_json()

    # Test invalid airport
    response = client.get('/predict?day=1&airport=999')
    assert response.status_code == 400
    assert 'error' in response.get_json()

@pytest.fixture
def mock_csv_data():
    return """OriginAirportID,OriginAirportName
14683,San Diego International
14771,San Francisco International"""

def test_airports_route_success(client, mock_csv_data):
    """Test successful airports list retrieval."""
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = pd.read_csv(
            pd.StringIO(mock_csv_data)
        )
        response = client.get('/airports')
        assert response.status_code == 200
        data = response.get_json()
        assert 'airports' in data
        assert 'count' in data
        assert data['status'] == 'success'
        assert len(data['airports']) == 2

def test_airports_route_file_not_found(client):
    """Test airports route when CSV file is not found."""
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.side_effect = FileNotFoundError
        response = client.get('/airports')
        assert response.status_code == 500
        assert response.get_json()['error'] == 'CSV file not found'

def test_airports_route_corrupted_file(client):
    """Test airports route with corrupted CSV file."""
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.side_effect = pd.errors.ParserError
        response = client.get('/airports')
        assert response.status_code == 500
        assert response.get_json()['error'] == 'CSV file is corrupted'
