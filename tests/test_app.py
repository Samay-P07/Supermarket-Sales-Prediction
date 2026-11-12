"""
Unit tests for ProfitPulse Flask web application and REST API endpoints.
"""

import pytest
import json
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test API health check returns 200 OK and valid status."""
    res = client.get('/api/v1/health')
    assert res.status_code == 200
    data = res.get_json()
    assert data['status'] == 'healthy'
    assert 'version' in data
    assert 'available_models' in data

def test_models_metadata_endpoint(client):
    """Test metadata API endpoint returns models and metrics."""
    res = client.get('/api/v1/models')
    assert res.status_code == 200
    data = res.get_json()
    assert data['status'] == 'success'
    assert 'metadata' in data

def test_home_page(client):
    """Test landing page renders successfully."""
    res = client.get('/')
    assert res.status_code == 200
    assert b"Profit" in res.data

def test_predict_page_get(client):
    """Test prediction form renders."""
    res = client.get('/predict')
    assert res.status_code == 200
    assert b"Forecasting Studio" in res.data or b"Prediction" in res.data

def test_dashboard_page(client):
    """Test analytics dashboard renders."""
    res = client.get('/dashboard')
    assert res.status_code == 200
    assert b"Supermarket Business Analytics" in res.data

def test_benchmarks_page(client):
    """Test model comparison leaderboard renders."""
    res = client.get('/benchmarks')
    assert res.status_code == 200
    assert b"Model Performance" in res.data

def test_api_predict_success(client):
    """Test JSON prediction API endpoint with valid transaction payload."""
    payload = {
        "Branch": "A",
        "City": "Yangon",
        "Customer type": "Member",
        "Gender": "Female",
        "Product line": "Electronic accessories",
        "Payment": "Ewallet",
        "Unit price": 74.50,
        "Quantity": 5,
        "Rating": 8.5,
        "Hour": 14,
        "DayOfWeek": 2,
        "Month": 3,
        "model_name": "xgboost"
    }
    res = client.post('/api/v1/predict', 
                      data=json.dumps(payload),
                      content_type='application/json')
    assert res.status_code == 200
    data = res.get_json()
    assert data['status'] == 'success'
    assert 'predicted_total' in data
    assert data['predicted_total'] > 0
    assert 'financial_breakdown' in data
    assert 'estimated_cogs' in data['financial_breakdown']

def test_api_predict_missing_fields(client):
    """Test API error handling when required fields are missing."""
    payload = {"Branch": "A", "Unit price": 50.0}
    res = client.post('/api/v1/predict', 
                      data=json.dumps(payload),
                      content_type='application/json')
    assert res.status_code == 422
    data = res.get_json()
    assert 'error' in data

def test_form_prediction_post(client):
    """Test Web form POST prediction."""
    form_data = {
        "Branch": "C",
        "City": "Naypyitaw",
        "Customer type": "Normal",
        "Gender": "Male",
        "Product line": "Sports and travel",
        "Payment": "Credit card",
        "Unit price": "45.00",
        "Quantity": "3",
        "Rating": "7.5",
        "Hour": "16",
        "model_name": "random forest regressor"
    }
    res = client.post('/predict', data=form_data)
    assert res.status_code == 200
    assert b"Estimated Transaction Value" in res.data
