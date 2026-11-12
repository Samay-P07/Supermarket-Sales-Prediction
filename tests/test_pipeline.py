"""
Unit tests for data preprocessing and ML model inference pipelines.
"""

import pytest
import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

def test_best_model_exists_and_loads():
    """Verify that the primary serialized model artifact is present and functional."""
    best_model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    assert os.path.exists(best_model_path), "best_model.joblib does not exist"
    model = joblib.load(best_model_path)
    assert hasattr(model, 'predict'), "Model artifact does not have a predict method"

def test_inference_pipeline_predictions():
    """Verify inference pipeline processes tabular input correctly."""
    best_model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    model = joblib.load(best_model_path)
    
    sample_data = pd.DataFrame([{
        'Branch': 'B',
        'City': 'Mandalay',
        'Customer type': 'Member',
        'Gender': 'Male',
        'Product line': 'Health and beauty',
        'Payment': 'Cash',
        'Unit price': 50.0,
        'Quantity': 4,
        'Rating': 8.0,
        'Hour': 15,
        'DayOfWeek': 1,
        'Month': 2,
        'IsWeekend': 0
    }])
    
    pred = model.predict(sample_data)
    assert len(pred) == 1
    # For price 50, qty 4, total is around 50*4*1.05 = 210.0
    assert 180 <= pred[0] <= 240, f"Predicted value {pred[0]} outside expected realistic range"
