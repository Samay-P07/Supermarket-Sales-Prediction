"""
ProfitPulse: Supermarket Sales Prediction System
Production-grade Flask Web Application & REST Microservice
"""

import os
import io
import time
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify, send_file, Response

# Base Directory & Metadata
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics.json')

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Cache for loaded models
LOADED_MODELS = {}
MODEL_METRICS = {}

def load_all_artifacts():
    """Load machine learning pipelines and metadata into memory."""
    global LOADED_MODELS, MODEL_METRICS
    
    # Load metrics metadata
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as f:
                MODEL_METRICS = json.load(f)
        except Exception as e:
            print(f"[!] Warning: Could not load metrics.json: {e}")
            MODEL_METRICS = {}
    
    # Map friendly UI names to joblib file paths
    model_mapping = {
        'xgboost': 'xgboost.joblib',
        'random forest regressor': 'random_forest.joblib',
        'gradient boosting': 'gradient_boosting.joblib',
        'ridge regression': 'ridge_regression.joblib',
        'lasso regression': 'lasso_regression.joblib',
        'support vector machine (svm)': 'support_vector_machine_svm.joblib',
        'k-nearest neighbors (knn)': 'k-nearest_neighbors_knn.joblib',
        'best': 'best_model.joblib'
    }
    
    for key, filename in model_mapping.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            try:
                LOADED_MODELS[key] = joblib.load(filepath)
            except Exception as e:
                print(f"[!] Warning: Failed loading model {filename}: {e}")
        else:
            print(f"[!] Model file missing: {filepath}")

# Load models on startup
load_all_artifacts()

def get_active_model(model_name: str = 'best'):
    """Return model pipeline instance or default best model."""
    normalized_name = (model_name or 'best').strip().lower()
    if normalized_name in LOADED_MODELS:
        return LOADED_MODELS[normalized_name], normalized_name
    # Fallback to best or first available
    if 'best' in LOADED_MODELS:
        return LOADED_MODELS['best'], 'best (xgboost)'
    if LOADED_MODELS:
        first_key = next(iter(LOADED_MODELS))
        return LOADED_MODELS[first_key], first_key
    raise RuntimeError("No machine learning models available. Please run train_models.py first.")

def calculate_financials(unit_price: float, quantity: int, predicted_total: float):
    """Compute retail financial breakdowns (COGS, Tax, Gross Margin)."""
    cogs = float(unit_price * quantity)
    tax_5pct = float(cogs * 0.05)
    gross_income = tax_5pct
    gross_margin_pct = 4.7619
    
    return {
        'estimated_cogs': round(cogs, 2),
        'estimated_tax': round(tax_5pct, 2),
        'estimated_gross_income': round(gross_income, 2),
        'gross_margin_pct': gross_margin_pct,
        'effective_total': round(predicted_total, 2)
    }

# ==========================================================
# Web UI Routes
# ==========================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start")
def start():
    return render_template("start.html")

@app.route("/about")
def about():
    return render_template("about.html", metrics=MODEL_METRICS.get('metrics', {}))

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", metrics=MODEL_METRICS.get('metrics', {}))
    
    try:
        # Form inputs
        branch = request.form.get("Branch", "A")
        city = request.form.get("City", "Yangon")
        customer_type = request.form.get("Customer type", "Member")
        gender = request.form.get("Gender", "Female")
        product_line = request.form.get("Product line", "Electronic accessories")
        payment = request.form.get("Payment", "Ewallet")
        
        unit_price = float(request.form.get("Unit price", 50.0))
        quantity = int(request.form.get("Quantity", 1))
        rating = float(request.form.get("Rating", 7.0))
        
        # Temporal features (defaults to current time if not provided)
        now = datetime.now()
        hour = int(request.form.get("Hour", now.hour))
        day_of_week = int(request.form.get("DayOfWeek", now.weekday()))
        month = int(request.form.get("Month", now.month))
        is_weekend = 1 if day_of_week >= 5 else 0
        
        selected_model_name = request.form.get("model_name", "xgboost")
        model, resolved_name = get_active_model(selected_model_name)
        
        # Build inference DataFrame
        features_dict = {
            'Branch': [branch],
            'City': [city],
            'Customer type': [customer_type],
            'Gender': [gender],
            'Product line': [product_line],
            'Payment': [payment],
            'Unit price': [unit_price],
            'Quantity': [quantity],
            'Rating': [rating],
            'Hour': [hour],
            'DayOfWeek': [day_of_week],
            'Month': [month],
            'IsWeekend': [is_weekend]
        }
        input_df = pd.DataFrame(features_dict)
        
        # Execute prediction through scikit-learn pipeline
        prediction = float(model.predict(input_df)[0])
        financials = calculate_financials(unit_price, quantity, prediction)
        
        transaction_summary = {
            'model_name': resolved_name.title(),
            'branch': branch,
            'city': city,
            'customer_type': customer_type,
            'gender': gender,
            'product_line': product_line,
            'payment': payment,
            'unit_price': unit_price,
            'quantity': quantity,
            'rating': rating,
            'hour': f"{hour:02d}:00"
        }
        
        return render_template(
            "result.html",
            prediction=prediction,
            financials=financials,
            transaction=transaction_summary
        )
        
    except Exception as e:
        return render_template(
            "predict.html",
            error_message=f"Prediction error: {str(e)}",
            metrics=MODEL_METRICS.get('metrics', {})
        )

@app.route("/dashboard")
def dashboard():
    """Interactive Analytics Dashboard page."""
    return render_template("dashboard.html", metrics=MODEL_METRICS)

@app.route("/benchmarks")
def benchmarks():
    """Model comparison and performance leaderboard page."""
    return render_template("benchmarks.html", metrics=MODEL_METRICS)

@app.route("/batch", methods=["GET", "POST"])
def batch_predict_page():
    """Batch prediction page allowing CSV uploads."""
    if request.method == "GET":
        return render_template("batch.html")
    
    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("batch.html", error_message="Please upload a valid CSV file.")
    
    try:
        df_input = pd.read_csv(file)
        
        # Ensure required columns exist, impute missing temporal columns if necessary
        req_cols = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment', 'Unit price', 'Quantity', 'Rating']
        missing = [c for c in req_cols if c not in df_input.columns]
        if missing:
            return render_template("batch.html", error_message=f"CSV is missing required columns: {missing}")
        
        # Impute temporal features if missing
        now = datetime.now()
        if 'Hour' not in df_input.columns:
            if 'Time' in df_input.columns:
                df_input['Hour'] = pd.to_datetime(df_input['Time'], format='%H:%M', errors='coerce').dt.hour.fillna(now.hour)
            else:
                df_input['Hour'] = now.hour
                
        if 'Month' not in df_input.columns or 'DayOfWeek' not in df_input.columns:
            if 'Date' in df_input.columns:
                d_series = pd.to_datetime(df_input['Date'], errors='coerce')
                df_input['Month'] = d_series.dt.month.fillna(now.month)
                df_input['DayOfWeek'] = d_series.dt.dayofweek.fillna(now.weekday())
            else:
                df_input['Month'] = now.month
                df_input['DayOfWeek'] = now.weekday()
                
        df_input['IsWeekend'] = df_input['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Perform prediction with best model
        model, _ = get_active_model('best')
        df_clean = df_input[['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment', 
                             'Unit price', 'Quantity', 'Rating', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend']]
        
        preds = model.predict(df_clean)
        df_input['Predicted_Total'] = np.round(preds, 2)
        df_input['Estimated_COGS'] = np.round(df_input['Unit price'] * df_input['Quantity'], 2)
        df_input['Estimated_Tax_5%'] = np.round(df_input['Estimated_COGS'] * 0.05, 2)
        df_input['Estimated_Gross_Income'] = df_input['Estimated_Tax_5%']
        
        # Convert first 15 rows to preview dictionary
        preview_data = df_input.head(15).to_dict(orient='records')
        total_predicted_sum = round(float(preds.sum()), 2)
        avg_predicted_val = round(float(preds.mean()), 2)
        
        return render_template(
            "batch.html",
            preview_data=preview_data,
            total_records=len(df_input),
            total_predicted_sum=total_predicted_sum,
            avg_predicted_val=avg_predicted_val,
            filename=file.filename
        )
        
    except Exception as e:
        return render_template("batch.html", error_message=f"Failed to process CSV: {str(e)}")

@app.route("/download-sample-csv")
def download_sample_csv():
    """Provide a template CSV for batch predictions."""
    sample_data = {
        'Branch': ['A', 'C', 'B'],
        'City': ['Yangon', 'Naypyitaw', 'Mandalay'],
        'Customer type': ['Member', 'Normal', 'Member'],
        'Gender': ['Female', 'Male', 'Female'],
        'Product line': ['Electronic accessories', 'Health and beauty', 'Food and beverages'],
        'Payment': ['Ewallet', 'Cash', 'Credit card'],
        'Unit price': [74.50, 45.20, 19.80],
        'Quantity': [5, 3, 8],
        'Rating': [8.5, 7.2, 9.1],
        'Hour': [14, 18, 11],
        'DayOfWeek': [1, 4, 6],
        'Month': [3, 2, 1]
    }
    df_sample = pd.DataFrame(sample_data)
    csv_bytes = df_sample.to_csv(index=False).encode('utf-8')
    
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=supermarket_sales_sample.csv"}
    )

# ==========================================================
# REST API Endpoints (Production ML Microservice)
# ==========================================================

@app.route("/api/v1/health", methods=["GET"])
def api_health():
    """Health check endpoint for Docker/K8s."""
    return jsonify({
        "status": "healthy",
        "service": "ProfitPulse ML Inference Engine",
        "version": "2.0.0",
        "available_models": list(LOADED_MODELS.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@app.route("/api/v1/models", methods=["GET"])
def api_models():
    """Return model performance metrics and features metadata."""
    return jsonify({
        "status": "success",
        "metadata": MODEL_METRICS
    })

@app.route("/api/v1/predict", methods=["POST"])
def api_predict():
    """REST JSON Inference endpoint."""
    start_time = time.time()
    
    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    
    data = request.get_json()
    
    # Required attributes
    required_fields = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment', 'Unit price', 'Quantity', 'Rating']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required parameter '{field}'"}), 422
    
    try:
        now = datetime.now()
        hour = int(data.get('Hour', now.hour))
        day_of_week = int(data.get('DayOfWeek', now.weekday()))
        month = int(data.get('Month', now.month))
        is_weekend = 1 if day_of_week >= 5 else 0
        
        unit_price = float(data['Unit price'])
        quantity = int(data['Quantity'])
        rating = float(data['Rating'])
        
        model_name = data.get('model_name', 'best')
        model, resolved_name = get_active_model(model_name)
        
        input_df = pd.DataFrame([{
            'Branch': str(data['Branch']),
            'City': str(data['City']),
            'Customer type': str(data['Customer type']),
            'Gender': str(data['Gender']),
            'Product line': str(data['Product line']),
            'Payment': str(data['Payment']),
            'Unit price': unit_price,
            'Quantity': quantity,
            'Rating': rating,
            'Hour': hour,
            'DayOfWeek': day_of_week,
            'Month': month,
            'IsWeekend': is_weekend
        }])
        
        pred = float(model.predict(input_df)[0])
        financials = calculate_financials(unit_price, quantity, pred)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        return jsonify({
            "status": "success",
            "model_used": resolved_name,
            "predicted_total": round(pred, 2),
            "financial_breakdown": financials,
            "latency_ms": latency_ms
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
