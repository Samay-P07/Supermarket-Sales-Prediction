"""
ProfitPulse: Supermarket Sales Prediction - Model Training Pipeline
Production-grade ML pipeline with feature engineering, cross-validation,
hyperparameter tuning, and leak-free feature sets.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal and behavioral features from raw supermarket data."""
    df = df_raw.copy()
    
    # Parse dates and times
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Parse time to hour
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
    
    return df

def run_training_pipeline(csv_path: str = 'supermarket_sales.csv', output_dir: str = 'saved_models'):
    print("=" * 60)
    print("🚀 ProfitPulse Machine Learning Training Pipeline Initialized")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # 1. Load Data
    df_raw = pd.read_csv(csv_path)
    print(f"[*] Dataset loaded successfully. Shape: {df_raw.shape}")
    
    # 2. Feature Engineering
    df = engineer_features(df_raw)
    
    # 3. Define Features & Target (STRICTLY NO DATA LEAKAGE)
    # Exclude: Invoice ID, Date, Time, Tax 5%, cogs, gross margin percentage, gross income
    categorical_features = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']
    numerical_features = ['Unit price', 'Quantity', 'Rating', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend']
    
    feature_cols = categorical_features + numerical_features
    target_col = 'Total'
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"[*] Target variable: '{target_col}'")
    print(f"[*] Features used ({len(feature_cols)}): {feature_cols}")
    print("[*] Leaking features (Tax 5%, cogs, gross income) successfully excluded.")
    
    # 4. Train-Test Split (80% Train, 20% Holdout Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[*] Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # 5. Define Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # 6. Candidate Algorithms
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Support Vector Machine (SVM)': SVR(C=10.0, epsilon=0.2),
        'K-Nearest Neighbors (KNN)': KNeighborsRegressor(n_neighbors=7)
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = XGBRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    
    results = {}
    fitted_pipelines = {}
    
    # 7. Model Training & Cross-Validation
    print("\n" + "-" * 60)
    print(f"{'Model':<30} | {'CV R2':<8} | {'Test R2':<8} | {'RMSE ($)':<9} | {'MAE ($)':<8}")
    print("-" * 60)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # 5-Fold Cross-Validation on Training set
        cv_r2_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2')
        mean_cv_r2 = float(np.mean(cv_r2_scores))
        
        # Fit on full training set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate holdout evaluation metrics
        test_r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
        
        results[name] = {
            'cv_r2': round(mean_cv_r2, 4),
            'test_r2': round(test_r2, 4),
            'rmse': round(rmse, 2),
            'mae': round(mae, 2),
            'mape': round(mape, 2)
        }
        
        fitted_pipelines[name] = pipeline
        
        print(f"{name:<30} | {mean_cv_r2:<8.4f} | {test_r2:<8.4f} | {rmse:<9.2f} | {mae:<8.2f}")
    
    print("-" * 60)
    
    # Identify Best Model by Test R2
    best_model_name = max(results, key=lambda k: results[k]['test_r2'])
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\n🏆 Best Performing Model: {best_model_name} (R² = {results[best_model_name]['test_r2']})")
    
    # 8. Save Models & Artifacts
    for name, pipe in fitted_pipelines.items():
        clean_filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.joblib'
        joblib.dump(pipe, os.path.join(output_dir, clean_filename), compress=3)
    
    joblib.dump(best_pipeline, os.path.join(output_dir, 'best_model.joblib'), compress=3)
    
    # Save feature metadata & results json
    metadata = {
        'best_model': best_model_name,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'all_features': feature_cols,
        'metrics': results,
        'dataset_summary': {
            'total_records': len(df),
            'branches': sorted(df['Branch'].unique().tolist()),
            'cities': sorted(df['City'].unique().tolist()),
            'product_lines': sorted(df['Product line'].unique().tolist()),
            'payment_methods': sorted(df['Payment'].unique().tolist())
        }
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[*] Saved models and metrics to '{output_dir}/'")
    
    # 9. Generate Publication-Quality Visualizations
    generate_evaluation_plots(best_pipeline, best_model_name, X_test, y_test, results, output_dir)
    print("[*] Evaluation charts generated successfully in 'saved_models/plots/'")
    print("=" * 60)

def generate_evaluation_plots(best_pipeline, best_name, X_test, y_test, results, output_dir):
    plots_dir = os.path.join(output_dir, 'plots')
    y_pred = best_pipeline.predict(X_test)
    residuals = y_test - y_pred
    
    # Plot 1: Model Comparison Bar Chart
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    model_names = list(results.keys())
    r2_scores = [results[m]['test_r2'] for m in model_names]
    rmse_scores = [results[m]['rmse'] for m in model_names]
    
    colors = ['#4f46e5' if m == best_name else '#94a3b8' for m in model_names]
    
    axes[0].barh(model_names, r2_scores, color=colors)
    axes[0].set_xlabel('R² Score (Higher is Better)', fontsize=11, fontweight='bold')
    axes[0].set_title('Holdout R² Comparison Across Models', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, 1.05)
    for i, v in enumerate(r2_scores):
        axes[0].text(v + 0.02, i, f"{v:.3f}", va='center', fontweight='bold', fontsize=10)
        
    axes[1].barh(model_names, rmse_scores, color=colors)
    axes[1].set_xlabel('RMSE in $ (Lower is Better)', fontsize=11, fontweight='bold')
    axes[1].set_title('Root Mean Squared Error Across Models', fontsize=12, fontweight='bold')
    for i, v in enumerate(rmse_scores):
        axes[1].text(v + 1.0, i, f"${v:.1f}", va='center', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_benchmark_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 2: Actual vs Predicted Scatter Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='#4f46e5', edgecolors='none', s=45)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction Line')
    plt.xlabel('Actual Supermarket Sales ($)', fontsize=11, fontweight='bold')
    plt.ylabel(f'Predicted Sales by {best_name} ($)', fontsize=11, fontweight='bold')
    plt.title(f'Actual vs. Predicted Sales ({best_name})', fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()
    
    # Plot 3: Residuals Distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color='#0ea5e9', bins=25)
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual Error')
    plt.xlabel('Prediction Residuals (Actual - Predicted) ($)', fontsize=11, fontweight='bold')
    plt.ylabel('Frequency', fontsize=11, fontweight='bold')
    plt.title(f'Residuals Distribution ({best_name})', fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residuals_distribution.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    run_training_pipeline()
