# -*- coding: utf-8 -*-
"""
Step 3: Baseline Models for Demand Forecasting
Random Forest, Gradient Boosting, and comparison with Naive forecast
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STEP 3: BASELINE MODELS")
print("=" * 70)

# Load feature-engineered data
print("\n[INFO] Loading feature data...")
df = pd.read_csv('data/processed/walmart_features.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"[OK] Loaded {len(df)} records with {len(df.columns)} columns")

# Calculate Weighted Mean Absolute Percentage Error
def calculate_wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

# Split data by time (last 13 weeks for testing)
dates = sorted(df['Date'].unique())
train_dates = dates[:-13]  # All except last 13 weeks
test_dates = dates[-13:]   # Last 13 weeks for testing

train_data = df[df['Date'].isin(train_dates)]
test_data = df[df['Date'].isin(test_dates)]

print(f"\n[INFO] Data Split:")
print(f"   Training: {train_data['Date'].min().date()} to {train_data['Date'].max().date()}")
print(f"     Records: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
print(f"   Testing: {test_data['Date'].min().date()} to {test_data['Date'].max().date()}")
print(f"     Records: {len(test_data)} ({len(test_data)/len(df)*100:.1f}%)")

# Prepare features (exclude non-numeric and target)
exclude_cols = ['Store', 'Date', 'Weekly_Sales']
feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

X_train = train_data[feature_cols].fillna(0)
y_train = train_data['Weekly_Sales']
X_test = test_data[feature_cols].fillna(0)
y_test = test_data['Weekly_Sales']

print(f"\n[INFO] Using {len(feature_cols)} features for modeling")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Random Forest Model
print("\n" + "-" * 70)
print("TRAINING RANDOM FOREST MODEL")
print("-" * 70)

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

# 2. Gradient Boosting Model
print("\n" + "-" * 70)
print("TRAINING GRADIENT BOOSTING MODEL")
print("-" * 70)

gb = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb.fit(X_train_scaled, y_train)
gb_pred = gb.predict(X_test_scaled)

# 3. Naive Forecast (using previous week's sales)
print("\n" + "-" * 70)
print("NAIVE FORECAST (Previous Week)")
print("-" * 70)

if 'sales_lag_1w' in test_data.columns:
    naive_pred = test_data['sales_lag_1w'].fillna(test_data['Weekly_Sales'].mean())
else:
    naive_pred = np.full_like(y_test, y_train.mean())

# Calculate metrics for all models
models = {
    'Random Forest': rf_pred,
    'Gradient Boosting': gb_pred,
    'Naive Forecast': naive_pred
}

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)
print(f"{'Model':<20} {'MAE ($)':<15} {'RMSE ($)':<15} {'WMAPE (%)':<12} {'R2':<8}")
print("-" * 70)

results = {}
for name, pred in models.items():
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    wmape_val = calculate_wmape(y_test.values, pred)
    r2 = r2_score(y_test, pred)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'WMAPE': wmape_val, 'R2': r2}
    print(f"{name:<20} ${mae:>12,.2f} ${rmse:>12,.2f} {wmape_val:>11.2f}% {r2:>8.4f}")

# Find best model
best_model = min(results, key=lambda x: results[x]['WMAPE'])
best_wmape = results[best_model]['WMAPE']
print(f"\n[RESULT] BEST MODEL: {best_model} with WMAPE = {best_wmape:.2f}%")

# Feature Importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 70)
print("TOP 15 MOST IMPORTANT FEATURES (Random Forest)")
print("=" * 70)
for i, row in feature_importance.head(15).iterrows():
    # Create a simple text bar using equals signs
    bar_length = int(row['importance'] * 50)
    bar = '=' * bar_length
    print(f"   {row['feature']:35s} {bar:50s} {row['importance']:.4f}")

# Save predictions
results_df = pd.DataFrame({
    'Store': test_data['Store'].values,
    'Date': test_data['Date'].values,
    'Actual': y_test.values,
    'RandomForest_Predicted': rf_pred,
    'GradientBoosting_Predicted': gb_pred,
    'RF_Error': np.abs(y_test.values - rf_pred),
    'RF_Percent_Error': np.abs((y_test.values - rf_pred) / y_test.values) * 100
})
results_df.to_csv('reports/model_predictions.csv', index=False)
print(f"\n[SAVED] Predictions to: reports/model_predictions.csv")

# Save feature importance
feature_importance.to_csv('reports/feature_importance.csv', index=False)
print(f"[SAVED] Feature importance to: reports/feature_importance.csv")

print("\n" + "=" * 70)
print("STEP 3 COMPLETED SUCCESSFULLY!")
print("=" * 70)

# Summary
print(f"\nPROJECT SUMMARY")
print(f"   Training weeks: {len(train_dates)}")
print(f"   Testing weeks: {len(test_dates)}")
print(f"   Stores: {df['Store'].nunique()}")
print(f"   Features engineered: {len(feature_cols)}")
print(f"   Best WMAPE: {best_wmape:.2f}%")
print(f"\n   Target for TFT: 15-20% reduction")
print(f"   Target WMAPE range: {best_wmape * 0.8:.2f}% to {best_wmape * 0.85:.2f}%")