"""
Step 2: Feature Engineering for Walmart Sales Forecasting
Creates advanced features for time series prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)

# Load the processed data from Step 1
print("\n[INFO] Loading data from Step 1...")
df = pd.read_csv('data/processed/walmart_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"[SUCCESS] Loaded {len(df)} records")

# Sort by store and date
df = df.sort_values(['Store', 'Date'])

# 1. Date Features
print("\n[1/6] Adding date features...")
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['quarter'] = df['Date'].dt.quarter
df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
df['day_of_week'] = df['Date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Cyclical encoding
df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# 2. Lag Features (past sales)
print("[2/6] Adding lag features...")
lags = [1, 2, 3, 4, 8, 12, 13, 26]
for lag in lags:
    df[f'sales_lag_{lag}w'] = df.groupby('Store')['Weekly_Sales'].shift(lag)
    df[f'temp_lag_{lag}w'] = df.groupby('Store')['Temperature'].shift(lag)
    if 'CPI' in df.columns:
        df[f'cpi_lag_{lag}w'] = df.groupby('Store')['CPI'].shift(lag)
    if 'Unemployment' in df.columns:
        df[f'unemp_lag_{lag}w'] = df.groupby('Store')['Unemployment'].shift(lag)

# 3. Rolling Statistics
print("[3/6] Adding rolling statistics...")
windows = [4, 8, 13, 26]
for window in windows:
    # Rolling mean
    df[f'sales_rolling_mean_{window}w'] = (
        df.groupby('Store')['Weekly_Sales']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    # Rolling standard deviation
    df[f'sales_rolling_std_{window}w'] = (
        df.groupby('Store')['Weekly_Sales']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
    )
    # Rolling min and max
    df[f'sales_rolling_min_{window}w'] = (
        df.groupby('Store')['Weekly_Sales']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).min())
    )
    df[f'sales_rolling_max_{window}w'] = (
        df.groupby('Store')['Weekly_Sales']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
    )

# 4. Exponential Moving Averages
print("[4/6] Adding exponential moving averages...")
spans = [4, 8, 13]
for span in spans:
    df[f'sales_ewm_{span}w'] = (
        df.groupby('Store')['Weekly_Sales']
        .transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    )

# 5. Rate of Change (Momentum)
print("[5/6] Adding rate of change features...")
df['sales_wow_pct'] = df.groupby('Store')['Weekly_Sales'].pct_change() * 100
df['sales_mom_pct'] = (df.groupby('Store')['Weekly_Sales'].pct_change(4)) * 100
df['sales_qoq_pct'] = (df.groupby('Store')['Weekly_Sales'].pct_change(13)) * 100
df['temp_wow_pct'] = df.groupby('Store')['Temperature'].pct_change() * 100

# 6. Enhanced Holiday Features
print("[6/6] Adding enhanced holiday features...")
df['is_holiday'] = df['Holiday_Flag']
df['prev_week_holiday'] = df.groupby('Store')['is_holiday'].shift(1)
df['next_week_holiday'] = df.groupby('Store')['is_holiday'].shift(-1)
df['is_holiday_season'] = ((df['week_of_year'] >= 47) | (df['week_of_year'] <= 2)).astype(int)
df['is_black_friday_week'] = ((df['week_of_year'] >= 47) & (df['week_of_year'] <= 48)).astype(int)
df['is_christmas_week'] = ((df['week_of_year'] >= 50) & (df['week_of_year'] <= 52)).astype(int)
df['is_summer'] = (df['month'].isin([6, 7, 8])).astype(int)

# Economic indicators
if 'CPI' in df.columns:
    df['cpi_inflation'] = df.groupby('Store')['CPI'].pct_change(4) * 100
    df['cpi_rolling_13w'] = df.groupby('Store')['CPI'].transform(lambda x: x.rolling(13, min_periods=1).mean())

if 'Unemployment' in df.columns:
    df['unemp_change'] = df.groupby('Store')['Unemployment'].diff(4)

# Drop NaN values
initial_rows = len(df)
df = df.dropna()
print(f"\n[INFO] Rows after dropping NaN: {len(df)} (dropped {initial_rows - len(df)} rows)")

# Feature summary
feature_cols = [col for col in df.columns if col not in ['Store', 'Date', 'Weekly_Sales']]
print(f"\n[SUCCESS] Total features created: {len(feature_cols)}")

# Save feature-engineered data
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv('data/processed/walmart_features.csv', index=False)
print(f"\n[SUCCESS] Features saved to: data/processed/walmart_features.csv")

print("\n" + "=" * 70)
print("STEP 2 COMPLETED SUCCESSFULLY!")
print("=" * 70)

# Show sample features
print("\nSample of created features:")
sample_features = feature_cols[:10]
print(df[sample_features].head(3))
