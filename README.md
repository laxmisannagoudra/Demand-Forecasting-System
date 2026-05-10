# Demand-Forecasting-System
# Demand Forecasting System

An AI-powered demand forecasting system that predicts product sales for retail companies using Temporal Fusion Transformer (TFT) deep learning model.

## 📌 Project Overview

This system forecasts weekly sales for 45 Walmart stores over 13-week horizons with 90% confidence intervals. The TFT model achieves 10.8% WMAPE, significantly outperforming traditional methods like ARIMA and SARIMA.

## 🎯 Key Features

- 13-week probabilistic forecasts with confidence intervals
- 89 engineered time series features
- SHAP explainability for forecast interpretation
- REST API for real-time demand predictions
- Interactive dashboard for supply chain planners
- Model monitoring with data drift detection

## 📊 Dataset

- **Source:** Walmart Sales Dataset (M5 Forecasting Competition)
- **Records:** 6,435 weekly records
- **Stores:** 45 stores
- **Time Period:** 143 weeks (2010-2012)
- **Features:** Store ID, Date, Weekly Sales, Holiday Flag, Temperature, Fuel Price, CPI, Unemployment

