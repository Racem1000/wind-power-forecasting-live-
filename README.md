# Wind Power Forecasting System
> Probabilistic 24-hour ahead wind power forecast with live dashboard

[![Live Demo](https://img.shields.io/badge/Live-Demo-39d353?style=flat-square)](https://Racem1000.github.io/wind-power-forecasting)
[![Python](https://img.shields.io/badge/Python-3.10-58a6ff?style=flat-square)]()
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-f0883e?style=flat-square)]()

## 🎯 Headline Result

> **XGBoost ensemble outperformed the persistence baseline by 12.1% (RMSE: 842 MW vs 958 MW),  
> equivalent to ~€34.3M estimated annual imbalance cost savings  
> on the German onshore wind portfolio.**

**[→ View Live Dashboard](https://Racem1000.github.io/wind-power-forecasting)**

---

## 📊 Results Summary

| Model | RMSE (MW) | MAE (MW) | vs Baseline |
|-------|-----------|----------|-------------|
| Persistence (baseline) | 958 | 709 | — |
| XGBoost | 843 | 631 | +11.9% |
| LightGBM | 879 | 662 | +8.2% |
| **Ensemble (best)** | **842** | **—** | **+12.1%** |
| LSTM | 964 | 724 | -0.7% |

**Probabilistic forecast:** 78.3% calibrated coverage on P10–P90 interval  
**24h horizon:** RMSE grows from 1,687 MW (+1h) to 8,642 MW (+24h)  
**Auto-retraining:** RMSE improved 51% over 3 months with live data

---

## 🗂 Project Structure

| Notebook | Description | Key Output |
|----------|-------------|------------|
| `01_data_collection.ipynb` | Download ERA5 weather + OPSD generation | 8,760 hrs dataset |
| `02_eda.ipynb` | Seasonal patterns, power curve, correlations | 4 insight charts |
| `03_features.ipynb` | Lag features, rolling averages, sin/cos encoding | 20 features |
| `04_models.ipynb` | Persistence / XGBoost / LSTM benchmark | RMSE comparison |
| `05_evaluation.ipynb` | Economic framing in €/MWh | €34.3M headline |
| `06_probabilistic.ipynb` | Quantile regression + conformal calibration | 78.3% coverage |
| `07_api.ipynb` | FastAPI REST endpoint P10/P50/P90 | Live API |
| `08_retraining.ipynb` | Drift detection + auto-retrain pipeline | 51% improvement |
| `09_ensemble.ipynb` | LightGBM + SHAP + ensemble blending | 842 MW RMSE |
| `10_24h_forecast.ipynb` | Direct multi-step 24h horizon models | Full day forecast |
| `11_dashboard.ipynb` | Interactive frontend dashboard | Live demo |

---

## 🔧 Tech Stack

- **Data:** Open Power System Data · Open-Meteo ERA5 · ENTSO-E
- **Models:** XGBoost · LightGBM · LSTM (TensorFlow) · Ensemble
- **Evaluation:** SHAP · Conformal prediction · Pinball loss · €/MWh
- **Deployment:** FastAPI · REST API · GitHub Pages dashboard
- **Pipeline:** Auto-retraining · Drift detection · MLflow logging

---

## 📈 Key Findings

1. `wind_mw_lag1` is the dominant feature (SHAP: 6,231) — wind is persistent
2. Winter produces **3x more power** than summer — month encoding is critical
3. Wind speed at 100m correlates **0.77** with power — strongest weather feature
4. XGBoost beats LSTM with limited data — gradient boosting wins under 10k samples
5. Conformal calibration fixed coverage from 67.4% → **78.3%** (target: 80%)
6. Auto-retraining reduced RMSE by **51%** over 3 months of fresh data

---

## 🚀 Quick Start
```bash
git clone https://github.com/Racem1000/wind-power-forecasting.git
cd wind-power-forecasting
pip install -r requirements.txt
```

Open notebooks in order: 01 → 02 → ... → 11

---

## 🌍 Context

This project is part of a larger goal: building an end-to-end wind power forecasting
system that feeds into VPP dispatch optimization 

---

*Built by [@Racem1000](https://github.com/Racem1000) 
