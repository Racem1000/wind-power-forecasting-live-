
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load models
models     = joblib.load("wind_forecast_models.pkl")
FEATURES   = joblib.load("wind_forecast_features.pkl")
ADJUSTMENT = joblib.load("wind_forecast_adjustment.pkl")

app = FastAPI(title="Wind Power Forecast API",
              description="Probabilistic wind power forecast — P10/P50/P90",
              version="1.0")

# Input schema — what the user sends
class ForecastRequest(BaseModel):
    wind_speed_10m:       float   # m/s
    wind_speed_100m:      float   # m/s
    wind_direction_100m:  float   # degrees 0-360
    hour:                 int     # 0-23
    month:                int     # 1-12
    dayofweek:            int     # 0=Monday, 6=Sunday
    quarter:              int     # 1-4
    wind_mw_lag1:         float   # MW 1 hour ago
    wind_mw_lag24:        float   # MW 24 hours ago
    wind_mw_lag48:        float   # MW 48 hours ago
    wind_mw_lag168:       float   # MW 168 hours ago
    wspeed_lag1:          float   # wind speed 1h ago
    wspeed_lag24:         float   # wind speed 24h ago
    wind_mw_roll6:        float   # 6h rolling avg MW
    wind_mw_roll24:       float   # 24h rolling avg MW
    wspeed_roll6:         float   # 6h rolling avg wind speed
    wspeed_roll24:        float   # 24h rolling avg wind speed

def build_features(req: ForecastRequest) -> pd.DataFrame:
    """Convert request into model feature vector"""
    row = {
        "wind_speed_10m":       req.wind_speed_10m,
        "wind_speed_100m":      req.wind_speed_100m,
        "wind_dir_sin":         np.sin(np.deg2rad(req.wind_direction_100m)),
        "wind_dir_cos":         np.cos(np.deg2rad(req.wind_direction_100m)),
        "hour_sin":             np.sin(2 * np.pi * req.hour / 24),
        "hour_cos":             np.cos(2 * np.pi * req.hour / 24),
        "month_sin":            np.sin(2 * np.pi * req.month / 12),
        "month_cos":            np.cos(2 * np.pi * req.month / 12),
        "wind_mw_lag1":         req.wind_mw_lag1,
        "wind_mw_lag24":        req.wind_mw_lag24,
        "wind_mw_lag48":        req.wind_mw_lag48,
        "wind_mw_lag168":       req.wind_mw_lag168,
        "wspeed_lag1":          req.wspeed_lag1,
        "wspeed_lag24":         req.wspeed_lag24,
        "wind_mw_roll6":        req.wind_mw_roll6,
        "wind_mw_roll24":       req.wind_mw_roll24,
        "wspeed_roll6":         req.wspeed_roll6,
        "wspeed_roll24":        req.wspeed_roll24,
        "dayofweek":            req.dayofweek,
        "quarter":              req.quarter,
    }
    return pd.DataFrame([row])[FEATURES]

@app.get("/")
def root():
    return {"message": "Wind Power Forecast API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    X = build_features(req)

    p10 = float(np.clip(models["p10"].predict(X)[0] - ADJUSTMENT, 0, None))
    p50 = float(np.clip(models["p50"].predict(X)[0], 0, None))
    p90 = float(np.clip(models["p90"].predict(X)[0] + ADJUSTMENT, 0, None))

    return {
        "p10_mw":    round(p10),
        "p50_mw":    round(p50),
        "p90_mw":    round(p90),
        "bandwidth": round(p90 - p10),
        "unit":      "MW",
        "note":      "P10-P90 = 78% calibrated interval"
    }
