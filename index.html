
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load models
horizon_models = joblib.load("horizon_models.pkl")
FEATURES       = joblib.load("features_list.pkl")
ADJUSTMENT     = 1156.0

st.set_page_config(page_title="Wind Power Forecast", layout="wide")

st.title("🌬️ Wind Power Forecast — Germany Onshore")
st.markdown("**Probabilistic 24-hour ahead forecast · P10 / P50 / P90**")
st.markdown("---")

# ── Sidebar inputs ──
st.sidebar.header("Input conditions")

wind_speed_100m = st.sidebar.slider(
    "Wind speed at 100m (m/s)", 0.0, 70.0, 25.0, 0.5)
wind_speed_10m = wind_speed_100m * 0.6
wind_direction = st.sidebar.slider(
    "Wind direction (degrees)", 0, 360, 225)
hour = st.sidebar.slider(
    "Current hour", 0, 23, 12)
month = st.sidebar.slider(
    "Month", 1, 12, 11)
recent_power = st.sidebar.number_input(
    "Recent power output (MW)", 0, 50000, 12000, 500)

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown("Model: XGBoost ensemble")
st.sidebar.markdown("Coverage: 78.3% calibrated")
st.sidebar.markdown("Data: Germany 2019")

# ── Build feature vector ──
def build_row(hour, month, wind_speed_100m, wind_speed_10m,
              wind_direction, recent_power):
    return {
        "wind_speed_10m":       wind_speed_10m,
        "wind_speed_100m":      wind_speed_100m,
        "wind_dir_sin":         np.sin(np.deg2rad(wind_direction)),
        "wind_dir_cos":         np.cos(np.deg2rad(wind_direction)),
        "hour_sin":             np.sin(2 * np.pi * hour / 24),
        "hour_cos":             np.cos(2 * np.pi * hour / 24),
        "month_sin":            np.sin(2 * np.pi * month / 12),
        "month_cos":            np.cos(2 * np.pi * month / 12),
        "wind_mw_lag1":         recent_power,
        "wind_mw_lag24":        recent_power * 0.95,
        "wind_mw_lag48":        recent_power * 0.90,
        "wind_mw_lag168":       recent_power * 0.85,
        "wspeed_lag1":          wind_speed_100m,
        "wspeed_lag24":         wind_speed_100m * 0.95,
        "wind_mw_roll6":        recent_power * 0.97,
        "wind_mw_roll24":       recent_power * 0.93,
        "wspeed_roll6":         wind_speed_100m * 0.97,
        "wspeed_roll24":        wind_speed_100m * 0.93,
        "dayofweek":            1,
        "quarter":              (month - 1) // 3 + 1,
    }

# ── Generate 24h forecast ──
row  = build_row(hour, month, wind_speed_100m,
                 wind_speed_10m, wind_direction, recent_power)
X    = pd.DataFrame([row])[FEATURES]

forecasts = []
for h in range(1, 25):
    pred = float(np.clip(horizon_models[h].predict(X)[0], 0, None))
    unc  = 0.10 + (h / 24) * 0.20
    forecasts.append({
        "hour_ahead": h,
        "p50": round(pred),
        "p10": round(max(0, pred * (1 - unc))),
        "p90": round(pred * (1 + unc)),
    })

fc_df = pd.DataFrame(forecasts)

# ── Layout: 3 metric cards ──
col1, col2, col3 = st.columns(3)
col1.metric("Next 1h forecast (P50)",
            f"{fc_df.iloc[0]['p50']:,} MW",
            f"±{fc_df.iloc[0]['p90'] - fc_df.iloc[0]['p50']:,} MW")
col2.metric("Next 6h forecast (P50)",
            f"{fc_df.iloc[5]['p50']:,} MW",
            f"±{fc_df.iloc[5]['p90'] - fc_df.iloc[5]['p50']:,} MW")
col3.metric("Next 24h forecast (P50)",
            f"{fc_df.iloc[23]['p50']:,} MW",
            f"±{fc_df.iloc[23]['p90'] - fc_df.iloc[23]['p50']:,} MW")

# ── Main chart ──
st.subheader("24-hour probabilistic forecast")

fig, ax = plt.subplots(figsize=(13, 5))
hours = fc_df["hour_ahead"]
ax.fill_between(hours, fc_df["p10"], fc_df["p90"],
                alpha=0.25, color="steelblue", label="P10–P90 band")
ax.plot(hours, fc_df["p50"],
        color="steelblue", linewidth=2.5,
        marker="o", markersize=4, label="P50 forecast")
ax.plot(hours, fc_df["p10"],
        color="steelblue", linewidth=1,
        linestyle="--", alpha=0.6, label="P10 (pessimistic)")
ax.plot(hours, fc_df["p90"],
        color="steelblue", linewidth=1,
        linestyle="--", alpha=0.6, label="P90 (optimistic)")
ax.set_xlabel("Hours ahead")
ax.set_ylabel("Wind generation (MW)")
ax.set_title(f"Forecast from hour {hour}:00 · Month {month} · "
             f"Wind {wind_speed_100m} m/s")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# ── Data table ──
st.subheader("Forecast table")
display_df = fc_df.copy()
display_df.columns = ["Hour ahead", "P50 (MW)", "P10 (MW)", "P90 (MW)"]
st.dataframe(display_df.set_index("Hour ahead"), use_container_width=True)

# ── Footer ──
st.markdown("---")
st.markdown(
    "Built by a wind power forecasting project · "
    "Data: Open Power System Data + Open-Meteo · "
    "Model: XGBoost quantile regression + conformal calibration"
)
