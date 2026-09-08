"""Recursive model-only forecasts. No future Open-Meteo forecast values are used."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import joblib, numpy as np, pandas as pd
from .fetch import WEATHER_COLUMNS
from .training import MODEL_PATH, make_features

LIMITS = {"Temperature (°C)":(-10,55), "Radiation (W/m²)":(0,1400), "Wind Speed (m/s)":(0,60),
          "Humidity (%)":(0,100), "Precipitation (mm/hr)":(0,150), "Cloud Coverage (%)":(0,100), "Pressure (kPa)":(85,110)}

def _model():
    if not MODEL_PATH.exists():
        raise RuntimeError("No trained model found. Run: python -m backend.training")
    return joblib.load(MODEL_PATH)

def _records(history: pd.DataFrame, hours: int) -> list[dict]:
    package = _model(); h = history[["datetime", *WEATHER_COLUMNS]].copy()
    h["datetime"] = pd.to_datetime(h["datetime"]); h = h.sort_values("datetime").drop_duplicates("datetime")
    h = h.set_index("datetime").resample("h").mean().interpolate("time", limit_direction="both").reset_index()
    if len(h) < 169: raise ValueError("At least 169 hourly history rows are required")
    output=[]
    for _ in range(hours):
        target = h["datetime"].iloc[-1] + pd.Timedelta(hours=1)
        candidate = {"datetime":target, **{c:float(h[c].iloc[-1]) for c in WEATHER_COLUMNS}}
        trial = pd.concat([h,pd.DataFrame([candidate])], ignore_index=True)
        row = make_features(trial).iloc[[-1]][package["features"]]
        values = package["model"].predict(row)[0]
        result = {c:round(float(np.clip(values[i], *LIMITS[c])),2) for i,c in enumerate(WEATHER_COLUMNS)}
        if target.hour < 5 or target.hour >= 19: result["Radiation (W/m²)"]=0.0
        if result["Precipitation (mm/hr)"] < .03: result["Precipitation (mm/hr)"]=0.0
        output.append({**result,"datetime":target.strftime("%Y-%m-%d %H:00"),"hour":target.hour,"date":target.strftime("%Y-%m-%d"),"source":"custom ML model"})
        h = pd.concat([h,pd.DataFrame([{**result,"datetime":target}])], ignore_index=True)
    return output

def _group(rows):
    groups=defaultdict(list)
    for row in rows: groups[row["date"]].append(row)
    return {"hourly":rows,"daily":[{"date":d,"hourly":v} for d,v in groups.items()]}
def predict_next_24h(history): return _records(history,24)
def predict_next_72h(history): return _group(_records(history,72))
def predict_next_168h(history): return _group(_records(history,168))
