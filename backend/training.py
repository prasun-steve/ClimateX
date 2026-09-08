"""Train the project's own one-hour-ahead weather model from past data only."""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from .fetch import WEATHER_COLUMNS, fetch_historical_data

MODEL_DIR = Path(__file__).resolve().parent / "models"
# New model name deliberately avoids loading the old, non-portable pickle that
# was created with a different scikit-learn internals layout.
MODEL_PATH = MODEL_DIR / "custom_weather_model_v2.joblib"
LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 120, 168)

def make_features(data: pd.DataFrame) -> pd.DataFrame:
    x = data.copy().sort_values("datetime").reset_index(drop=True)
    dt = pd.to_datetime(x["datetime"])
    f = pd.DataFrame(index=x.index)
    f["hour_sin"] = np.sin(2*np.pi*dt.dt.hour/24); f["hour_cos"] = np.cos(2*np.pi*dt.dt.hour/24)
    f["doy_sin"] = np.sin(2*np.pi*dt.dt.dayofyear/365.25); f["doy_cos"] = np.cos(2*np.pi*dt.dt.dayofyear/365.25)
    for col in WEATHER_COLUMNS:
        for lag in LAGS: f[f"{col}__lag_{lag}"] = x[col].shift(lag)
        previous = x[col].shift(1)
        f[f"{col}__mean_6"] = previous.rolling(6, min_periods=2).mean()
        f[f"{col}__mean_24"] = previous.rolling(24, min_periods=2).mean()
    return f

def train(start_date: str = "2022-01-01") -> dict:
    """Fit t -> t+1 model, evaluate chronologically, then activate it."""
    weather = fetch_historical_data(start_date=start_date)
    features = make_features(weather)
    # Targets are explicitly shifted one hour into the future: no leakage.
    targets = weather[WEATHER_COLUMNS].shift(-1)
    joined = pd.concat([features, targets.add_prefix("target__")], axis=1).dropna()
    x = joined[features.columns]; y = joined[targets.add_prefix("target__").columns]
    split = int(len(x)*0.8)
    model = MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=350, learning_rate=.05, max_leaf_nodes=31, l2_regularization=1., random_state=42))
    model.fit(x.iloc[:split], y.iloc[:split])
    predicted = model.predict(x.iloc[split:])
    metrics = {WEATHER_COLUMNS[i]: round(float(mean_absolute_error(y.iloc[split:, i], predicted[:, i])), 3)
               for i in range(len(WEATHER_COLUMNS))}
    MODEL_DIR.mkdir(exist_ok=True)
    package = {"model": model, "features": list(x.columns), "targets": WEATHER_COLUMNS,
               "lags": LAGS, "trained_through": str(weather["datetime"].max()), "mae": metrics}
    joblib.dump(package, MODEL_PATH)
    report = {"model": MODEL_PATH.name, "trained_through": package["trained_through"],
              "train_rows": split, "test_rows": len(x)-split, "mae": metrics, "activated": True}
    (MODEL_DIR / "custom_weather_model_v2_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__": print(json.dumps(train(), indent=2))
