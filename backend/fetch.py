"""Open-Meteo data access for Bakkhali, West Bengal."""

from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

LATITUDE, LONGITUDE, TIMEZONE = 21.63, 88.17, "Asia/Kolkata"
WEATHER_COLUMNS = [
    "Temperature (°C)", "Radiation (W/m²)", "Wind Speed (m/s)",
    "Humidity (%)", "Precipitation (mm/hr)", "Cloud Coverage (%)", "Pressure (kPa)",
]
API_TO_COLUMN = {
    "time": "datetime", "temperature_2m": "Temperature (°C)",
    "shortwave_radiation": "Radiation (W/m²)", "wind_speed_10m": "Wind Speed (m/s)",
    "relative_humidity_2m": "Humidity (%)", "rain": "Precipitation (mm/hr)",
    "cloud_cover": "Cloud Coverage (%)", "surface_pressure": "Pressure (kPa)",
}
HOURLY_FIELDS = ",".join(name for name in API_TO_COLUMN if name != "time")


def _request_json(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "hourly" not in data:
        raise RuntimeError(f"Weather provider returned no hourly data: {data.get('reason', data)}")
    return data


def _normalise_hourly(data: dict) -> pd.DataFrame:
    frame = pd.DataFrame(data["hourly"]).rename(columns=API_TO_COLUMN)
    required = ["datetime", *WEATHER_COLUMNS]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(f"Weather provider omitted required fields: {missing}")
    frame = frame[required].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    for column in WEATHER_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Open-Meteo surface pressure is hPa; the API exposes kPa.
    frame["Pressure (kPa)"] /= 10.0
    return frame.dropna(subset=required).sort_values("datetime").reset_index(drop=True)


def fetch_forecast_data(days: int = 8) -> pd.DataFrame:
    """Future hourly numerical-weather forecasts, in local Bakkhali time."""
    data = _request_json(
        "https://api.open-meteo.com/v1/forecast",
        {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": HOURLY_FIELDS,
         "forecast_days": days, "timezone": TIMEZONE},
    )
    return _normalise_hourly(data)


def fetch_historical_data(start_date: str = "2020-01-01", end_date: Optional[str] = None) -> pd.DataFrame:
    """Archive data for offline training, not for future forecasts."""
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()
    data = _request_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {"latitude": LATITUDE, "longitude": LONGITUDE, "start_date": start_date,
         "end_date": end_date, "hourly": HOURLY_FIELDS, "timezone": TIMEZONE},
    )
    return _normalise_hourly(data)


def get_recent_data(hours: int = 240) -> pd.DataFrame:
    """Compatibility helper for older callers."""
    start = (date.today() - timedelta(days=max(30, hours // 24 + 6))).isoformat()
    return fetch_historical_data(start_date=start).tail(hours + 120)
