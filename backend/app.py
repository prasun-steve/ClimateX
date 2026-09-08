from datetime import datetime
import os
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .fetch import get_recent_data
from .predict import predict_next_24h, predict_next_72h, predict_next_168h

app = FastAPI(title="Bakkhali Weather Prediction API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Bakkhali Weather Prediction API",
        "endpoints": {
            "health": "/health",
            "predict_24h": "/api/predict/24h",
            "predict_72h": "/api/predict/72h",
            "predict_168h": "/api/predict/168h",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "forecast_source": "Open-Meteo"}


@app.get("/api/predict/24h")
async def predict_24h():
    """Predict next 24 hours — returns a flat list of hourly predictions."""
    try:
        print("Fetching 24-hour provider forecast...")
        predictions = predict_next_24h(get_recent_data(hours=360))

        return {
            "success": True,
            "horizon": "24h",
            "predictions": predictions,
            "error": None,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/72h")
async def predict_72h():
    """Predict the next 72 hours (3 days)."""
    try:
        print("Fetching 72-hour provider forecast...")
        result = predict_next_72h(get_recent_data(hours=360))

        return {
            "success": True,
            "horizon": "72h",
            "hourly": result["hourly"],
            "daily": result["daily"],
            "error": None,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/168h")
async def predict_168h():
    """Predict the next 168 hours (7 days)."""
    try:
        print("Fetching 168-hour provider forecast...")
        result = predict_next_168h(get_recent_data(hours=360))

        return {
            "success": True,
            "horizon": "168h",
            "hourly": result["hourly"],
            "daily": result["daily"],
            "error": None,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
