
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load trained model
saved_data = joblib.load("models/day19_fastapi_model.joblib")

model = saved_data["model"]
threshold = saved_data["threshold"]

# Create FastAPI application
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="ML model serving using FastAPI",
    version="1.0"
)


# Input data format
class PredictionInput(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="30 numerical features"
    )


# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Model API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):

    try:
        input_data = np.array(data.features).reshape(1, -1)

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        prediction = int(probability >= threshold)

        return {
            "prediction": prediction,
            "result": (
                "Benign" if prediction == 1 else "Malignant"
            ),
            "probability": round(probability, 4),
            "threshold": threshold
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
