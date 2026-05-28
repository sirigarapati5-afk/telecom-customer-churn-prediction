
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("small_churn_model.pkl")

@app.get("/")

def home():
    return {"message": "Churn Prediction API"}

@app.get("/predict")

def predict(tenure: int, monthly: float):

    data = np.array([[tenure, monthly]])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
