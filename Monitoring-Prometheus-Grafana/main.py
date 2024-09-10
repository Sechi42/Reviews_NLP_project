from fastapi import FastAPI
import os
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from src.prediction_model.predict import generate_predictions
from prometheus_fastapi_instrumentator import Instrumentator

port = int(os.environ.get("PORT", 8005))
app = FastAPI(
    title="Review Prediction App using API -CI CD Jenkins",
    description="A simple CI CD Demo",
    version='1.0'
)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Corregido el parámetro
    allow_methods=['*'],
    allow_headers=['*']  # También corregido a 'allow_headers' en plural
)

class ReviewPrediciton(BaseModel):
    review: str
    

@app.get("/")
def index():
    return {"message": "Welcome to Review Sentiment Prediction APP using API - CI CD Jenkins"}

@app.post("/prediction_api")
def predict(loan_details: ReviewPrediciton):
    data = loan_details.model_dump()  # Aquí pasas los datos como diccionario
    prediction = generate_predictions([data])["prediction"][0]
    
    if prediction == "P":
        pred = "Positive"
    else:
        pred = "Negative"
        
    return {"Sentiment": pred}

@app.post("/predicition_ui")
def predict_gui(review: str):
    input_data = [review]
    cols = ['review']
    
    data_dict = dict(zip(cols, input_data))
    prediction = generate_predictions([data_dict])["prediction"][0]
    
    if prediction == "P":
        pred = "Positive"
    else:
        pred = "Negative"
        
    return {"Sentiment": pred}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
    
Instrumentator().instrument(app).expose(app)
