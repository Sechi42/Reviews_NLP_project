import pandas as pd
import numpy as np
import joblib
from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline, load_dataset
import torch

BERT_class_pipeline = load_pipeline(config.MODEL_NAME)

'''def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    pred = BERT_class_pipeline(data[config.FEATURES])
    result = {'Predictions':pred.tolist()}
    return result'''
    
def generate_predictions(data):
    # data es la reseña que proviene del API
    # Aquí asumimos que data es una lista o diccionario de reseñas
    
    # Si necesitas convertir los datos a un formato específico, puedes hacerlo aquí
    df = pd.DataFrame(data)  # Si es una lista de diccionarios, esto funcionará
    
    # Generar predicción con el pipeline de BERT (simulado aquí con una condición)
    pred = BERT_class_pipeline.predict(df)
    
    # Convertir las predicciones en 'P' o 'N'
    output = np.where(pred == 1, 'P', 'N')
    print(output)
    return {"prediction": output}



if __name__=='__main__':
    # Ejemplo de datos para probar la función
    example_data = [{"review": "I don't like the movie"}]
    predictions = generate_predictions(data=example_data)
    print(predictions)
    