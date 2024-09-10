import pandas as pd
import numpy as np
import joblib
from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline, load_dataset
import torch

BERT_class_pipeline = load_pipeline(config.MODEL_NAME)

def generate_predictions_1(data_input):
    data = pd.DataFrame(data_input)
    pred = BERT_class_pipeline(data)
    result = {'Predictions':pred.tolist()}
    return result
    
def generate_predictions():
    test_data = load_dataset(config.TEST_DATA)
    pred = BERT_class_pipeline.predict(test_data)
    output = np.where(pred == 1, 'P', 'N')
    print(output)
    return output


if __name__=='__main__':
    predictions = generate_predictions()
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv("predictions.csv", index=False)