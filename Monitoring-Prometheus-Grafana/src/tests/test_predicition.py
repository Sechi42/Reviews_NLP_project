import pytest
import numpy
from pathlib import Path
import os
import sys

# Adding the below path to avoid module not found error

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import generate_predictions
from prediction_model.processing.data_handling import load_pipeline

BERT_pipeline = load_pipeline(config.MODEL_NAME)

@pytest.fixture
def single_prediction():
    test_data = load_dataset(config.TEST_DATA)
    single_data = test_data.iloc[0:1]  # Selecciona la primera fila
    pred = BERT_pipeline.predict(single_data)
    return pred

def test_single_pred_not_none(single_prediction): #output is no none
    assert single_prediction is not None
    
def test_single_pred_str_type(single_prediction): #data type is not none
    print("single_predicition[0]: {} type: {}".format(single_prediction[0], type(single_prediction[0])))
    assert isinstance(single_prediction[0], numpy.int64)
    