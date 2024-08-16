import os
import pandas as pd
import joblib
from prediction_model.config import config 

def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    # Verificar si el archivo es .tsv o no
    if file_name.endswith('.tsv'):
        _data = pd.read_csv(filepath, sep='\t', dtype={'votes': 'Int64'})
    else:
        _data = pd.read_csv(filepath, dtype={'votes': 'Int64'})
    return _data

# Serialization
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print("Model has been saved under the name {}".format(config.MODEL_NAME))
    
# Deserialization   
def load_pipeline(pipeline_to_load):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print("Model has been loaded")
    return model_loaded