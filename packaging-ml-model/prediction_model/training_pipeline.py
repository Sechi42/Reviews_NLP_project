import pandas as pd
import numpy as pd
from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from prediction_model.processing.data_handling import load_dataset, save_pipeline
import prediction_model.pipeline as pipe
import sys

def perform_training():
    
    train_data = load_dataset(config.TRAIN_DATA)
    target_train = train_data[config.TARGET]
    pipe.BERT_pipeline.fit(train_data, target_train)
    save_pipeline(pipe.BERT_pipeline)
    
if __name__=='__main__':
    perform_training()