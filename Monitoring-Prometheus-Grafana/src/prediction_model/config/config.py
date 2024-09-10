import pathlib 
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

DATA_FILE = 'imdb_reviews.tsv'
TRAIN_DATA = 'imdb_train.tsv'
TEST_DATA = 'imdb_test.tsv'
TENSOR_FILE = 'features_9.npz'

MODEL_NAME = 'BERT_model.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')

TARGET = 'pos'

# Final features use in the model
FEATURES_TO_REMOVE = ['tconst', 'title_type', 'primary_title', 'original_title', 'start_year',
       'end_year', 'runtime_minutes', 'is_adult', 'genres', 'average_rating',
       'votes', 'rating', 'sp', 'ds_part', 'idx', 'pos']

REVIEW_FEATURES = 'review'

FEATURE_TO_MODIFY = 'review'
FEATURES_TO_ADD = 'review_norm'

# En este caso es lo mismo que las review features
FEATURES_TO_TENSOR = 'review_norm'
FEATURES = 'review_norm'



