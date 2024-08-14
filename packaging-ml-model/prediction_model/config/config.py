import pathlib 
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

DATA_FILE = 'imdb_reviews.tsv'
TENSOR_FILE = 'features_9.npz'

SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')

TARGET = 'pos'

# Final features use in the model
FEATURES = ['tconst', 'title_type', 'primary_title', 'original_title', 'start_year',
       'end_year', 'runtime_minutes', 'is_adult', 'genres', 'average_rating',
       'votes', 'review', 'rating', 'sp', 'ds_part', 'idx',
       'review_norm']

REVIEW_FEATURES = ['review']

FEATURES_TO_ADD = ['review_norm']

# En este caso es lo mismo que las review features
FEATURES_TO_TENSOR = ['review_norm']



