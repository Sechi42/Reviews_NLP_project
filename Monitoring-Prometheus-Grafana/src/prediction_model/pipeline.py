from sklearn.pipeline import Pipeline
from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.linear_model import LogisticRegression
import numpy as np

BERT_pipeline = Pipeline(
    [
        ('ReviewNormalization', pp.ReviewNormImputer(variable_to_add=config.FEATURES_TO_ADD, variable_to_modify=config.FEATURE_TO_MODIFY)),
        ('BERTTensorTransform', pp.BERTTextEmbeddingsTransformer(force_device='cuda', column_name=config.FEATURES)),
        ('LogisticClassifier', LogisticRegression(random_state=12345)),
    ]
)


