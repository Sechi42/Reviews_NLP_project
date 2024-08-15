from sklearn.base import BaseEstimator, TransformerMixin
import re

class ReviewNormImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variable_to_modify=None, variable_to_add = None):
        self.variable_to_modify = variable_to_modify
        self.variable_to_add = variable_to_add
        self.review_norm = {}

    def fit(self, X, y=None):
        pattern = r"[^a-z\s']"  # Solo letras, espacios y apóstrofes
        for col in self.variable_to_modify:
            self.review_norm[col] = X[col].str.lower().apply(lambda x: re.sub(pattern, " ", x))
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.variable_to_modify:
            X[self.variable_to_add] = self.review_norm[col]
        return X

import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import math
import numpy as np

class BERTTextEmbeddingsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_length=512, batch_size=100, force_device=None, disable_progress_bar=True):

        self.max_length = max_length
        self.batch_size = batch_size
        self.force_device = force_device
        self.disable_progress_bar = disable_progress_bar
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def fit(self, X, y=None):
        # No hay que ajustar nada, por lo que simplemente se devuelve self
        return self

    def transform(self, X):
        ids_list = []
        attention_mask_list = []

        # Convertir textos a IDs de tokens y máscaras de atención
        for text in X:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            ids_list.append(encoding['input_ids'].squeeze().tolist())
            attention_mask_list.append(encoding['attention_mask'].squeeze().tolist())

        if self.force_device is not None:
            device = torch.device(self.force_device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda' and not torch.cuda.is_available():
            print("Advertencia: CUDA no está disponible en este sistema. Usando CPU en su lugar.")
            device = torch.device('cpu')

        self.model.to(device)
        if not self.disable_progress_bar:
            print(f'Uso del dispositivo {device}.')

        # Obtener embeddings en lotes
        embeddings = []

        for i in tqdm(range(math.ceil(len(ids_list) / self.batch_size)), disable=self.disable_progress_bar):
            ids_batch = torch.LongTensor(ids_list[self.batch_size * i:self.batch_size * (i + 1)]).to(device)
            attention_mask_batch = torch.LongTensor(attention_mask_list[self.batch_size * i:self.batch_size * (i + 1)]).to(device)

            with torch.no_grad():
                self.model.eval()
                batch_embeddings = self.model(input_ids=ids_batch, attention_mask=attention_mask_batch)
            embeddings.append(batch_embeddings.last_hidden_state[:, 0, :].detach().cpu().numpy())

        return np.concatenate(embeddings)

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables_to_drop, axis=1)
        return X