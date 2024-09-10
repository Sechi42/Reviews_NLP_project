import mlflow
import pandas as pd
import torch
import transformers
import re
import numpy as np
from tqdm import tqdm
import math
import pickle
import mlflow.pyfunc

logged_model = 'runs:/c19ce66d76604e6c8bf4dee8aacf064b/trained_model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

artifact_uri = 'runs:/c19ce66d76604e6c8bf4dee8aacf064b/scaler.pkl'
local_scaler_path = mlflow.artifacts.download_artifacts(artifact_uri)

# Cargar el scaler desde el archivo descargado
with open(local_scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Función para normalizar texto
def normalize_text(text):
    pattern = r"[^a-z\s]"
    return re.sub(pattern, " ", text.lower())

# Función para convertir texto a embeddings usando BERT
def BERT_text_to_embeddings(texts, max_length=512, batch_size=100, disable_progress_bar=False):
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    model = transformers.BertModel.from_pretrained('bert-base-uncased')
    
    ids_list = []
    attention_mask_list = []

    # Normalización y tokenización de los textos
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        ids_list.append(encoding['input_ids'].squeeze().tolist())
        attention_mask_list.append(encoding['attention_mask'].squeeze().tolist())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    embeddings = []

    # Procesamiento en lotes
    for i in tqdm(range(math.ceil(len(ids_list) / batch_size)), disable=disable_progress_bar):
        ids_batch = torch.LongTensor(ids_list[batch_size * i:batch_size * (i + 1)]).to(device)
        attention_mask_batch = torch.LongTensor(attention_mask_list[batch_size * i:batch_size * (i + 1)]).to(device)

        with torch.no_grad():
            model.eval()
            batch_embeddings = model(input_ids=ids_batch, attention_mask=attention_mask_batch)
        embeddings.append(batch_embeddings[0][:, 0, :].detach().cpu().numpy())

    return np.concatenate(embeddings)

# Ejemplo de reseñas normalizadas
my_reviews = pd.DataFrame([
    'I did not simply like it, not my kind of movie.',
    'Well, I was bored and fell asleep in the middle of the movie.',
    'I was really fascinated with the movie.',
    'Even the actors looked really old and disinterested.',
    'I didn’t expect the reboot to be so good. The writers really cared.',
    'The movie had its upsides and downsides; overall it’s decent.',
    'What a rotten attempt at a comedy. Not a single joke lands.',
    'Launching on Netflix was brave. I appreciate binging on episodes.',
    'Mira esto es lo que hay en artifacts esta este YAML.',
], columns=['review'])

# Normalizar las reseñas
my_reviews['review_norm'] = my_reviews['review'].apply(normalize_text)

# Convertir texto a embeddings
my_reviews_features = BERT_text_to_embeddings(my_reviews['review_norm'], batch_size=4, disable_progress_bar=True)

# Escalar las características usando el scaler cargado
my_reviews_features_scaled = scaler.transform(my_reviews_features)

# Realizar predicciones
my_reviews_pred_prob = loaded_model.predict(pd.DataFrame(my_reviews_features_scaled))

# Imprimir resultados
for i, review in enumerate(my_reviews['review_norm'].str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')








