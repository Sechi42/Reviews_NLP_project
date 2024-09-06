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