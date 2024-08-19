import os
import mlflow
import argparse
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import re
import torch
import math
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel
import transformers

# Función para evaluar el rendimiento del clasificador
def eval_function(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='binary')
    recall = recall_score(actual, pred, average='binary')
    f1 = f1_score(actual, pred, average='binary')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    return metrics

# Función para cargar los datos
def load_data():
    npz_path = r"C:\Users\evolu\Desktop\Sergio\DATA_SCIENTIST\Sprint_14\Project_14\MLFlow-Manage-ML-Experiments\datasets\features_9.npz"
    tsv_path = r"C:\Users\evolu\Desktop\Sergio\DATA_SCIENTIST\Sprint_14\Project_14\Experiments\imdb_reviews.tsv"
    
    if os.path.exists(npz_path):
        print("Cargando características desde el archivo .npz...")
        with np.load(npz_path) as data:
            train_features = data['train_features_9']
            test_features = data['test_features_9']

        # Cargar los objetivos desde el archivo .tsv
        print("Cargando objetivos desde el archivo .tsv...")
        data = pd.read_csv(tsv_path, sep='\t')

        df_reviews_train = data.query('ds_part == "train"').copy()
        df_reviews_test = data.query('ds_part == "test"').copy()  

        TARGET = "pos"
        y_train = df_reviews_train[TARGET]
        y_test = df_reviews_test[TARGET]
            
    else:
        print("Cargando y procesando datos desde el archivo .tsv...")
        data = pd.read_csv(tsv_path, sep='\t')

        pattern = r"[^'a-z\s']"
        data['review_norm'] = data['review'].str.lower().apply(lambda x: re.sub(pattern, " ", x))    

        df_reviews_train = data.query('ds_part == "train"').copy()
        df_reviews_test = data.query('ds_part == "test"').copy()  

        TARGET = "pos"
        X_train = df_reviews_train['review_norm']
        y_train = df_reviews_train[TARGET]
        X_test = df_reviews_test['review_norm']
        y_test = df_reviews_test[TARGET]

        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        model = transformers.BertModel.from_pretrained('bert-base-uncased')

        def BERT_text_to_embeddings(texts, max_length=512, batch_size=15, force_device='cuda', disable_progress_bar=False):
            ids_list = []
            attention_mask_list = []

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

            if force_device is not None:
                device = torch.device(force_device)
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model.to(device)

            embeddings = []
            for i in tqdm(range(math.ceil(len(ids_list) / batch_size)), disable=disable_progress_bar):
                ids_batch = torch.LongTensor(ids_list[batch_size * i:batch_size * (i + 1)]).to(device)
                attention_mask_batch = torch.LongTensor(attention_mask_list[batch_size * i:batch_size * (i + 1)]).to(device)

                with torch.no_grad():
                    model.eval()
                    batch_embeddings = model(input_ids=ids_batch, attention_mask=attention_mask_batch)
                embeddings.append(batch_embeddings[0][:, 0, :].detach().cpu().numpy())

            return np.concatenate(embeddings)

        train_features = BERT_text_to_embeddings(X_train)
        test_features = BERT_text_to_embeddings(X_test)

    return train_features, test_features, y_train, y_test

def main(penalty, solver, C, max_iter, random_state=12345):
    train_features, test_features, y_train, y_test = load_data()
    
    mlflow.set_experiment("ML-Model-1")
    with mlflow.start_run():
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("solver", solver)
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)

        model = LogisticRegression(penalty=penalty, solver=solver, C=C, max_iter=max_iter, random_state=random_state)
        model.fit(train_features, y_train)
        y_pred = model.predict(test_features)
        
        # Evaluar el modelo
        metrics = eval_function(y_test, y_pred)
        
        # Registrar las métricas en MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        mlflow.sklearn.log_model(model, "trained_model") # model, folder
    
# Configuración de los argumentos de línea de comandos
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--penalty", "-p", type=str, default='l2')
    args.add_argument("--solver", "-s", type=str, default='lbfgs')
    args.add_argument("--C", "-C", type=float, default=1)
    args.add_argument("--max_iter", "-mi", type=int, default=200)
    parsed_args = args.parse_args()
    
    # Llamada a la función principal con los parámetros proporcionados
    main(parsed_args.penalty, parsed_args.solver, parsed_args.C, parsed_args.max_iter)
