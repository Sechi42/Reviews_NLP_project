import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Definir el nombre del experimento
experiment_name = "review_prediction"

# Asegurarte de que el experimento existe o crearlo
mlflow.set_experiment(experiment_name)

# Cargar características y objetivos
print("Cargando características desde el archivo .npz...")
features_train = np.load('datasets/features_9.npz')['train_features_9']
features_test = np.load('datasets/features_9.npz')['test_features_9']
print("Cargando objetivos desde el archivo .tsv...")
df = pd.read_csv('datasets/imdb_reviews.tsv', sep='\t')

df_reviews_train = df.query('ds_part == "train"').copy()
df_reviews_test = df.query('ds_part == "test"').copy()

train_targets = df_reviews_train['pos']
test_targets = df_reviews_test['pos']

# Entrenar el modelo de Regresión Logística directamente
lr = LogisticRegression(random_state=12345, solver='liblinear', C=1.0, penalty='l2')
lr.fit(features_train, train_targets)

# Hacer predicciones
predictions = lr.predict(features_test)

# Registrar el modelo y las métricas en MLflow
with mlflow.start_run() as run:
    mlflow.log_param("solver", 'liblinear')
    mlflow.log_param("C", 1.0)
    mlflow.log_param("penalty", 'l2')
    
    mlflow.log_metric("accuracy", metrics.accuracy_score(test_targets, predictions))
    mlflow.log_metric("precision", metrics.precision_score(test_targets, predictions))
    mlflow.log_metric("recall", metrics.recall_score(test_targets, predictions))
    mlflow.log_metric("f1_score", metrics.f1_score(test_targets, predictions))
    
    # Registrar el modelo
    mlflow.sklearn.log_model(lr, "logistic_regression_model")
    
    print("Modelo registrado exitosamente en MLflow")
    print(f"Accuracy: {metrics.accuracy_score(test_targets, predictions):.4f}")
    print(f"Precision: {metrics.precision_score(test_targets, predictions):.4f}")
    print(f"Recall: {metrics.recall_score(test_targets, predictions):.4f}")
    print(f"F1 Score: {metrics.f1_score(test_targets, predictions):.4f}")



