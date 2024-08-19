import mlflow

# Establecer la URI de seguimiento de MLflow, que es la dirección del servidor de MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Crear un nuevo experimento en MLflow llamado 'Review_Prediction'
# Si el experimento ya existe, se utilizará el existente
exp_id = mlflow.create_experiment('Review_Prediction')

# Iniciar un nuevo run en MLflow con el nombre 'Logistic_Regression_Model'
# Este contexto se usa para agrupar las operaciones de logging en un único run
with mlflow.start_run(run_name='Logistic_Regression_Model') as run:
    
    # Establecer una etiqueta (tag) para el run, por ejemplo, para indicar la versión del modelo
    mlflow.set_tag("version", "1.0.0")
    
    # Aquí normalmente colocarías el código para entrenar el modelo, hacer predicciones, etc.
    pass  # Este 'pass' es un marcador de posición, se reemplazaría por el código del modelo

# El run se finaliza automáticamente al salir del bloque 'with'
# No es necesario llamar explícitamente a 'mlflow.end_run()' aquí, lo incluyo para claridad
mlflow.end_run()

# Parámetros del modelo de regresión logística
penalty = 'l2'        # Tipo de penalización (regularización L2)
C = 1                 # Inversa de la fuerza de regularización
solver = 'lbfgs'      # Algoritmo de optimización
max_iter = 200        # Número máximo de iteraciones para la convergencia
multi_class = 'multinomial'  # Estrategia para problemas multiclasificación

# Registrar los parámetros del modelo en MLflow
mlflow.log_param("penalty", penalty)
mlflow.log_param("C", C)
mlflow.log_param("solver", solver)
mlflow.log_param("max_iter", max_iter)
mlflow.log_param("multi_class", multi_class)
