import os
import mlflow
import argparse
import time

# Función que calcula una métrica basada en los dos parámetros numéricos
def eval(p1, p2):
    output_metric = p1**2 + p2**2
    return output_metric


def main(inp1, inp2):
    mlflow.set_experiment("Demo_Experiment")
    #with mlflow.start_run(run_name='example_demo'):
    with mlflow.start_run():
        # Registrar la etiqueta antes de cualquier otra operación
        mlflow.set_tag("version", "1.0.0")
        
        # Registrar los parámetros en MLflow
        mlflow.log_param('param1', inp1)
        mlflow.log_param('param2', inp2)
        
        # Calcular la métrica y registrarla en MLflow
        metric = eval(p1=inp1, p2=inp2)
        mlflow.log_metric('Eval_Metric', metric)
        
        # Crear un archivo de artefacto y registrarlo en MLflow
        os.makedirs("dummy", exist_ok=True)
        with open("dummy/example.txt", "wt") as f:
            f.write("Artifact created as {}".format(time.asctime()))
        mlflow.log_artifact("dummy")      
 

# Configuración de los argumentos de línea de comandos
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--param1", "-p1", type=int, default=1)
    args.add_argument("--param2", "-p2", type=int, default=200)
    parsed_args = args.parse_args()
    
    # Llamada a la función principal con los parámetros proporcionados
    main(parsed_args.param1, parsed_args.param2)
