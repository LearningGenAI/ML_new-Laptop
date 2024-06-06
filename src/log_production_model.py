from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from http import client
from get_data import read_params
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib
import os
import mlflow.sklearn

def log_production_model(config_path):
    config = read_params(config_path)
    # Print config for debugging
    print("Config:", config)

    mlflow_config = config["mlflow_config"]
    model_name = mlflow_config["registered_model_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    runs = mlflow.search_runs(experiment_ids='1')
    lowest = runs["metrics.mae"].sort_values(ascending=True)[0]
    lowest_run_id = runs[runs["metrics.mae"] == lowest]["run_id"][0]

    client = MlflowClient()
    logged_model = None

    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        if mv["run_id"] == lowest_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(name=model_name, version=current_version, stage="Production")
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(name=model_name, version=current_version, stage="Staging")

    if logged_model is not None:
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        model_dirs = config["model_dir"]

        # Debugging: Print model_dirs to ensure it's correct
        print("Model Dirs:", model_dirs)

        model_path = model_dirs["model_dir"]

        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(loaded_model, model_path)
        
    else:
        print("No logged model found.")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)
