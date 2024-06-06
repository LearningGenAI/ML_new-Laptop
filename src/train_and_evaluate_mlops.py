import os
import yaml
import pandas as pd
import numpy as np
import argparse
from pkgutil import get_data
from get_data import read_params
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # mean_sq measures average difference statistical models (predicted valued & actual value) 
from sklearn.linear_model import ElasticNet
import joblib #this is used for saving the model
import json
import mlflow #for orchestration of mlops
from urllib.parse import urlparse

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

#Step2: copy the entire thing from train_and_evaluate.py file, since there will be no much changes.
def train_and_evaluate_mlops(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"]
    test_data_path = config["split_data"]["test_path"]#copy from here till random state from split date file.
    train_data_path = config["split_data"]["train_path"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    model_dir = config["model_dir"] #Model_dir will be the location where ML model would be saved

#now create 2 json report files mentioned in params file
#step3 after creating the models we need to save parameters inside that location
    alpha = config["estimators"]["ElasticNet"]["Params"]["alpha"] #params.jason values
    l1_ratio = config["estimators"]["ElasticNet"]["Params"]["l1_ratio"]  #Score values

    target = config["base"]["target_col"]#we need create one more Variable called target
    train = pd.read_csv(train_data_path, sep=",")#post target we need to read our train & test values
    test = pd.read_csv(test_data_path, sep=",")
 #now we need to split this data in terms of target & features. 
    train_x = train.drop(target, axis=1) #drop is to remove to target column
    test_x = test.drop(target, axis=1)

    train_y = train[target]#train & test_y is nothing but target column
    test_y = test[target]
    #print("Unique values in train_y:", train_y.unique())

 #######################step5###########################################
# We need to call mlflow_config & start the modification.
#Now some modification will be done above. 
    mlflow_config = config["mlflow_config"]# this will allow, URI tracking, run ID, experiment name, etc
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri) #this is to track MLflow orchrestration. 
 #######################step4 MLflow server - slight modification ###################################
  #lr is linear regrestion - From here the main code works. Now some modification will be done below.
    experiment_name = mlflow_config.get("experiment_name", "MyExperiment")
    mlflow.set_experiment(mlflow_config["experiment_name"])
#scripts from lr should run in a loop, for that we need to create a loop here.
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
#######################step4 MLflow server ###################################
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        lr.fit(train_x, train_y) #make sure the raw data 100% clean, if not this line will not work.

        predicted_values = lr.predict(test_x)  #we are storing the predictions inside a variable based on test_x

        (rmse, mae, r2) = eval_metrics(test_y, predicted_values) #to check the accuracy of this ML model
#after creating the above varibale metrics (rmse,mae,r2), to trck in MLops orchestration follow the below method.
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
     
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
#we are commenting the below so tht score file will be stored in MLops orchestrated platform
    # score_file = config["reports"]["score"] #instead of print, we can save the results in Json files
    # params_file = config["reports"]["params"]
#this score_file is one the parameter
#     with open(score_file, "w") as f:
#         score = {
#             "rmse" : rmse,
#             "mae" : mae,
#             "r2" : r2
#         }
#         json.dump(score, f, indent=4)

# ################Step 5##################################
#     with open(score_file, "w") as f:
#         score = {
#             "alpha" : alpha,
#             "l1_ratio" : l1_ratio,
#             "r2" : r2
#         }
#         json.dump(score, f, indent=4)
#now we need to create a close loop by creating a variable. 
#     tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme #this is the std format.
# #if artifacts is already created, don't create it again. we can put this condition in below "if" condition
#     if tracking_url_type_store != "file": #if artifact folder not there
#         mlflow.log_model(lr, "model", registered_model_name=mlflow_config["registered_model_name"])
#     else: #if artifact folder is availble don't create the folder
#         mlflow.sklearn.log_model(lr,"model") #now run this codes
        mlflow.sklearn.log_model(lr,"model")
        registered_model_name = mlflow_config["registered_model_name"]
        model_info = mlflow.register_model(f"runs:/{mlops_run.info.run_id}/model", registered_model_name)
        #print(f"Model registered as {registered_model_name}, version {model_info.version}")  

    os.makedirs(model_dir, exist_ok=True) #this is ti save this file with dat.
    model_path = os.path.join(model_dir, "models.joblib")
    joblib.dump(lr, model_path)


#Step1:
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args=args.parse_args()
    train_and_evaluate_mlops(config_path=parsed_args.config)