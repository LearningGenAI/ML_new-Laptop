#__init__.py is mandatory for this get data to run. Here OOP concept (Object oriented Programing).
import os #operating system
import yaml #because we have 2 yaml file for automation
import pandas as pd #to work on data we require pandas & numpy
import numpy as np #this is also required for data structure.
import argparse #this will help us to pass the models from 1 file to another OR exchange data between files.

#Step2: We need to define get_data in a function here.
def get_data(config_path):
    config = read_params(config_path) #this will read the parameters
    data_path = config["data_source"]["s3_source"] #copy the list details ([]) from params.yaml file.
    df = pd.read_csv(data_path, sep=',', encoding='utf-8') #telling interpeter its CSV file by this "sep=',',encoding='utf-8'"
    return df

def read_params(config_path):
    with open(config_path) as yaml_file:
        config=yaml.safe_load(yaml_file)
        return config

#Step 1: here we will create an entry door for automation. Params.yaml is required for automation.
if __name__=="__main__": #this is general syntax for any data structure.
    args = argparse.ArgumentParser() #this args variable has become an object which will help is passing modules from 1 file to another.
    args.add_argument("--config", default="params.yaml") #we are passing the computing file(params.yaml) for automation. We are making it as default file for automation.
#we generally pass the file using "--config" with above line all the values will be fetched from params
    parsed_args=args.parse_args() #this is another variable, to pass params.yaml.
    data = get_data(config_path=parsed_args.config) #From __main__ till this line is the entry door. It is user defined method called get_data. 
#try to run this in terminal using "python .\src\get_data.py ."