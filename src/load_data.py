import os
import yaml #because we have 2 yaml file for automation
import pandas as pd #to work on data we require pandas & numpy
import numpy as np #this is also required for data structure.
import argparse #this will help us to pass the models from 1 file to another OR exchange data between files.
from get_data import read_params, get_data #This means we are reusing the codes from get_data & read_params

#step2:
def load_save_data(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    new_cols = [col.replace(" ","_") for col in df.columns] #this will replace empty spaces with _.
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path, sep=",", index=False, header=new_cols)

#Step 1: here we will create an entry door for automation. Params.yaml is required for automation.
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args=args.parse_args()
    load_save_data(config_path=parsed_args.config) #load save data is used as new function here