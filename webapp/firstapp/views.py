from django.shortcuts import render, HttpResponse, redirect
import joblib
import json, os
import numpy as np
import pandas as pd
import psycopg2
from .models import ehr

# Create your views here.

# Stage 2: Integration
def index(request):
    return render(request, "index.html")

# Stage 5: Modification of Views.py file in Django. 
def result(request):
    model_path = os.path.join(os.path.dirname(__file__), '../models/models.joblib')
    model = joblib.load(model_path)  # We need to call the ML model here

    ############### Codes to get sex column in text #########
    sex_input = request.GET['sex'].lower() 
    if sex_input == 'male':
        sex_value = 1
    elif sex_input == 'female':
        sex_value = 0
    else:
        return HttpResponse("Invalid input for 'sex'. Please enter 'male' or 'female'.")
    
    ########################################################################   
    data_list = []  # Creating an empty list
    data_list.append(float(request.GET['HAEMATOCRIT']))
    data_list.append(float(request.GET['HAEMOGLOBIN']))
    data_list.append(float(request.GET['ERYTHROCYTE']))
    data_list.append(float(request.GET['LEUCOCYTE']))
    data_list.append(int(request.GET['THROMBOCYTE']))
    data_list.append(float(request.GET['MCH']))
    data_list.append(float(request.GET['MCHC']))
    data_list.append(float(request.GET['MCV']))
    data_list.append(int(request.GET['AGE']))
    data_list.append(sex_value)
    
    # Complete data present in list will be passed to ML model now
    answer = model.predict([data_list]).tolist()[0]  # This line will predict & display the result.
    rounded_answer = round(answer)  # Round the result to the nearest integer (0 or 1)
    
    # Determine the result message based on the rounded answer
    result_message = "In-patient" if rounded_answer == 1 else "Out-patient"

    b = ehr(
        HAEMATOCRIT=request.GET['HAEMATOCRIT'], 
        HAEMOGLOBIN=request.GET['HAEMOGLOBIN'], 
        ERYTHROCYTE=request.GET['ERYTHROCYTE'],
        LEUCOCYTE=request.GET['LEUCOCYTE'], 
        THROMBOCYTE=request.GET['THROMBOCYTE'], 
        MCH=request.GET['MCH'],
        MCHC=request.GET['MCHC'], 
        MCV=request.GET['MCV'], 
        AGE=request.GET['AGE'], 
        SEX=sex_value,
    )
    b.save()
    
    return render(request, "result.html", {'answer': result_message}) #result_message
