from django.shortcuts import render
import mlflow
#import pandas as pd

# Create your views here.
def home():
    mlflow.set_tracking_uri('http://192.168.1.201:5000')

    experiment_id = 'Housing'
    runs = mlflow.search_runs(experiment_names=[experiment_id], filter_string="attributes.status = 'FINISHED'")

    #for a in runs.iterrows():
    #    print(a[1]['metrics.accuracy'])
    print("Runs: ", len(runs.values.tolist()))

home()
