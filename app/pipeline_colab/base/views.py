from django.shortcuts import render
import mlflow

# Create your views here.
def home(request):
    mlflow.set_tracking_uri('http://192.168.1.201:5000')

    experiment_id = 'Housing'
    runs = mlflow.search_runs(experiment_ids=experiment_id, filter_string="status = 'FINISHED'")

    return render(request, 'home.html', {"runs":runs.values.tolist()})
