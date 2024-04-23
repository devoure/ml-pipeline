import mlflow
import random


def get_data():
    mlflow.set_tracking_uri('http://192.168.1.201:5000')

    experiment_id = 'Housing'
    runs = mlflow.search_runs(experiment_names=[experiment_id],
                              filter_string="attributes.status = 'FINISHED'")

    agents = []
    index = 0
    for run in runs.iterrows():
        agent = {}
        agent.update({"pic": 'img/agent{}.png'.format(random.randint(1, 3))})
        agent.update({"index": index})
        agent.update({"run_name": run[1]['tags.mlflow.runName']})
        agent.update({"mae": run[1]['metrics.mean_absolute_error']})
        agent.update({"overfitting": round(run[1]['metrics.overfitting'], 2)})
        agent.update({"rmse": run[1]['metrics.rmse']})
        agent.update({"acc": round(run[1]['metrics.accuracy'], 2)})
        agent.update({"dev": run[1]['params.developer']})
        agent.update({"algo": run[1]['params.algorithm']})

        agents.append(agent)
        index = index + 1

    context = {"agents": agents}

    return context




def get_predictions(run_id):
    mlflow.set_tracking_uri('http://192.168.1.201:5000')

    experiment_id = 'Housing'
    run = mlflow.search_runs(experiment_names=[experiment_id],
                              filter_string="attributes.run_id = '{}'".format(run_id))
    for row in run.iterrows():
        print(row[1]["tags.mlflow.log-model.history"])



def get_run():
    mlflow.set_tracking_uri('http://192.168.1.201:5000')

    experiment_id = 'Housing'
    runs = mlflow.search_runs(experiment_names=[experiment_id],
                              filter_string="attributes.status = 'FINISHED'")

    for run in runs.iterrows():
        agent = {}
        agent.update({"run_id": run[1]['run_id']}) 

    print(runs)
    print("SSS>>", agent)
    get_predictions(agent["run_id"])

get_run()
