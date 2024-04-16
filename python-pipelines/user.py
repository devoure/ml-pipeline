from kfp.dsl import component, pipeline
import kfp


@component(
        base_image="devourey/scikit-sandbox:latest"
        )
def load_data():
    import pandas as pd
    from minio import Minio
    import io

    minio_client = Minio(
        "192.168.1.201:9090",
        access_key="minio",
        secret_key="minio123",
        secure=False
        )
    
    user_id = "34445"

    # Bucket name with user id
    bucket_name = user_id
    thedataset = 'Housing.csv'

    try:
        res = minio_client.get_object('housing', thedataset)

    except Exception as err:
        print(err)

    housing = pd.read_csv(io.BytesIO(res.data))
    data = pd.DataFrame(housing)

    encodeddata = data.to_csv(index=False).encode("utf-8")

    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
        minio_client.put_object(
                bucket_name,
                'data.csv',
                data=io.BytesIO(encodeddata),
                length=len(encodeddata),
                content_type='application/csv'
                )
    except Exception as err:
        print(err)

@component(
        base_image="devourey/scikit-sandbox:latest"
        )
def process_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from minio import Minio
    import io
    from sklearn.preprocessing import StandardScaler


    minio_client = Minio(
        "192.168.1.201:9090",
        access_key="minio",
        secret_key="minio123",
        secure=False
        )

    user_id = "34445"

    bucket_name = user_id
    try:
        res = minio_client.get_object(bucket_name, 'data.csv')

    except Exception as err:
        print(err)

    data = pd.read_csv(io.BytesIO(res.data))
    data = pd.DataFrame(data)

    columns_to_drop = ["hotwaterheating", "guestroom", "parking", "airconditioning"]
    data = data.drop(columns=columns_to_drop)
    data.replace({'yes':1, 'no':0}, inplace=True)
    data.replace({'semi-furnished':50, 'unfurnished':0, 'furnished':100}, inplace=True)

    X = data[["area","bedrooms", "stories", "furnishingstatus", "mainroad"]]
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_scale_fields = ["area", "bedrooms", "stories", "furnishingstatus"]
    scaler = StandardScaler()

    X_train[X_scale_fields] = scaler.fit_transform(X_train[X_scale_fields])
    X_test[X_scale_fields] = scaler.transform(X_test[X_scale_fields])

    X_train_bytes = X_train.to_csv(index=False).encode("utf-8")
    X_test_bytes = X_test.to_csv(index=False).encode("utf-8")
    y_train_bytes = y_train.to_csv(index=False).encode("utf-8")
    y_test_bytes = y_test.to_csv(index=False).encode("utf-8")

    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
        minio_client.put_object(
                bucket_name,
                'X-train.csv',
                data=io.BytesIO(X_train_bytes),
                length=len(X_train_bytes),
                content_type = "application/csv"
                )
        minio_client.put_object(
                bucket_name,
                'X-test.csv',
                data=io.BytesIO(X_test_bytes),
                length=len(X_test_bytes),
                content_type = "application/csv"
                )
        minio_client.put_object(
                bucket_name,
                'y-train.csv',
                data=io.BytesIO(y_train_bytes),
                length=len(y_train_bytes),
                content_type = "application/csv"
                )
        minio_client.put_object(
                bucket_name,
                'y-test.csv',
                data=io.BytesIO(y_test_bytes),
                length=len(y_test_bytes),
                content_type = "application/csv"
                )

    except Exception as err:
        print(err)


@component(
        base_image="devourey/scikit-pipeline:latest",
        )
def train_data():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from minio import Minio
    import io
    import mlflow
    import mlflow.sklearn
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    user_id = "34445"

    minio_client = Minio(
        "192.168.1.201:9090",
        access_key="minio",
        secret_key="minio123",
        secure=False
        )

    bucket_name = user_id
    try:
        res_for_X = minio_client.get_object(bucket_name, 'X-train.csv')
        res_for_y = minio_client.get_object(bucket_name, 'y-train.csv')
        res_for_X_test = minio_client.get_object(bucket_name, 'X-test.csv')
        res_for_y_test = minio_client.get_object(bucket_name, 'y-test.csv')

    except Exception as err:
        print(err)

    X_train = pd.read_csv(io.BytesIO(res_for_X.data))
    y_train = pd.read_csv(io.BytesIO(res_for_y.data))
    X_test = pd.read_csv(io.BytesIO(res_for_X_test.data))
    y_test = pd.read_csv(io.BytesIO(res_for_y_test.data))

    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)

    # Set tracking URI to the location of your MLflow server
    mlflow.set_tracking_uri("http://192.168.1.201:5000")

    # Set up a shared experiment
    experiment_name = "Housing"
    mlflow.set_experiment(experiment_name)

    # Set user parameters
    user_params = {
    "developer": "Athumani Bakari",
    "algorithm": "Linear Regression",
    }

    # User 1
    with mlflow.start_run():
        mlflow.log_params(user_params)

        lr = LinearRegression()
        model = lr.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_2 = model.predict(X_train)



        # Compute evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        overfitting = (r2_score(y_train, y_pred_2) * 100)


        mlflow.log_metric("mean_absolute_error", mae)
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("rmse", mse)
        mlflow.log_metric("accuracy", (r2 * 100))
        mlflow.log_metric("overfitting", overfitting)



        mlflow.sklearn.log_model(model, "model.pkl")




@pipeline(
        name="Housing ML Pipeline",
        description=" A simple yet challenging project, to predict the housing price based on certain factors like house area, bedrooms, furnished, nearness to mainroad, etc. The dataset is small yet, it's complexity arises due to the fact that it has strong multicollinearity. "
        )
def pl_pipeline():
    load_data_component = load_data()
    process_data_component = process_data().after(load_data_component)
    train_data_component = train_data().after(process_data_component)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pl_pipeline, "housing_lr.yaml")
