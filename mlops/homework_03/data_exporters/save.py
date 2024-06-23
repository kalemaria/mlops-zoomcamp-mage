import pickle
import mlflow

mlflow.set_tracking_uri("http://mlflow:5000") # because the mlflow container is called 'mlflow' in the docker_compose.yml and the exposed port is 5000 (mage is running inside localhost)
mlflow.set_experiment("ncy-taxi-experiment")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

# Training and logging should be in the same block. Split to be explicit.
@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here

    dv, lr = data
    dict_vectorizer_file = 'dict_vectorizer.bin'

    with mlflow.start_run():
        with open(dict_vectorizer_file, 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(dict_vectorizer_file)
        mlflow.sklearn.log_model(lr, 'model')

    print('OK')
