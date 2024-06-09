import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    select_feature1 = kwargs.get('select_feature1')
    select_feature2 = kwargs.get('select_feature2')
    # Convert pickup and dropoff datetime columns to datetime type
    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    # Calculate the trip duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    # Filter out trips that are less than 1 minute or more than 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    # Convert location IDs to string to treat them as categorical features
    categorical = [select_feature1, select_feature2]
    df[categorical] = df[categorical].astype(str)

    return df
