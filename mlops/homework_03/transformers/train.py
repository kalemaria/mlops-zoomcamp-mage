from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def train(df, *args, **kwargs):
    """
    Trains a linear regression model with default parameters, using pick up and drop off locations separately.

    Args:
        data: The DataFrame output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        dv: DictVectorizer
        lr: trained linear regression model
    """
    # Specify your transformation logic here
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    train_dics = df[categorical].to_dict(orient='records')

    # Fit a dict vectorizer
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dics)

    target = 'duration'
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(lr.intercept_)

    return dv, lr


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'