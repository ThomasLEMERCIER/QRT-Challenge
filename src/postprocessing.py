import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def compute_prediction(y_pred: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
    one_hot_encoder = OneHotEncoder(sparse_output=False, dtype=np.int32)

    y_pred = y_pred.reshape(-1, 1)
    y_pred = one_hot_encoder.fit_transform(y_pred)

    predictions = pd.DataFrame(y_pred, columns=["HOME_WINS","DRAW","AWAY_WINS"])
    predictions.index = x_test.index

    return predictions

def save_predictions(predictions: pd.DataFrame, path: str) -> None:
    predictions.to_csv(path)
