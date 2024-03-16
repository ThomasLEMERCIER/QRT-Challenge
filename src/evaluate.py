from sklearn.metrics import accuracy_score
from xgboost import DMatrix, Booster
from lightgbm import Booster as LGBMBooster
import pandas as pd
import numpy as np

def evaluate_xgb_model(bst: Booster, d_test: DMatrix, y_test: pd.DataFrame) -> float:
    y_pred = bst.predict(d_test, iteration_range=(0, bst.best_iteration))
    return accuracy_score(y_test, y_pred), y_pred

def evaluate_lgb_model(model: LGBMBooster, x_test, y_test):
    y_pred = np.argmax(model.predict(x_test, num_iteration=model.best_iteration), axis=1)
    return accuracy_score(y_test, y_pred), y_pred
