from sklearn.metrics import accuracy_score
from xgboost import DMatrix, Booster
import pandas as pd

def evaluate_model(bst: Booster, d_test: DMatrix, y_test: pd.DataFrame) -> float:
    y_pred = bst.predict(d_test, iteration_range=(0, bst.best_iteration))
    return accuracy_score(y_test, y_pred), y_pred
