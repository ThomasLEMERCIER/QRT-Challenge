from .models import XGBoost, LinearRegression, SVM
from .crossval import BASELINE_PARAMS, XGBOOST_PARAMS, REG_LIN_PARAMS, XGBOOST_RANK_PARAMS

CrossValParams = {
    "baseline": BASELINE_PARAMS,
    "xgboost": XGBOOST_PARAMS,
    "xgboost_rank": XGBOOST_RANK_PARAMS,
    "reg_lin": REG_LIN_PARAMS,
    "svm": REG_LIN_PARAMS,
}

ModelTypes = {
    "baseline": XGBoost,
    "xgboost": XGBoost,
    "xgboost_rank": XGBoost,
    "reg_lin": LinearRegression,
    "svm": SVM,
}
