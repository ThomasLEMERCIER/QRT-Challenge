from .models import XGBoost, LinearRegression, SVM, MLP
from .crossval import BASELINE_PARAMS, XGBOOST_PARAMS, REG_LIN_PARAMS, XGBOOST_RANK_PARAMS, SVM_PARAMS, MLP_PARAMS

CrossValParams = {
    "baseline": BASELINE_PARAMS,
    "xgboost": XGBOOST_PARAMS,
    "xgboost_rank": XGBOOST_RANK_PARAMS,
    "reg_lin": REG_LIN_PARAMS,
    "svm": SVM_PARAMS,
    "mlp": MLP_PARAMS,
}

ModelTypes = {
    "baseline": XGBoost,
    "xgboost": XGBoost,
    "xgboost_rank": XGBoost,
    "reg_lin": LinearRegression,
    "svm": SVM,
    "mlp": MLP,
}
