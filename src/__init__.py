from .models import XGBoost, LinearRegression
from .crossval import XGBOOST_PARAMS, REG_LIN_PARAMS, XGBOOST_RANK_PARAMS

CrossValParams = {
    "xgboost": XGBOOST_PARAMS,
    "xgboost_rank": XGBOOST_RANK_PARAMS,
    "reg_lin": REG_LIN_PARAMS,
    "test": REG_LIN_PARAMS,
}

ModelTypes = {
    "xgboost": XGBoost,
    "xgboost_rank": XGBoost,
    "reg_lin": LinearRegression,
    "test": LinearRegression,
}
