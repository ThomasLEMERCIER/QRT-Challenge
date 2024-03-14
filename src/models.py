from __future__ import annotations

from src.evaluate import evaluate_xgb_model
from src.postprocessing import compute_prediction
from src.crossval import CrossValidation

import xgboost as xgb
from sklearn.linear_model import LogisticRegression

class Pipeline:

    def __init__(self, model_type: Model, model_params):
        self.model = model_type(model_params)
        self.model_params = model_params

        self.iterator_params = self.model_params.get("knee_point", None)

    def run(self, crossval: CrossValidation):
        for fold, ((x_train, y_train), (x_val, y_val), (x_test, y_test), x_pred) in enumerate(crossval.iterate(self.iterator_params)):
            acc_val, acc_test, predictions = self.model.run(x_train, y_train, x_val, y_val, x_test, y_test, x_pred)
            yield acc_val, acc_test, predictions

class Model:
    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
        raise NotImplementedError

class XGBoost(Model):
    def __init__(self, args):
        super(XGBoost, self).__init__()
        args["booster"] = "gbtree"
        args["device"] = "cuda"

        args["objective"] = "multi:softmax"
        args["num_class"] = 3
        args["eval_metric"] = "merror"
        args["tree_method"] = "hist"
        args["verbosity"] = 0

        self.num_boost_round = args["num_boost_round"]
        self.early_stopping_rounds = args["early_stopping_rounds"]
        args.pop("num_boost_round")
        args.pop("early_stopping_rounds")

        self.args = args

    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        dtest = xgb.DMatrix(x_test, label=y_test)

        evals = [(dtrain, "train"), (dval, "val")]
        self.model = xgb.train(
            params=self.args, 
            dtrain=dtrain, 
            num_boost_round=self.num_boost_round, 
            evals=evals, 
            early_stopping_rounds=self.early_stopping_rounds, 
            verbose_eval=False
        )

        acc_val, _ = evaluate_xgb_model(self.model, dval, y_val)
        acc_test, _ = evaluate_xgb_model(self.model, dtest, y_test)

        dpred = xgb.DMatrix(x_pred)
        y_pred = self.model.predict(dpred, iteration_range=(0, self.model.best_iteration))
        predictions = compute_prediction(y_pred, x_pred)

        return acc_val, acc_test, predictions

class LinearRegression(Model):
    def __init__(self, args):
        super(LinearRegression, self).__init__()
        self.args = args

    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
        l1_ratio = self.args["l1_ratio"]
        C = self.args["C"]
        multi_class = self.args["multi_class"]

        y_train, y_val, y_test = y_train.values.ravel(), y_val.values.ravel(), y_test.values.ravel()

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=l1_ratio,
            C=C,
            max_iter=2000,
            multi_class=multi_class,
            random_state=42,
        )
        model.fit(x_train, y_train)

        acc_val = model.score(x_val, y_val)
        acc_test = model.score(x_test, y_test)

        y_pred = model.predict(x_pred)
        predictions = compute_prediction(y_pred, x_pred)

        return acc_val, acc_test, predictions
