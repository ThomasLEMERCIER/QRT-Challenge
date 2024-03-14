from __future__ import annotations

from src.evaluate import evaluate_model
from src.postprocessing import compute_prediction
from src.preprocessing import find_knee_point
from src.crossval import CrossValidation

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression


class Pipeline:

    def __init__(self, model_type: Model, args, run_name=None):
        self.model = model_type(args)
        self.args = args
        self.run_name = run_name

    def run(self, crossval: CrossValidation, save_predictions=False):
        for fold, (
            (x_train, y_train),
            (x_val, y_val),
            (x_test, y_test),
            x_pred
        ) in enumerate(crossval.iterate()):
            acc_val, acc_test, predictions = self.model.run(x_train, y_train, x_val, y_val, x_test, y_test, x_pred)

            if save_predictions:
                save_predictions(predictions, f"data/runs/{self.run_name}-fold{fold}.csv")

            yield acc_val, acc_test, predictions

class Model:
    def __init__(self, args):
        self.args = args

    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
        raise NotImplementedError

class XGBoost(Model):

    def __init__(self, args):
        super(XGBoost, self).__init__(args)

        # === Define model parameters ===
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
            self.args,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )

        acc_val, _ = evaluate_model(self.model, dval, y_val)
        acc_test, _ = evaluate_model(self.model, dtest, y_test)

        dpred = xgb.DMatrix(x_pred)
        y_pred = self.model.predict(dpred, iteration_range=(0, self.model.best_iteration))

        predictions = compute_prediction(y_pred, x_pred)

        return acc_val, acc_test, predictions


class LinearRegression(Model):

    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
        scores = np.load("features_importance_mutual_info_based.npy")

        order = np.argsort(scores)[::-1]
        scores_sorted = scores[order]

        k = 10
        knee_indices = [find_knee_point(scores_sorted)]

        for i in range(k - 1):
            knee_indices.append(
                find_knee_point(scores_sorted[knee_indices[i] :]) + knee_indices[i]
            )
        # =======================================

        # === Extract best features ===
        index_knee = self.args["knee_points"]
        columns_selected = x_train.columns[order[: knee_indices[index_knee]]]

        features = list(columns_selected)
        features = set([feature[5:] for feature in features])

        columns_to_keep = ["HOME_" + feature for feature in features] + [
            "AWAY_" + feature for feature in features
        ]

        x_train = x_train[columns_to_keep]
        x_val = x_val[columns_to_keep]
        x_test = x_test[columns_to_keep]
        # =============================

        # === Define model parameters ===
        l1_ratio = self.args["l1_ratio"]
        C = self.args["C"]
        multi_class = self.args["multi_class"]
        # ===============================

        # === Train model ===
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
        # ===================

        # === Evaluate model ===
        acc_val = model.score(x_val, y_val)
        acc_test = model.score(x_test, y_test)

        y_pred = model.predict(x_test)
        predictions = compute_prediction(y_pred, x_test)

        return acc_val, acc_test, predictions