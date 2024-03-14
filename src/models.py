from src.dataset import load_team_data, load_agg_player_data
from src.evaluate import evaluate_model
from src.postprocessing import compute_prediction, save_predictions
from src.preprocessing import (
    impute_missing_values,
    split_data,
    remove_name_columns,
    encode_target_variable,
    data_augmentation,
    remove_na_columns,
    find_knee_point,
)
from src.crossval import CrossValidation

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression


class Pipeline:

    def __init__(self, args, run_name=None):
        self.model = None
        self.args = args
        self.run_name = run_name

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

    def run(self, crossval: CrossValidation, save_predictions=False):
        for fold, (
            (x_train, y_train),
            (x_val, y_val),
            (x_test, y_test),
            x_pred
        ) in enumerate(crossval.iterate()):
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

            acc_val, val_predictions = evaluate_model(self.model, dval, y_val)
            acc_test, test_predictions = evaluate_model(self.model, dtest, y_test)

            dpred = xgb.DMatrix(x_pred)
            y_pred = self.model.predict(dpred, iteration_range=(0, self.model.best_iteration))

            predictions = compute_prediction(y_pred, x_pred)

            if save_predictions:
                save_predictions(predictions, f"data/runs/{self.run_name}-fold{fold}.csv")

            yield acc_val, acc_test, predictions


# def baseline_model(args, run_name=None):
#     # === Load and preprocess data ===
#     x, y = load_team_data()
#     x = remove_name_columns(x)
#     y = encode_target_variable(y)
#     (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, y)

#     x_train, imputer, numeric_columns = impute_missing_values(x_train)
#     x_val, _, _ = impute_missing_values(
#         x_val, imputer=imputer, numeric_columns=numeric_columns
#     )
#     x_test, _, _ = impute_missing_values(
#         x_test, imputer=imputer, numeric_columns=numeric_columns
#     )
#     # ================================

#     # === Define model parameters ===
#     args["booster"] = "gbtree"
#     args["device"] = "cuda"

#     args["objective"] = "multi:softmax"
#     args["num_class"] = 3
#     args["eval_metric"] = "merror"
#     args["tree_method"] = "hist"
#     args["verbosity"] = 0

#     num_boost_round = args["num_boost_round"]
#     early_stopping_rounds = args["early_stopping_rounds"]
#     args.pop("num_boost_round")
#     args.pop("early_stopping_rounds")
#     # ===============================

#     # === Train model ===
#     dtrain = xgb.DMatrix(x_train, label=y_train)
#     dval = xgb.DMatrix(x_val, label=y_val)
#     dtest = xgb.DMatrix(x_test, label=y_test)
#     evals = [(dtrain, "train"), (dval, "val")]
#     bst = xgb.train(
#         args,
#         dtrain,
#         num_boost_round=num_boost_round,
#         evals=evals,
#         early_stopping_rounds=early_stopping_rounds,
#         verbose_eval=False,
#     )
#     # ===================

#     # === Evaluate model ===
#     acc_val = evaluate_model(bst, dval, y_val)
#     acc_test = evaluate_model(bst, dtest, y_test)

#     # ======================

#     predictions = None
#     if run_name:
#         x_test = load_team_data(train=False)
#         x_test = remove_name_columns(x_test)
#         x_test, _, _ = impute_missing_values(
#             x_test, imputer=imputer, numeric_columns=numeric_columns
#         )

#         dtest = xgb.DMatrix(x_test)
#         y_pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration))
#         predictions = compute_prediction(y_pred, x_test)

#         save_predictions(predictions, f"data/runs/baseline-{run_name}.csv")

#     return bst, acc_val, acc_test, predictions


# def team_agg_model(args, run_name=None):
#     # === Load and preprocess data ===
#     team_statistics, y = load_team_data()
#     player_statistics = load_agg_player_data()
#     x = pd.concat([team_statistics, player_statistics], axis=1, join="inner")

#     x = remove_name_columns(x)
#     y = encode_target_variable(y)
#     (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, y)

#     x_train, imputer, columns = impute_missing_values(x_train)
#     x_val, _, _ = impute_missing_values(x_val, imputer=imputer, numeric_columns=columns)
#     x_test, _, _ = impute_missing_values(
#         x_test, imputer=imputer, numeric_columns=columns
#     )
#     # ================================

#     # === Define model parameters ===
#     args["booster"] = "gbtree"
#     args["device"] = "cuda"

#     args["objective"] = "multi:softmax"
#     args["num_class"] = 3
#     args["eval_metric"] = "merror"
#     args["tree_method"] = "hist"
#     args["verbosity"] = 0

#     num_boost_round = args["num_boost_round"]
#     early_stopping_rounds = args["early_stopping_rounds"]
#     args.pop("num_boost_round")
#     args.pop("early_stopping_rounds")
#     # ===============================

#     # === Train model ===
#     dtrain = xgb.DMatrix(x_train, label=y_train)
#     dval = xgb.DMatrix(x_val, label=y_val)
#     dtest = xgb.DMatrix(x_test, label=y_test)
#     evals = [(dtrain, "train"), (dval, "val")]
#     bst = xgb.train(
#         args,
#         dtrain,
#         num_boost_round=num_boost_round,
#         evals=evals,
#         early_stopping_rounds=early_stopping_rounds,
#         verbose_eval=False,
#     )
#     # ===================

#     # === Evaluate model ===
#     acc_val = evaluate_model(bst, dval, y_val)
#     acc_test = evaluate_model(bst, dtest, y_test)

#     # ======================

#     # === Submit predictions ===
#     predictions = None
#     if run_name:
#         team_statistics = load_team_data(train=False)
#         player_statistics = load_agg_player_data(train=False)

#         x_test = pd.concat([team_statistics, player_statistics], axis=1, join="inner")
#         x_test = remove_name_columns(x_test)
#         x_test, _, _ = impute_missing_values(
#             x_test, imputer=imputer, numeric_columns=columns
#         )

#         dtest = xgb.DMatrix(x_test)
#         y_pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration))
#         predictions = compute_prediction(y_pred, x_test)

#         save_predictions(predictions, f"data/runs/team_agg-{run_name}.csv")
#     # ==========================

#     return bst, acc_val, acc_test, predictions


# def features_aug_model(args, run_name=None):
#     # === Load and preprocess data ===
#     team_statistics, y = load_team_data()
#     player_statistics = load_agg_player_data()
#     x = pd.concat([team_statistics, player_statistics], axis=1, join="inner")

#     x = remove_name_columns(x)
#     y = encode_target_variable(y)
#     (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, y)

#     x_train, imputer, columns = impute_missing_values(x_train)
#     x_val, _, _ = impute_missing_values(x_val, imputer=imputer, numeric_columns=columns)
#     x_test, _, _ = impute_missing_values(
#         x_test, imputer=imputer, numeric_columns=columns
#     )

#     best_features = pd.read_csv("best_features_team_agg_based.csv").values.flatten()

#     x_train = data_augmentation(x_train, best_features)
#     x_val = data_augmentation(x_val, best_features)
#     x_test = data_augmentation(x_test, best_features)
#     # ================================

#     # === Define model parameters ===
#     args["booster"] = "gbtree"
#     args["device"] = "cuda"

#     args["objective"] = "multi:softmax"
#     args["num_class"] = 3
#     args["eval_metric"] = "merror"
#     args["tree_method"] = "hist"
#     args["verbosity"] = 0

#     num_boost_round = args["num_boost_round"]
#     early_stopping_rounds = args["early_stopping_rounds"]
#     args.pop("num_boost_round")
#     args.pop("early_stopping_rounds")
#     # ===============================

#     # === Train model ===
#     dtrain = xgb.DMatrix(x_train, label=y_train)
#     dval = xgb.DMatrix(x_val, label=y_val)
#     dtest = xgb.DMatrix(x_test, label=y_test)
#     evals = [(dtrain, "train"), (dval, "val")]
#     bst = xgb.train(
#         args,
#         dtrain,
#         num_boost_round=num_boost_round,
#         evals=evals,
#         early_stopping_rounds=early_stopping_rounds,
#         verbose_eval=False,
#     )
#     # ===================

#     # === Evaluate model ===
#     acc_val = evaluate_model(bst, dval, y_val)
#     acc_test = evaluate_model(bst, dtest, y_test)

#     # ======================

#     # === Submit predictions ===
#     predictions = None
#     if run_name:
#         team_statistics = load_team_data(train=False)
#         player_statistics = load_agg_player_data(train=False)

#         x_test = pd.concat([team_statistics, player_statistics], axis=1, join="inner")
#         x_test = remove_name_columns(x_test)
#         x_test, _, _ = impute_missing_values(
#             x_test, imputer=imputer, numeric_columns=columns
#         )
#         x_test = data_augmentation(df=x_test, best_features=best_features)

#         dtest = xgb.DMatrix(x_test)
#         y_pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration))
#         predictions = compute_prediction(y_pred, x_test)

#         save_predictions(predictions, f"data/runs/features_aug-{run_name}.csv")
#     # ==========================

#     return bst, acc_val, acc_test, predictions


# def reg_lin_model(args, run_name=None):
#     # === Load and preprocess data ===
#     team_statistics, y = load_team_data()
#     player_statistics = load_agg_player_data()
#     x = pd.concat([team_statistics, player_statistics], axis=1, join="inner")

#     x = remove_name_columns(x)
#     y = encode_target_variable(y)
#     (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, y)

#     x_train, imputer, numeric_columns = impute_missing_values(x_train)
#     x_val, _, _ = impute_missing_values(
#         x_val, imputer=imputer, numeric_columns=numeric_columns
#     )
#     x_test, _, _ = impute_missing_values(
#         x_test, imputer=imputer, numeric_columns=numeric_columns
#     )

#     x_train, non_na_columns = remove_na_columns(x_train)
#     x_val, _ = remove_na_columns(x_val, non_na_columns=non_na_columns)
#     x_test, _ = remove_na_columns(x_test, non_na_columns=non_na_columns)

#     y_train = y_train.to_numpy().flatten()
#     y_val = y_val.to_numpy().flatten()
#     y_test = y_test.to_numpy().flatten()
#     # ================================

#     # === Load mutual information feature ===
#     scores = np.load("features_importance_mutual_info_based.npy")

#     order = np.argsort(scores)[::-1]
#     scores_sorted = scores[order]

#     k = 10
#     knee_indices = [find_knee_point(scores_sorted)]

#     for i in range(k - 1):
#         knee_indices.append(
#             find_knee_point(scores_sorted[knee_indices[i] :]) + knee_indices[i]
#         )
#     # =======================================

#     # === Extract best features ===
#     index_knee = args["knee_points"]
#     columns_selected = x_train.columns[order[: knee_indices[index_knee]]]

#     features = list(columns_selected)
#     features = set([feature[5:] for feature in features])

#     columns_to_keep = ["HOME_" + feature for feature in features] + [
#         "AWAY_" + feature for feature in features
#     ]

#     x_train = x_train[columns_to_keep]
#     x_val = x_val[columns_to_keep]
#     x_test = x_test[columns_to_keep]
#     # =============================

#     # === Define model parameters ===
#     l1_ratio = args["l1_ratio"]
#     C = args["C"]
#     multi_class = args["multi_class"]
#     # ===============================

#     # === Train model ===
#     model = LogisticRegression(
#         penalty="elasticnet",
#         solver="saga",
#         l1_ratio=l1_ratio,
#         C=C,
#         max_iter=2000,
#         multi_class=multi_class,
#         random_state=42,
#     )
#     model.fit(x_train, y_train)
#     # ===================

#     # === Evaluate model ===
#     acc_val = model.score(x_val, y_val)
#     acc_test = model.score(x_test, y_test)
#     # ======================

#     # === Submit predictions ===
#     predictions = None
#     if run_name:
#         team_statistics = load_team_data(train=False)
#         player_statistics = load_agg_player_data(train=False)

#         x_test = pd.concat([team_statistics, player_statistics], axis=1, join="inner")
#         x_test = remove_name_columns(x_test)
#         x_test, _, _ = impute_missing_values(
#             x_test, imputer=imputer, numeric_columns=numeric_columns
#         )
#         x_test, _ = remove_na_columns(x_test, non_na_columns=non_na_columns)

#         x_test = x_test[columns_to_keep]

#         y_pred = model.predict(x_test)
#         predictions = compute_prediction(y_pred, x_test)

#         save_predictions(predictions, f"data/runs/reg_lin-{run_name}.csv")

#     return model, acc_val, acc_test, predictions
