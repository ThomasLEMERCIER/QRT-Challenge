import wandb
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from src.dataset import load_team_data, load_agg_player_data
from src.postprocessing import  compute_prediction, save_predictions
from src.preprocessing import impute_missing_values, split_data, remove_name_columns, encode_target_variable, remove_na_columns, find_knee_point

def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost baseline for the QRT Challenge", prog="python baseline.py")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--wandb", type=bool, default=False)

    parser.add_argument("--knee_points", type=int, default=1)

    parser.add_argument("--l1_ratio", type=float, default=0.5)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--multi_class", type=str, default="multinomial")

    args = parser.parse_args()

    args_reg = argparse.Namespace(l1_ratio=args.l1_ratio, 
                                  C=args.C,
                                  multi_class=args.multi_class
    )

    return args, args_reg


if __name__ == "__main__":

    # === Parse command line arguments ===
    args, args_reg = parse_args()
    # ====================================

    # === Initialize Weights and Biases ===
    if args.wandb:
        wandb.init(project="QRT-Challenge-reg_lin", entity="thomas_l")
    # =====================================

    # === Load and preprocess data ===
    team_statistics, y = load_team_data()
    player_statistics = load_agg_player_data()
    x = pd.concat([team_statistics, player_statistics], axis=1, join='inner')

    x = remove_name_columns(x)
    y = encode_target_variable(y)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, y)

    x_train, imputer, numeric_columns = impute_missing_values(x_train)
    x_val, _, _ = impute_missing_values(x_val, imputer=imputer, numeric_columns=numeric_columns)
    x_test, _, _ = impute_missing_values(x_test, imputer=imputer, numeric_columns=numeric_columns)

    x_train, non_na_columns = remove_na_columns(x_train)
    x_val, _ = remove_na_columns(x_val, non_na_columns=non_na_columns)
    x_test, _ = remove_na_columns(x_test, non_na_columns=non_na_columns)

    y_train = y_train.to_numpy().flatten()
    y_val = y_val.to_numpy().flatten()
    y_test = y_test.to_numpy().flatten()
    # ================================

    # === Load mutual information feature ===
    scores = np.load("features_importance_mutual_info_based.npy")

    order = np.argsort(scores)[::-1]
    scores_sorted = scores[order]

    k = 10
    knee_indices = [find_knee_point(scores_sorted)]

    for i in range(k-1):
        knee_indices.append(find_knee_point(scores_sorted[knee_indices[i]:]) + knee_indices[i])
    # =======================================
        
    # === Extract best features ===
    index_knee = args.knee_points
    columns_selected = x_train.columns[order[:knee_indices[index_knee]]]

    features = list(columns_selected)
    features = set([feature[5:] for feature in features])

    columns_to_keep = ["HOME_" + feature for feature in features] + ["AWAY_" + feature for feature in features]

    x_train = x_train[columns_to_keep]
    x_val = x_val[columns_to_keep]
    x_test = x_test[columns_to_keep]
    # =============================

    # === Define model parameters ===
    l1_ratio = args_reg.l1_ratio
    C = args_reg.C
    multi_class = args_reg.multi_class
    # ===============================

    # === Log model parameters ===
    if args.wandb:
        wandb.config.update({"index_knee": index_knee})
        wandb.config.update({"l1_ratio": l1_ratio, "C": C, "multi_class": multi_class})
    # ============================

    # === Train model ===
    model = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, C=C, max_iter=2000, multi_class=multi_class, random_state=42)
    model.fit(x_train, y_train)
    # ===================

    # === Evaluate model ===
    acc_val = model.score(x_val, y_val)
    acc_test = model.score(x_test, y_test)

    print(f"Validation accuracy: {acc_val:.4f}")
    print(f"Test accuracy: {acc_test:.4f}")
    # ======================

    # === Log evaluation metrics ===
    if args.wandb:
        wandb.log({"val_acc": acc_val, "test_acc": acc_test})
    # ==============================

    # === Submit predictions ===
    if args.submit:
        team_statistics = load_team_data(train=False)
        player_statistics = load_agg_player_data(train=False)

        x_test = pd.concat([team_statistics, player_statistics], axis=1, join='inner')
        x_test = remove_name_columns(x_test)
        x_test, _, _ = impute_missing_values(x_test, imputer=imputer, numeric_columns=numeric_columns)
        x_test, _ = remove_na_columns(x_test, non_na_columns=non_na_columns)

        x_test = x_test[columns_to_keep]

        y_pred = model.predict(x_test)
        predictions = compute_prediction(y_pred, x_test)

        save_predictions(predictions, "reg_lin.csv")
    # ==========================
