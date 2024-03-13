import wandb
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from src.dataset import load_team_data, load_agg_player_data
from src.postprocessing import  compute_prediction, save_predictions
from src.preprocessing import impute_missing_values, split_data, remove_name_columns, encode_target_variable, remove_na_columns, find_knee_point, data_augmentation

def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost baseline for the QRT Challenge", prog="python baseline.py")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--wandb", type=bool, default=False)

    parser.add_argument("--knee_points", type=int, default=1)

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")

    args = parser.parse_args()

    args_svm = argparse.Namespace(C=args.C, gamma=args.gamma)

    return args, args_svm

if __name__ == "__main__":

    # === Parse command line arguments ===
    args, args_svm = parse_args()
    # ====================================

    # === Initialize Weights and Biases ===
    if args.wandb:
        wandb.init(project="QRT-Challenge-svm_rbf", entity="thomas_l")
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

    best_features = pd.read_csv('best_features_team_agg_based.csv').values.flatten()

    x_train = data_augmentation(x_train, best_features)
    x_val = data_augmentation(x_val, best_features)
    x_test = data_augmentation(x_test, best_features)

    best_features = list(best_features)

    columns_to_keep = columns_to_keep + [best_feature + "_DIFF" for best_feature in best_features] + ["HOME_" + best_feature for best_feature in best_features] + ["AWAY_" + best_feature for best_feature in best_features]
    list(set(columns_to_keep))

    x_train = x_train[columns_to_keep]
    x_val = x_val[columns_to_keep]
    x_test = x_test[columns_to_keep]
    # =============================

    # === Define model parameters ===
    C = args_svm.C
    kernel = "rbf"
    degree = 0
    gamma = args_svm.gamma
    coef0 = 0.
    # ===============================

    # === Log model parameters ===
    if args.wandb:
        wandb.config.update({"knee_points": index_knee})
        wandb.config.update({"C": C, "kernel": kernel, "degree": degree, "gamma": gamma, "coef0": coef0})
    # ============================

    # === Train model ===
    model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None)
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

        save_predictions(predictions, "svm_rbf.csv")
    # ==========================
