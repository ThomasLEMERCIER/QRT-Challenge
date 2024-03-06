import wandb
import argparse
import xgboost as xgb

from src.dataset import load_team_data
from src.evaluate import evaluate_model
from src.postprocessing import  compute_prediction, save_predictions
from src.preprocessing import impute_missing_values, split_data, remove_name_columns, encode_target_variable

def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost baseline for the QRT Challenge", prog="python baseline.py")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--wandb", type=bool, default=False)

    parser.add_argument("--num_boost_round", type=int, default=1000)
    parser.add_argument("--early_stopping_rounds", type=int, default=10)

    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_child_weight", type=float, default=1)
    parser.add_argument("--subsample", type=float, default=1)
    parser.add_argument("--colsample_bytree", type=float, default=1)
    parser.add_argument("--colsample_bylevel", type=float, default=1)
    parser.add_argument("--colsample_bynode", type=float, default=1)
    parser.add_argument("--l2_reg", type=float, default=1)
    parser.add_argument("--l1_reg", type=float, default=0)
    parser.add_argument("--max_leaves", type=int, default=0)

    args = parser.parse_args()

    args_xgb = argparse.Namespace(num_boost_round=args.num_boost_round, 
                                    early_stopping_rounds=args.early_stopping_rounds
    )

    booster_params = {
        "eta": args.eta,
        "gamma": args.gamma,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "colsample_bylevel": args.colsample_bylevel,
        "colsample_bynode": args.colsample_bynode,
        "lambda": args.l2_reg,
        "alpha": args.l1_reg,
        "max_leaves": args.max_leaves
    }

    return args, args_xgb, booster_params

if __name__ == "__main__":

    # === Parse command line arguments ===
    args, args_xgb, booster_params = parse_args()
    # ====================================

    # === Initialize Weights and Biases ===
    if args.wandb:
        wandb.init(project="QRT-Challenge-baseline", entity="thomas_l")
    # =====================================

    # === Load and preprocess data ===
    x, y = load_team_data()
    x = remove_name_columns(x)
    y = encode_target_variable(y)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, y)

    x_train, imputer, columns = impute_missing_values(x_train)
    x_val, _, _ = impute_missing_values(x_val, imputer=imputer, numeric_columns=columns)
    x_test, _, _ = impute_missing_values(x_test, imputer=imputer, numeric_columns=columns)
    # ================================

    # === Define model parameters ===
    booster_params["booster"] = "gbtree"
    booster_params["device"] = "cuda"

    booster_params["objective"] = "multi:softmax"
    booster_params["num_class"] = 3
    booster_params["eval_metric"] = "merror"
    booster_params["tree_method"] = "hist"
    booster_params["verbosity"] = 0

    num_boost_round = args_xgb.num_boost_round
    early_stopping_rounds = args_xgb.early_stopping_rounds
    # ===============================

    # === Log model parameters ===
    if args.wandb:
        wandb.config.update(booster_params)
        wandb.config.update({"num_boost_round": num_boost_round, "early_stopping_rounds": early_stopping_rounds})
    # ============================

    # === Train model ===
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    dtest = xgb.DMatrix(x_test, label=y_test)
    evals = [(dtrain, "train"), (dval, "val")]
    bst = xgb.train(booster_params, dtrain, num_boost_round=num_boost_round, evals=evals, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
    # ===================

    # === Evaluate model ===
    acc_val = evaluate_model(bst, dval, y_val)
    acc_test = evaluate_model(bst, dtest, y_test)

    print(f"Validation accuracy: {acc_val:.4f}")
    print(f"Test accuracy: {acc_test:.4f}")
    # ======================

    # === Log evaluation metrics ===
    if args.wandb:
        wandb.log({"val_acc": acc_val, "test_acc": acc_test})
    # ==============================

    # === Submit predictions ===
    if args.submit:
        x_test = load_team_data(train=False)
        x_test = remove_name_columns(x_test)
        x_test, _ = impute_missing_values(x_test, imputer=imputer)

        dtest = xgb.DMatrix(x_test)
        y_pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration))
        predictions = compute_prediction(y_pred, x_test)

        save_predictions(predictions, "xgboost.csv")
    # ==========================
