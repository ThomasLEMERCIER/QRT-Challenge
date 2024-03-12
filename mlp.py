import torch
import wandb
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.deep_learning import MLP, Dataset, train
from src.dataset import load_team_data, load_agg_player_data
from src.postprocessing import  compute_prediction, save_predictions
from src.preprocessing import impute_missing_values, split_data, remove_name_columns, encode_target_variable, remove_na_columns, find_knee_point

def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost baseline for the QRT Challenge", prog="python baseline.py")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--wandb", type=bool, default=False)

    parser.add_argument("--knee_points", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eta_min", type=float, default=0.)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    args = parser.parse_args()

    args_mlp = argparse.Namespace(hidden_dim=args.hidden_dim,
                                  dropout_rate=args.dropout_rate,
                                  n_epochs=args.n_epochs,
                                  lr=args.lr,
                                  eta_min=args.eta_min,
                                  weight_decay=args.weight_decay,
                                  label_smoothing=args.label_smoothing
    )

    return args, args_mlp

if __name__ == "__main__":
    torch.manual_seed(42)


    # === Parse command line arguments ===
    args, args_mlp = parse_args()
    # ===================================

    # === Initialize Weights and Biases ===
    if args.wandb:
        wandb.init(project="QRT-Challenge-mlp", entity="thomas_l")
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

    # === Define dataset ===
    batch_size = args.batch_size
    train_dataset = Dataset(x_train, y_train)
    val_dataset = Dataset(x_val, y_val)
    test_dataset = Dataset(x_test, y_test)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    # ======================

    # === Define model parameters ===
    hidden_dim = args_mlp.hidden_dim
    dropout_rate = args_mlp.dropout_rate
    n_epochs = args_mlp.n_epochs
    lr = args_mlp.lr
    eta_min = args_mlp.eta_min
    weight_decay = args_mlp.weight_decay
    label_smoothing = args_mlp.label_smoothing
    # ===============================

    # === Log model parameters ===
    if args.wandb:
        wandb.config.update({"knee_points": index_knee, "batch_size": batch_size})
        wandb.config.update({"hidden_dim": hidden_dim, "dropout_rate": dropout_rate, "n_epochs": n_epochs, "lr": lr, "eta_min": eta_min, "weight_decay": weight_decay, "label_smoothing": label_smoothing})
    # ============================

    # === Train model ===
    model = MLP(input_dim=x_train.shape[1], hidden_dim=hidden_dim, output_dim=3, dropout_rate=dropout_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=eta_min)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_model = train(model, optimizer, criterion, scheduler, train_dl, val_dl, n_epochs)
    # ===================

    # === Evaluate model ===
    model.load_state_dict(best_model)
    model.eval()

    acc_val = torch.mean((model(val_dataset.x).argmax(dim=1) == val_dataset.y).float()).item()
    acc_test = torch.mean((model(test_dataset.x).argmax(dim=1) == test_dataset.y).float()).item()

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

        x_test_tensor = torch.tensor(x_test.to_numpy()).float()
        y_pred = torch.argmax(model(x_test_tensor), dim=1).detach().numpy()
        predictions = compute_prediction(y_pred, x_test)

        save_predictions(predictions, "mlp.csv")
    # ==========================
