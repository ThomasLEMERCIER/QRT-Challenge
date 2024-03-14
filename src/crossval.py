from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pandas as pd

from src.dataset import load_team_data, load_agg_player_data
from src.postprocessing import compute_prediction, save_predictions
from src.preprocessing import (
    impute_missing_values,
    remove_name_columns,
    encode_target_variable,
    data_augmentation,
    remove_na_columns,
    find_knee_point,
)


class CrossValidation:

    def __init__(self, n_folds, random_state=42, data_augment=False, add_player=False):
        self.n_folds = n_folds
        self.skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)


        def process_data(split="train"):
            train = (split == "train")
            x, y = load_team_data(train=train)
            if add_player:
                x_player = load_agg_player_data(train=train)
                x = pd.concat([x, x_player], axis=1, join="inner")
            x = remove_name_columns(x)
            if train: y = encode_target_variable(y)

            # Impute missing values
            x, _, _ = impute_missing_values(x, rank="auto")

            if data_augment:
                best_features = pd.read_csv("best_features_team_agg_based.csv").values.flatten()
                x = data_augmentation(x, best_features)

            return x, y

        print("Loading training data...")
        self.x, self.y = process_data("train")
        print("Loading test data...")
        self.x_test, self.y_test = process_data("test")

    def iterate(self):
        for train_index, test_index in self.skf.split(self.x, self.y):
            x = self.x.iloc[train_index]
            y = self.y.iloc[train_index]

            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
            x_test = self.x.iloc[test_index]
            y_test = self.y.iloc[test_index]

            yield (x_train, y_train), (x_val, y_val), (x_test, y_test), self.x_test
