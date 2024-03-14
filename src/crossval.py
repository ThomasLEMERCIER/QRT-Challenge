from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.dataset import load_team_data, load_agg_player_data
from src.preprocessing import (
    impute_missing_values,
    remove_name_columns,
    encode_target_variable,
    data_augmentation,
    remove_na_columns,
    find_knee_point,
)

@dataclass
class CrossValidationParams:
    add_player: bool
    data_augment: bool
    remove_full_na: bool
    restrict_best_features: bool
    rank: str

    n_folds: int = 5
    random_state: int = 42
    k_max: int = 10

class CrossValidation:
    def __init__(self, params: CrossValidationParams):
        self.params = params
        self.skf = StratifiedKFold(n_splits=params.n_folds, random_state=params.random_state, shuffle=True)

        if self.params.restrict_best_features:
            self.load_features_importance()
        if self.params.data_augment:
            self.best_features = pd.read_csv("best_features_team_agg_based.csv").values.flatten()

        print("Loading training data...")
        self.x, self.y = self.process_data("train")
        print("Loading test data...")
        self.x_pred, _ = self.process_data("test")

    def iterate(self, knee_point=None):
        for train_index, test_index in self.skf.split(self.x, self.y):
            x = self.x.iloc[train_index]
            y = self.y.iloc[train_index]

            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
            x_test = self.x.iloc[test_index]
            y_test = self.y.iloc[test_index]
            x_pred = self.x_pred

            if self.params.restrict_best_features:
                columns_selected = self.x.columns[self.order[: self.knee_indices[knee_point]]]
                features = set([feature[5:] for feature in list(columns_selected)])

                columns_to_keep = ["HOME_" + feature for feature in features] + ["AWAY_" + feature for feature in features]

                if self.params.data_augment:
                    columns_to_keep = columns_to_keep + \
                                      [best_feature + "_DIFF" for best_feature in list(self.best_features)] + \
                                      ["HOME_" + best_feature for best_feature in list(self.best_features)] + \
                                      ["AWAY_" + best_feature for best_feature in list(self.best_features)]
                    
                columns_to_keep = list(set(columns_to_keep))

                x_train = x_train[columns_to_keep]
                x_val = x_val[columns_to_keep]
                x_test = x_test[columns_to_keep]
                x_pred = self.x_pred[columns_to_keep]

            
            yield (x_train, y_train), (x_val, y_val), (x_test, y_test), x_pred

    def process_data(self, split: str):
        train = (split == "train")

        x, y = load_team_data(train=train)
        if self.params.add_player:
            x_player = load_agg_player_data(train=train)
            x = pd.concat([x, x_player], axis=1, join="inner")

        x = remove_name_columns(x)

        if train: 
            y = encode_target_variable(y)

        x, _, _ = impute_missing_values(x, rank=self.params.rank)

        if self.params.data_augment:
            x = data_augmentation(x, self.best_features)

        if self.params.remove_full_na:
            x, _ = remove_na_columns(x)

        return x, y

    def load_features_importance(self):
        scores = np.load("features_importance_mutual_info_based.npy")
        self.order = np.argsort(scores)[::-1]
        scores_sorted = scores[self.order]

        self.knee_indices = [find_knee_point(scores_sorted)]
        for i in range(self.params.k_max):
            self.knee_indices.append(find_knee_point(scores_sorted[self.knee_indices[i] :]) + self.knee_indices[i])

BASELINE_PARAMS = CrossValidationParams(add_player=False, data_augment=False, remove_full_na=False, restrict_best_features=False, rank=None)
XGBOOST_PARAMS = CrossValidationParams(add_player=True, data_augment=True, remove_full_na=False, restrict_best_features=False, rank=None)
XGBOOST_RANK_PARAMS = CrossValidationParams(add_player=True, data_augment=True, remove_full_na=False, restrict_best_features=False, rank="auto")
REG_LIN_PARAMS = CrossValidationParams(add_player=True, data_augment=True, remove_full_na=True, restrict_best_features=True, rank=None)
SVM_PARAMS = CrossValidationParams(add_player=True, data_augment=True, remove_full_na=True, restrict_best_features=True, rank=None)
MLP_PARAMS = CrossValidationParams(add_player=True, data_augment=True, remove_full_na=True, restrict_best_features=True, rank=None)
