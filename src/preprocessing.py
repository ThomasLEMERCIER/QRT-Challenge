import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def remove_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=df.select_dtypes(include=["object"]).columns)

def encode_target_variable(y: pd.DataFrame) -> pd.DataFrame:
    y["target"] = y.to_numpy().nonzero()[1]
    y = y.drop(columns=y.columns.difference(["target"]))
    return y

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=categorical_columns)
    return df

def label_encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        df[column] = df[column].astype("category").cat.codes
    return df

def change_categorical_features_type(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        df[column] = df[column].astype("category")
    return df

def impute_missing_values(df: pd.DataFrame, strategy: str="mean", imputer: SimpleImputer=None, numeric_columns=None, rank='auto') -> tuple:
    old_df = df.copy()

    if imputer:
        df[numeric_columns] = imputer.transform(df[numeric_columns])
    else:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        num_na_per_column = df.isna().sum()
        full_na_columns = num_na_per_column[num_na_per_column == df.shape[0]].index
        numeric_columns = numeric_columns.difference(full_na_columns)

        imputer = SimpleImputer(strategy=strategy)
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Perform SVD to fill in missing values
    if rank is not None:
        nan_idx = np.isnan(old_df[numeric_columns])

        U, s, V = np.linalg.svd(df[numeric_columns], full_matrices=False)
        S = np.diag(s)

        if rank == "auto":
            # Use the s vector of singular values to find the best threshold that keeps the most information
            threshold = 0.7
            cumulative_sum = np.cumsum(s)
            threshold_index = np.argmax(cumulative_sum > threshold * cumulative_sum[-1])
            rank = threshold_index + 1

        print(f"Original rank: {np.linalg.matrix_rank(df[numeric_columns])} -> New rank: {rank}")

        # Impute the nan_idx values
        reconstructed_matrix = np.dot(U[:, :rank], np.dot(S[:rank, :rank], V[:rank, :]))
        imputed_matrix = np.where(nan_idx, reconstructed_matrix, df[numeric_columns])

        df[numeric_columns] = imputed_matrix
    return df, imputer, numeric_columns

def split_data(x: pd.DataFrame, y: pd.DataFrame, test_size: float=0.2, val_size: float=0.2) -> tuple:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=42)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def data_augmentation(df: pd.DataFrame, best_features: list) -> None:
    diff = []

    for feature in best_features:
        home_feature = "HOME_" + feature
        away_feature = "AWAY_" + feature

        diff.append(df[home_feature] - df[away_feature])

    diff = pd.concat(diff, axis=1)

    diff.columns = best_features + "_DIFF"

    df = pd.concat([df, diff], axis=1)

    return df

def remove_na_columns(df: pd.DataFrame, non_na_columns: list=None) -> pd.DataFrame:
    if non_na_columns is not None:
        df = df[non_na_columns]
    else:
        df = df.dropna(axis=1)
        non_na_columns = df.columns
    return df, non_na_columns

def find_knee_point(y):
    dy = np.diff(y)
    ddy = np.diff(dy)
    knee_idx = np.argmax(ddy) + 1
    return knee_idx