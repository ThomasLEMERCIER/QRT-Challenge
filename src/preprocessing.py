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

def impute_missing_values(df: pd.DataFrame, strategy: str="mean", imputer: SimpleImputer=None, numeric_columns=None) -> tuple:
    if imputer:
        df[numeric_columns] = imputer.transform(df[numeric_columns])
    else:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        num_na_per_column = df.isna().sum()
        full_na_columns = num_na_per_column[num_na_per_column == df.shape[0]].index
        numeric_columns = numeric_columns.difference(full_na_columns)

        imputer = SimpleImputer(strategy=strategy)
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df, imputer, numeric_columns

def split_data(x: pd.DataFrame, y: pd.DataFrame, test_size: float=0.2, val_size: float=0.2) -> tuple:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=42)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def data_augmentation(x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame, best_features: list) -> None:
    diff_columns_train = []
    diff_columns_val = []
    diff_columns_test = []

    product_columns_train = []
    product_columns_val = []
    product_columns_test = []

    for feature in best_features:
        home_feature = "HOME_" + feature
        away_feature = "AWAY_" + feature

        diff_columns_train.append(x_train[home_feature] - x_train[away_feature])
        diff_columns_val.append(x_val[home_feature] - x_val[away_feature])
        diff_columns_test.append(x_test[home_feature] - x_test[away_feature])

        product_columns_train.append(x_train[home_feature] * x_train[away_feature])
        product_columns_val.append(x_val[home_feature] * x_val[away_feature])
        product_columns_test.append(x_test[home_feature] * x_test[away_feature])

    diff_columns_train = pd.concat(diff_columns_train, axis=1)
    diff_columns_val = pd.concat(diff_columns_val, axis=1)
    diff_columns_test = pd.concat(diff_columns_test, axis=1)

    diff_columns_train.columns = best_features + "_DIFF"
    diff_columns_val.columns = best_features + "_DIFF"
    diff_columns_test.columns = best_features + "_DIFF"

    x_train = pd.concat([x_train, diff_columns_train], axis=1)
    x_val = pd.concat([x_val, diff_columns_val], axis=1)
    x_test = pd.concat([x_test, diff_columns_test], axis=1)

    product_columns_train = pd.concat(product_columns_train, axis=1)
    product_columns_val = pd.concat(product_columns_val, axis=1)
    product_columns_test = pd.concat(product_columns_test, axis=1)

    product_columns_train.columns = best_features + "_PRODUCT"
    product_columns_val.columns = best_features + "_PRODUCT"
    product_columns_test.columns = best_features + "_PRODUCT"

    x_train = pd.concat([x_train, product_columns_train], axis=1)
    x_val = pd.concat([x_val, product_columns_val], axis=1)
    x_test = pd.concat([x_test, product_columns_test], axis=1)

    return x_train, x_val, x_test