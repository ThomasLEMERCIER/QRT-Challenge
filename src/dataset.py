import pandas as pd
import numpy as np

def load_data(train=True):
    if train:
        away_team_statistics = pd.read_csv("data/train/train_away_team_statistics_df.csv", index_col=0)
        home_team_statistics = pd.read_csv("data/train/train_home_team_statistics_df.csv", index_col=0)
        y = pd.read_csv("data/train/Y_train.csv", index_col=0)

        train_away = away_team_statistics
        train_home = home_team_statistics

        train_home.columns = "HOME_" + train_home.columns
        train_away.columns = "AWAY_" + train_away.columns

        x = pd.concat([train_away, train_home], axis=1, join="inner")
        y = y.loc[x.index]

        return x, y
    else:
        away_team_statistics = pd.read_csv("data/test/test_away_team_statistics_df.csv", index_col=0)
        home_team_statistics = pd.read_csv("data/test/test_home_team_statistics_df.csv", index_col=0)

        test_away = away_team_statistics
        test_home = home_team_statistics

        test_home.columns = "HOME_" + test_home.columns
        test_away.columns = "AWAY_" + test_away.columns

        x_test = pd.concat([test_away, test_home], axis=1, join="inner")

        return x_test
