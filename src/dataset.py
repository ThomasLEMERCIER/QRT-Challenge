import pandas as pd
import numpy as np

from .preprocessing import remove_name_columns

def load_team_data(train=True):
    if train:
        home_team_statistics = pd.read_csv("data/train/train_home_team_statistics_df.csv", index_col=0)
        away_team_statistics = pd.read_csv("data/train/train_away_team_statistics_df.csv", index_col=0)
        y = pd.read_csv("data/train/Y_train.csv", index_col=0)

        home_team_statistics.columns = "HOME_" + home_team_statistics.columns
        away_team_statistics.columns = "AWAY_" + away_team_statistics.columns

        x = pd.concat([away_team_statistics, home_team_statistics], axis=1, join="inner")
        y = y.loc[x.index]

        return x, y
    else:
        home_team_statistics = pd.read_csv("data/test/test_home_team_statistics_df.csv", index_col=0)
        away_team_statistics = pd.read_csv("data/test/test_away_team_statistics_df.csv", index_col=0)

        home_team_statistics.columns = "HOME_" + home_team_statistics.columns
        away_team_statistics.columns = "AWAY_" + away_team_statistics.columns

        x_test = pd.concat([away_team_statistics, home_team_statistics], axis=1, join="inner")

        return x_test
    
def load_agg_player_data(train=True):
    if train:
        home_player_statistics = pd.read_csv("data/train/train_home_player_statistics_df.csv", index_col=0)
        away_player_statistics = pd.read_csv("data/train/train_away_player_statistics_df.csv", index_col=0)

        home_player_statistics.columns = "HOME_" + home_player_statistics.columns
        away_player_statistics.columns = "AWAY_" + away_player_statistics.columns

        home_player_statistics = remove_name_columns(home_player_statistics)
        away_player_statistics = remove_name_columns(away_player_statistics)

        home_player_statistics["match_id"] = home_player_statistics.index
        away_player_statistics["match_id"] = away_player_statistics.index

        home_player_statistics_grouped = home_player_statistics.groupby("match_id").mean()
        away_player_statistics_grouped = away_player_statistics.groupby("match_id").mean()

        player_statistics = pd.concat([home_player_statistics_grouped, away_player_statistics_grouped], axis=1, join="inner")

        return player_statistics
    else:
        home_player_statistics = pd.read_csv("data/test/test_home_player_statistics_df.csv", index_col=0)
        away_player_statistics = pd.read_csv("data/test/test_away_player_statistics_df.csv", index_col=0)

        home_player_statistics.columns = "HOME_" + home_player_statistics.columns
        away_player_statistics.columns = "AWAY_" + away_player_statistics.columns

        home_player_statistics = remove_name_columns(home_player_statistics)
        away_player_statistics = remove_name_columns(away_player_statistics)

        home_player_statistics["match_id"] = home_player_statistics.index
        away_player_statistics["match_id"] = away_player_statistics.index

        home_player_statistics_grouped = home_player_statistics.groupby("match_id").mean()
        away_player_statistics_grouped = away_player_statistics.groupby("match_id").mean()

        player_statistics = pd.concat([home_player_statistics_grouped, away_player_statistics_grouped], axis=1, join="inner")

        return player_statistics
