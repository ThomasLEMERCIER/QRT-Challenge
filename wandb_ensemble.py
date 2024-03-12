from __future__ import annotations
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os

import pandas as pd
import wandb

from dataclasses import dataclass, field

from models import baseline_model, team_agg_model, features_aug_model

@dataclass
class Run:
    project: str
    run_name: str
    config: dict
    summary: dict

    # would need to delete and set dynamically
    val_acc: float
    test_acc: float

class Retriver:

    projects: list[str] = [
        "thomas_l/QRT-Challenge-reg_lin",
        "thomas_l/QRT-Challenge-features_aug",
        "thomas_l/QRT-Challenge-team_agg",
        "thomas_l/QRT-Challenge-Baseline",
    ]

    def __init__(self, projects: list[str] = None):
        self.projects = projects if projects else Retriver.projects

        try:
            self.api = wandb.Api()
            print("Successfully connected to the Wandb API")
        except:
            print("Failed to connect to the Wandb API")
            sys.exit()

    def fetch(self, *anchors: list[str]) -> pd.DataFrame:
        """Fetch runs from all given projects, store them by project and run name in a DataFrame"""
        data = {
            "project": [],
            "run_name": [],
            "config": [],
            "summary": [],
            **{anchor: [] for anchor in anchors},
        }

        for project in self.projects:
            runs = self.api.runs(project)
            print("Fetching runs from project:", project)
            for run in runs:
                data["project"].append(project)
                data["run_name"].append(run.name)
                data["config"].append(
                    {k: v for k, v in run.config.items() if not k.startswith("_")}
                )
                data["summary"].append(run.summary._json_dict)

                for anchor in anchors:
                    if anchor in data["config"][-1]:
                        data[anchor].append(run.summary._json_dict[anchor])
                    elif anchor in data["summary"][-1]:
                        data[anchor].append(run.summary._json_dict[anchor])
                    else:
                        data[anchor].append(None)

        self.dataframe = pd.DataFrame(data)

        return self.dataframe

    def get(self, anchor, mode="desc", top=5):
        if mode == "desc":
            return self.dataframe.sort_values(anchor, ascending=False).head(top)
        elif mode == "asc":
            return self.dataframe.sort_values(anchor, ascending=True).head(top)
        else:
            return self.dataframe

class Ensembler:

    def __init__(self, runs: list[Run], project_models: dict[str, callable]):
        self.runs = runs
        self.project_models = project_models

        self.predictions = []

    def from_csv(paths) -> Ensembler:
        ensembler = Ensembler(None, None)
        ensembler.predictions = [pd.read_csv(path, index_col=0) for path in paths]
        return ensembler

    def launch(self):
        for run in self.runs:
            print(f"Running {run.project}/{run.run_name}")
            model, val_accuracy, test_accuracy, predictions = self.project_models[run.project](run.config, run.run_name)
            print(f"Validation accuracy {run.val_acc:.4f} -> (new) {val_accuracy:.4f}")
            print(f"Test accuracy {run.test_acc:.4f} -> (new) {test_accuracy:.4f}")

            self.predictions.append(predictions)

    def aggregate(self):
        # Addition all cells
        ensemble_predictions = pd.DataFrame(self.predictions[0], columns=["HOME_WINS", "DRAW", "AWAY_WINS"], index=self.predictions[0].index)
        for i in range(1, len(self.predictions)):
            ensemble_predictions += self.predictions[i]
        print(ensemble_predictions)

        def vote(row):
            new_row = row.copy()

            lim_ratio = 0.6

            if row["AWAY_WINS"] != 0:
                ratio = row["HOME_WINS"]/row["AWAY_WINS"]
                if ratio < 1-lim_ratio:
                    new_row["AWAY_WINS"] = 1
                    new_row["HOME_WINS"] = 0
                    new_row["DRAW"] = 0
                elif ratio > 1+lim_ratio:
                    new_row["AWAY_WINS"] = 0
                    new_row["HOME_WINS"] = 1
                    new_row["DRAW"] = 0
                else:
                    new_row["AWAY_WINS"] = 0
                    new_row["HOME_WINS"] = 0
                    new_row["DRAW"] = 1
            else:
                if row["HOME_WINS"] >= row["DRAW"]:
                    new_row["HOME_WINS"] = 1
                    new_row["DRAW"] = 0
                    new_row["AWAY_WINS"] = 0
                elif row["DRAW"] > row["HOME_WINS"]:
                    new_row["HOME_WINS"] = 0
                    new_row["DRAW"] = 1
                    new_row["AWAY_WINS"] = 0

            return new_row

        # Make voting system : if equal number of wins and losses -> draw
        ensemble_predictions = ensemble_predictions.apply(vote, axis=1)

        # Count for each column the number of votes
        sum_ensemble_predictions = ensemble_predictions.sum()
        print(sum_ensemble_predictions)

        # Save the predictions
        ensemble_predictions.to_csv("ensemble_predictions.csv")

if __name__ == "__main__":

    project_models = {
        # "thomas_l/QRT-Challenge-reg_lin": lambda args: None,
        "thomas_l/QRT-Challenge-features_aug": features_aug_model,
        "thomas_l/QRT-Challenge-team_agg": team_agg_model,
        "thomas_l/QRT-Challenge-Baseline": baseline_model
    }

    paths = os.listdir('data/runs/')
    paths = [f'data/runs/{path}' for path in paths]
    print(paths)


    # Launch the ensembler
    if not paths:
        retriver = Retriver()
        retriver.fetch('val_acc', 'test_acc')

        best_runs = retriver.get('test_acc', top=10)
        print(best_runs)

        # Convert the DataFrame into a list of Run objects
        best_runs = [Run(**row) for _, row in best_runs.iterrows()]

        ensembler = Ensembler(best_runs, project_models)
        ensembler.launch()
        ensembler.aggregate()
    else:
        ensembler = Ensembler.from_csv(paths)
        ensembler.aggregate()