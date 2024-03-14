from __future__ import annotations

import sys
import os

from sklearn.metrics import accuracy_score
import pandas as pd
import wandb

from dataclasses import dataclass
from src.models import (
    Pipeline,
)  # baseline_model, team_agg_model, features_aug_model, reg_lin_model
from src.crossval import CrossValidation


@dataclass
class Run:
    project: str
    run_name: str
    config: dict
    summary: dict

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

    def fetch(self, *anchors: list[str]):
        """Fetch runs from all given projects, store them by project and run name in a DataFrame"""

        data = {
            "project": [],
            "run_name": [],
            "config": [],
            "summary": [],
            **{anchor: [] for anchor in anchors},
        }

        for project in self.projects:
            runs = self.api.runs(path=project)
            for run in runs:
                data["project"].append(project)
                data["run_name"].append(run.name)
                data["config"].append(run.config)
                data["summary"].append(run.summary)

                for anchor in anchors:
                    if anchor in data["config"][-1]:
                        data[anchor].append(run.config[anchor])
                    elif anchor in data["summary"][-1]:
                        data[anchor].append(run.summary[anchor])
                    else:
                        data[anchor].append(None)

        self.dataframe = pd.DataFrame(data)

    def get(self, anchor, mode="desc", top=10, force_project_diversity=True):
        if mode == "desc":
            best_runs = []
            if force_project_diversity:
                for project in self.projects:
                    best_runs.append(
                        self.dataframe[self.dataframe["project"] == project]
                        .sort_values(anchor, ascending=False)
                        .head(1)
                    )
            best_runs = pd.concat(best_runs)
            overall = (
                self.dataframe[self.dataframe.index.isin(best_runs.index) == False]
                .sort_values(anchor, ascending=False)
                .head(max(0, top - len(best_runs)))
            )
            return pd.concat([best_runs, overall])
        elif mode == "asc":
            best_runs = []
            if force_project_diversity:
                for project in self.projects:
                    best_runs.append(
                        self.dataframe[self.dataframe["project"] == project]
                        .sort_values(anchor, ascending=True)
                        .head(1)
                    )
            best_runs = pd.concat(best_runs)
            overall = (
                self.dataframe.filter(items=best_runs.index, axis=0)
                .sort_values(anchor, ascending=True)
                .head(max(0, top - len(best_runs)))
            )
            return pd.concat([best_runs, overall])
        else:
            raise self.dataframe


class Ensembler:

    def __init__(self, runs: list[Run], project_configs: dict[str, callable]):
        self.runs = runs
        self.project_configs = project_configs

        self.predictions = []
        self.models = []

    def from_csv(paths) -> Ensembler:
        ensembler = Ensembler(None, None)
        ensembler.predictions = [pd.read_csv(path, index_col=0) for path in paths]
        return ensembler

    def launch(self, n_folds=5):
        for run in self.runs:
            print("~" * 50)
            print(f"Running {run.project}/{run.run_name}")
            print("~" * 50)

            cv = CrossValidation(n_folds, **self.project_configs[run.project])
            pipeline = Pipeline(run.config, run.run_name)

            val_accuracies, test_accuracies = [], []
            final_predictions = []
            for val_acc, test_acc, predictions in pipeline.run(cv):
                val_accuracies.append(val_acc)
                test_accuracies.append(test_acc)

                final_predictions.append(predictions)

                print(
                    f"Fold {len(val_accuracies)}/{n_folds}: Validation accuracy {val_acc:.4f}, Test accuracy {test_acc:.4f}"
                )

            print(f"Wandb validation accuracy: {run.val_acc:.4f}, test accuracy: {run.test_acc:.4f}")
            print(
                f"Average validation accuracy: {sum(val_accuracies) / len(val_accuracies):.4f}"
            )
            print(
                f"Average test accuracy: {sum(test_accuracies) / len(test_accuracies):.4f}"
            )

            ensemble_final_predictions = Ensembler.aggregate(final_predictions)
            self.predictions.append(ensemble_final_predictions)

            ensemble_final_predictions.to_csv(f"data/runs/{run.run_name}.csv")


    def aggregate(predictions):
        ensemble_predictions = pd.DataFrame(
            predictions[0],
            columns=["HOME_WINS", "DRAW", "AWAY_WINS"],
            index=predictions[0].index,
        )
        for i in range(1, len(predictions)):
            ensemble_predictions += predictions[i]
        ensemble_predictions /= len(predictions)

        def vote(row):
            new_row = row.copy()

            if row["HOME_WINS"] > 0.5:
                new_row["HOME_WINS"] = 1
                new_row["DRAW"] = 0
                new_row["AWAY_WINS"] = 0
            elif row["AWAY_WINS"] > 0.5:
                new_row["HOME_WINS"] = 0
                new_row["DRAW"] = 0
                new_row["AWAY_WINS"] = 1
            else:
                new_row["HOME_WINS"] = 0
                new_row["DRAW"] = 1
                new_row["AWAY_WINS"] = 0

            return new_row

        ensemble_predictions = ensemble_predictions.apply(vote, axis=1)
        ensemble_predictions = ensemble_predictions.astype(int)

        return ensemble_predictions


if __name__ == "__main__":

    project_configs = {
        # "thomas_l/QRT-Challenge-reg_lin": {"data_augment": False, "add_player": False},
        "thomas_l/QRT-Challenge-features_aug": {"data_augment": True, "add_player": True},
        "thomas_l/QRT-Challenge-team_agg": {"data_augment": False, "add_player": True},
        "thomas_l/QRT-Challenge-Baseline": {"data_augment": False, "add_player": False},
    }

    if not os.path.exists("./data/runs/"):
        os.makedirs("./data/runs/")
    runs = os.listdir("./data/runs/")
    paths = [f"data/runs/{run}" for run in runs]
    print(f"Previously saved runs: {paths}")

    if len(paths) == 0:
        retriver = Retriver(project_configs.keys())
        retriver.fetch("val_acc", "test_acc")

        best_runs = retriver.get("test_acc", top=15)
        best_runs = [Run(**row) for _, row in best_runs.iterrows()]
        print([(run.project, run.run_name) for run in best_runs])

        ensembler = Ensembler(best_runs, project_configs)
        ensembler.launch()
        ensemble_prediction = Ensembler.aggregate(ensembler.predictions)
    else:
        ensembler = Ensembler.from_csv(paths)
        ensemble_prediction = Ensembler.aggregate(ensembler.predictions)

    ensemble_prediction.to_csv("ensemble_predictions.csv")
