import wandb

from functools import partial

from src.crossval import CrossValidation
from src.models import XGBoost, Pipeline


def run(cv: CrossValidation):
    wandb.init(project="xgboost", entity="qrt-challenge")

    # Config to dict
    config = dict(wandb.config)
    pipeline = Pipeline(XGBoost, config, "xgboost_test")

    val_accuracies, test_accuracies = [], []
    final_predictions = []
    for fold, (val_acc, test_acc, predictions) in enumerate(pipeline.run(cv)):
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)

        final_predictions.append(predictions)

        print(
            f"Fold {fold+1}/{n_folds}: Validation accuracy {val_acc:.4f}, Test accuracy {test_acc:.4f}"
        )

    avg_val_acc = sum(val_accuracies) / len(val_accuracies)
    avg_test_acc = sum(test_accuracies) / len(test_accuracies)
    print(f"Average validation accuracy: {avg_val_acc:.4f}")
    print(f"Average test accuracy: {avg_test_acc:.4f}")

    wandb.log({"val_acc": avg_val_acc, "test_acc": avg_test_acc})


if __name__ == "__main__":

    n_folds = 5
    cv = CrossValidation(n_folds, data_augment=True, add_player=True, rank=None)
    run_partial = partial(run, cv)

    wandb.agent("qrt-challenge/xgboost/7to9ah0h", run_partial)
