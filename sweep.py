import wandb
import argparse

from src import CrossValParams, ModelTypes
from src.crossval import CrossValidation
from src.models import Pipeline

class Sweep:
    def __init__(self, entity, model_name, sweep_id):
        self.entity = entity
        self.model_name = model_name
        self.sweep_id = sweep_id

        print(f"Running sweep {sweep_id} for model {model_name}")

        self.cv_params = CrossValParams[model_name]
        self.cv = CrossValidation(self.cv_params)
        self.model_type = ModelTypes[model_name]
    
    def run(self):
        wandb.init(entity=self.entity, project=self.model_name)
        config = dict(wandb.config)
        pipeline = Pipeline(self.model_type, config)

        val_accuracies, test_accuracies = [], []
        for fold, (val_acc, test_acc, _) in enumerate(pipeline.run(self.cv)):
            val_accuracies.append(val_acc)
            test_accuracies.append(test_acc)

            print(f"Fold {fold+1}/{self.cv_params.n_folds}: Validation accuracy {val_acc:.4f}, Test accuracy {test_acc:.4f}")

        avg_val_acc = sum(val_accuracies) / len(val_accuracies)
        avg_test_acc = sum(test_accuracies) / len(test_accuracies)
        print(f"Average validation accuracy: {avg_val_acc:.4f}")
        print(f"Average test accuracy: {avg_test_acc:.4f}")

        wandb.log({"val_acc": avg_val_acc, "test_acc": avg_test_acc})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_path", type=str)

    args = parser.parse_args()
    sweep_path = args.sweep_path

    entity, model_name, sweep_id = sweep_path.split("/")

    sweep = Sweep(entity, model_name, sweep_id)
    wandb.agent(sweep_path, sweep.run)
