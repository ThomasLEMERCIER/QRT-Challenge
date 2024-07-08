from __future__ import annotations

from src.evaluate import evaluate_xgb_model, evaluate_lgb_model, evaluate_cat_model
from src.postprocessing import compute_prediction
from src.crossval import CrossValidation
from src.deep_learning import MLPClassifier, TabularDataset, train_model, test_epoch


import torch
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import Dataset, train
from catboost import CatBoost, Pool


class Pipeline:
    def __init__(self, model_type: Model, model_params):
        self.model = model_type(model_params)
        self.model_params = model_params

        self.iterator_params = self.model_params.get("knee_point", None)

    def run(self, crossval: CrossValidation):
        for fold, ((x_train, y_train), (x_val, y_val), (x_test, y_test), x_pred) in enumerate(crossval.iterate(self.iterator_params)):
            acc_val, acc_test, predictions = self.model.run(x_train, y_train, x_val, y_val, x_test, y_test, x_pred)
            yield acc_val, acc_test, predictions

class Model:
    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
        raise NotImplementedError

class XGBoost(Model):
    def __init__(self, args):
        super(XGBoost, self).__init__()
        args["booster"] = "gbtree"
        args["device"] = "cuda"

        args["objective"] = "multi:softmax"
        args["num_class"] = 3
        args["eval_metric"] = "merror"
        args["tree_method"] = "hist"
        args["verbosity"] = 0

        self.num_boost_round = args["num_boost_round"]
        self.early_stopping_rounds = args["early_stopping_rounds"]
        args.pop("num_boost_round")
        args.pop("early_stopping_rounds")

        self.args = args

    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        dtest = xgb.DMatrix(x_test, label=y_test)

        evals = [(dtrain, "train"), (dval, "val")]
        self.model = xgb.train(
            params=self.args, 
            dtrain=dtrain, 
            num_boost_round=self.num_boost_round, 
            evals=evals, 
            early_stopping_rounds=self.early_stopping_rounds, 
            verbose_eval=False
        )

        acc_val, _ = evaluate_xgb_model(self.model, dval, y_val)
        acc_test, _ = evaluate_xgb_model(self.model, dtest, y_test)

        dpred = xgb.DMatrix(x_pred)
        y_pred = self.model.predict(dpred, iteration_range=(0, self.model.best_iteration))
        predictions = compute_prediction(y_pred, x_pred)

        return acc_val, acc_test, predictions

class LinearRegression(Model):
    def __init__(self, args):
        super(LinearRegression, self).__init__()
        self.args = args

    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
        l1_ratio = self.args["l1_ratio"]
        C = self.args["C"]
        multi_class = self.args["multi_class"]

        y_train, y_val, y_test = y_train.values.ravel(), y_val.values.ravel(), y_test.values.ravel()

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=l1_ratio,
            C=C,
            max_iter=5000,
            multi_class=multi_class,
            random_state=42,
        )
        model.fit(x_train, y_train)

        acc_val = model.score(x_val, y_val)
        acc_test = model.score(x_test, y_test)

        y_pred = model.predict(x_pred)
        predictions = compute_prediction(y_pred, x_pred)

        return acc_val, acc_test, predictions

class SVM(Model):
    def __init__(self, args):
        super(SVM, self).__init__()
        self.args = args

    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
        C = self.args["C"]
        kernel = self.args["kernel"]
        degree = self.args["degree"]
        gamma = self.args["gamma"]
        coef0 = self.args["coef0"]
        class_weight = self.args["class_weight"]

        y_train, y_val, y_test = y_train.values.ravel(), y_val.values.ravel(), y_test.values.ravel()

        model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            class_weight=class_weight,
            probability=False,
            random_state=42,
        )
        model.fit(x_train, y_train)

        acc_val = model.score(x_val, y_val)
        acc_test = model.score(x_test, y_test)

        y_pred = model.predict(x_pred)
        predictions = compute_prediction(y_pred, x_pred)

        return acc_val, acc_test, predictions

class MLP(Model):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
        hidden_dim = self.args["hidden_dim"]
        weight_decay = self.args["weight_decay"]
        learning_rate = self.args["lr"]
        n_epochs = self.args["n_epochs"]
        label_smoothing = self.args["label_smoothing"]
        dropout_rate = self.args["dropout_rate"]
        batch_size = self.args["batch_size"]


        y_train, y_val, y_test = y_train.values.ravel(), y_val.values.ravel(), y_test.values.ravel()

        model = MLPClassifier(
            input_dim=x_train.shape[1],
            hidden_dim=hidden_dim,
            output_dim=3,
            dropout_rate=dropout_rate,
        )

        train_dl = torch.utils.data.DataLoader(TabularDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        val_dl = torch.utils.data.DataLoader(TabularDataset(x_val, y_val), batch_size=512, shuffle=False, drop_last=False)
        test_dl = torch.utils.data.DataLoader(TabularDataset(x_test, y_test), batch_size=512, shuffle=False, drop_last=False)

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)

        best_model = train_model(model, optimizer, criterion, scheduler, train_dl, val_dl, n_epochs)
        model.load_state_dict(best_model)

        _, acc_val = test_epoch(model, criterion, val_dl)
        _, acc_test = test_epoch(model, criterion, test_dl)

        x_pred_tensor = torch.tensor(x_pred.to_numpy()).float()
        y_pred = torch.argmax(model(x_pred_tensor), dim=1).detach().numpy()
        predictions = compute_prediction(y_pred, x_pred)

        return acc_val, acc_test, predictions

class LGBM(Model):
    def __init__(self, args):
        super(LGBM, self).__init__()
        args["objective"] = "multiclass"
        args["num_class"] = 3
        args["boosting"] = "gbdt"
        args["force_col_wise"] = True
        args["seed"] = 42
        args["bagging_seed"] = 42
        args["feature_fraction_seed"] = 42
        args["extra_seed"] = 42
        args["verbosity"] = -1

        self.num_boost_round = args.pop("num_boost_round")
        self.args = args

    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
        dtrain = Dataset(x_train, y_train)
        dval = Dataset(x_val, y_val, reference=dtrain)

        bst = train(self.args, dtrain, num_boost_round=self.num_boost_round, valid_sets=[dval])

        acc_val, _ = evaluate_lgb_model(bst, x_val, y_val)
        acc_test, _ = evaluate_lgb_model(bst, x_test, y_test)

        y_pred = np.argmax(bst.predict(x_pred, num_iteration=bst.best_iteration), axis=1)
        predictions = compute_prediction(y_pred, x_pred)

        return acc_val, acc_test, predictions

class CBoost(Model):
    def __init__(self, args):
        super(CBoost, self).__init__()

        args["loss_function"] = "MultiClass"
        args["eval_metric"] = "Accuracy"
        args["random_seed"] = 42
        args["task_type"] = "GPU"
        args["devices"] = "0"
        args["allow_writing_files"] = False

        self.args = args

    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
        train_pool = Pool(x_train, y_train)
        val_pool = Pool(x_val, y_val)

        model = CatBoost(self.args)
        model.fit(train_pool, eval_set=val_pool, verbose=False)

        acc_val, _ = evaluate_cat_model(model, x_val, y_val)
        acc_test, _ = evaluate_cat_model(model, x_test, y_test)

        y_pred = model.predict(x_pred, prediction_type="Class", ntree_end=model.get_best_iteration())
        predictions = compute_prediction(y_pred, x_pred)

        return acc_val, acc_test, predictions
