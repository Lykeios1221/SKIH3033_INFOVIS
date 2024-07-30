import optuna
import lightgbm as lgb
from optuna_integration import LightGBMPruningCallback
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

# Function to load validation data
def load_data(file_path):
    data = np.load(file_path)
    return (
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"],
        data["X_val_early_stop"],
        data["X_val_tune"],
        data["y_val_early_stop"],
        data["y_val_tune"],
    )


# Define the objective function
def objective(trial):
    # Define the hyperparameters to tune
    params = {
        "objective": "binary",
        "metric": "auc",
        "device_type": "cpu",
        "verbose": -1,
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["gbdt", "dart", "rf"]
        ),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "num_leaves": trial.suggest_int("num_leaves", 8, 32768),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    }

    # Load validation data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_val_early_stop,
        X_val_tune,
        y_val_early_stop,
        y_val_tune,
    ) = load_data("data/asg3/complete_data.npz")
    # Create the LightGBM model
    model = lgb.LGBMClassifier(**params)

    # Fit the model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val_early_stop, y_val_early_stop)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            LightGBMPruningCallback(trial, metric="auc"),
        ],
    )

    y_pred = model.predict(X_val_tune)
    auc = roc_auc_score(y_val_tune, y_pred)

    return auc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Optuna optimization.")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=2500,
        help="Number of trials for Optuna optimization"
    )
    args = parser.parse_args()
    n_trials = args.n_trials

    storage_url = "mysql+mysqldb://root@127.0.0.1/optuna"

    study = optuna.load_study(
        study_name="LightGBM_study_mysql",
        storage=storage_url,
        pruner=optuna.pruners.HyperbandPruner(),
    )

    # Define the function to ensure the number of trials
    def ensure_n_trials(study, trial):
        if trial.number >= n_trials:
            study.stop()

    # Optimize the study
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[ensure_n_trials],
        show_progress_bar=True,
    )

    print("Optimization completed!")
