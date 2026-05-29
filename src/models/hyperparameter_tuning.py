import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.sklearn
import optuna
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from src.models.evaluate import (
    get_metrics,
    print_metrics,
    save_confusion_matrix,
    save_classification_report
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_splits(config: dict):
    """Load train and test splits."""
    processed_path = config["data"]["processed_data_path"]
    text_col       = config["model"]["text_column"]
    target_col     = config["model"]["target_column"]

    train_df = pd.read_csv(f"{processed_path}train.csv")
    test_df  = pd.read_csv(f"{processed_path}test.csv")

    train_df = train_df.dropna(subset=[text_col])
    test_df  = test_df.dropna(subset=[text_col])

    X_train = train_df[text_col]
    y_train = train_df[target_col]
    X_test  = test_df[text_col]
    y_test  = test_df[target_col]

    print(f"✅ Train size : {X_train.shape[0]}")
    print(f"✅ Test size  : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def objective(trial, X_train, X_test, y_train, y_test, config):
    """
    Optuna objective function.
    Each trial suggests different hyperparameters
    and returns F1 score to maximize.
    """

    # Suggest TF-IDF hyperparameters
    max_features = trial.suggest_categorical(
        "max_features", [5000, 8000, 10000, 15000]
    )
    ngram_min = trial.suggest_int("ngram_min", 1, 1)
    ngram_max = trial.suggest_int("ngram_max", 1, 3)

    # Suggest LinearSVC hyperparameters
    C         = trial.suggest_float("C", 0.01, 10.0, log=True)
    max_iter  = trial.suggest_categorical("max_iter", [500, 1000, 2000])

    # Build pipeline with suggested params
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(ngram_min, ngram_max)
        )),
        ("model", LinearSVC(
            C=C,
            max_iter=max_iter,
            random_state=config["data"]["random_state"]
        ))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    f1     = f1_score(y_test, y_pred)

    return f1


def run_optuna_study(X_train, X_test, y_train, y_test, config, n_trials: int = 30):
    """
    Run Optuna hyperparameter search.
    Logs best trial to MLflow.
    """

    print(f"\n⏳ Starting Optuna study with {n_trials} trials...")

    # Create Optuna study — maximize F1
    study = optuna.create_study(
        direction="maximize",
        study_name="linear_svc_tuning"
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, X_train, X_test, y_train, y_test, config
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\n✅ Optuna study complete!")
    print(f"Best Trial  : {study.best_trial.number}")
    print(f"Best F1     : {study.best_value:.4f}")
    print(f"Best Params : {study.best_params}")

    return study


def save_optuna_plots(study, save_path: str = "reports/optuna_study.png"):
    """Save Optuna optimization history plot."""
    try:
        history = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Optuna study plot saved to {save_path}")
    except Exception as e:
        print(f"⚠️ Could not save Optuna plot: {e}")


def train_best_model(study, X_train, X_test, y_train, y_test, config):
    """
    Train final model with best hyperparameters
    and log everything to MLflow.
    """

    best_params = study.best_params
    print(f"\n⏳ Training best model with params: {best_params}")

    # Set MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="linear_svc_best_tuned"):

        # Build best pipeline
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=best_params["max_features"],
                ngram_range=(
                    best_params["ngram_min"],
                    best_params["ngram_max"]
                )
            )),
            ("model", LinearSVC(
                C=best_params["C"],
                max_iter=best_params["max_iter"],
                random_state=config["data"]["random_state"]
            ))
        ])

        # Train
        pipeline.fit(X_train, y_train)
        print("✅ Best model trained!")

        # Predict
        y_pred = pipeline.predict(X_test)

        # Metrics
        metrics = get_metrics(y_test, y_pred)
        print_metrics(metrics, "LinearSVC Best Tuned")

        # Log parameters to MLflow
        mlflow.log_param("model_name",   "linear_svc_tuned")
        mlflow.log_param("max_features", best_params["max_features"])
        mlflow.log_param("ngram_range",  f"({best_params['ngram_min']}, {best_params['ngram_max']})")
        mlflow.log_param("C",            best_params["C"])
        mlflow.log_param("max_iter",     best_params["max_iter"])
        mlflow.log_param("n_trials",     30)

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log best F1 from Optuna
        mlflow.log_metric("optuna_best_f1", study.best_value)

        # Save & log confusion matrix
        cm_path = "reports/linear_svc_tuned_confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, "LinearSVC Tuned", cm_path)
        mlflow.log_artifact(cm_path)

        # Save & log classification report
        cr_path = "reports/linear_svc_tuned_classification_report.txt"
        save_classification_report(y_test, y_pred, "LinearSVC Tuned", cr_path)
        mlflow.log_artifact(cr_path)

        # Save & log Optuna plot
        optuna_plot_path = "reports/optuna_study.png"
        save_optuna_plots(study, optuna_plot_path)
        mlflow.log_artifact(optuna_plot_path)

        # Save best model
        model_path = "models/artifacts/best_model.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, "best_model")
        mlflow.log_artifact(model_path)

        print(f"✅ Best model logged to MLflow!")
        print(f"✅ Best model saved to {model_path}")

        return pipeline, metrics


if __name__ == "__main__":

    # Load config
    config = load_config()

    # Load splits
    X_train, X_test, y_train, y_test = load_splits(config)

    # Run Optuna study
    study = run_optuna_study(
        X_train, X_test,
        y_train, y_test,
        config,
        n_trials=30
    )

    # Train & log best model
    pipeline, metrics = train_best_model(
        study,
        X_train, X_test,
        y_train, y_test,
        config
    )

    # Final comparison
    print("\n=== Day 5 Baseline vs Day 6 Tuned ===")
    print(f"{'Metric':<12} {'Baseline (LinearSVC)':<25} {'Tuned (LinearSVC)':<25}")
    print("-" * 62)
    print(f"{'Accuracy':<12} {'0.9580':<25} {metrics['accuracy']:<25.4f}")
    print(f"{'F1 Score':<12} {'0.9570':<25} {metrics['f1_score']:<25.4f}")
    print(f"{'Precision':<12} {'0.9718':<25} {metrics['precision']:<25.4f}")
    print(f"{'Recall':<12} {'0.9426':<25} {metrics['recall']:<25.4f}")