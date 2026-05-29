import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features.text_preprocessor import clean_text
from src.models.evaluate import (
    get_metrics,
    print_metrics,
    save_confusion_matrix,
    save_classification_report
)
import os


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

    X_train = train_df[text_col]
    y_train = train_df[target_col]
    X_test  = test_df[text_col]
    y_test  = test_df[target_col]

    print(f"✅ Train size : {X_train.shape[0]}")
    print(f"✅ Test size  : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def build_model_pipeline(model, config: dict) -> Pipeline:
    """Build full pipeline — TF-IDF + Model."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=config["features"]["max_features"],
            ngram_range=tuple(config["features"]["ngram_range"])
        )),
        ("model", model)
    ])


def get_models() -> dict:
    """Return dictionary of models to train."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "naive_bayes": MultinomialNB(),
        "linear_svc": LinearSVC(
            max_iter=1000,
            random_state=42
        )
    }


def train_and_log(model_name, model, X_train, X_test, y_train, y_test, config):
    """Train model and log everything to MLflow."""

    # Set MLflow tracking URI & experiment
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_name):

        print(f"\n⏳ Training {model_name}...")

        # Build pipeline
        pipeline = build_model_pipeline(model, config)

        # Handle NaN values
        X_train = X_train.fillna('')
        X_test = X_test.fillna('')

        # Train
        pipeline.fit(X_train, y_train)
        print(f"✅ {model_name} trained!")

        # Predict
        y_pred = pipeline.predict(X_test)

        # Predict probability (not available for LinearSVC)
        y_prob = None
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Get metrics
        metrics = get_metrics(y_test, y_pred, y_prob)
        print_metrics(metrics, model_name)

        # Log parameters to MLflow
        mlflow.log_param("model_name",   model_name)
        mlflow.log_param("max_features", config["features"]["max_features"])
        mlflow.log_param("ngram_range",  str(config["features"]["ngram_range"]))
        mlflow.log_param("test_size",    config["data"]["test_size"])
        mlflow.log_param("random_state", config["data"]["random_state"])

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Save & log confusion matrix
        cm_path = f"reports/{model_name}_confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, model_name, cm_path)
        mlflow.log_artifact(cm_path)

        # Save & log classification report
        cr_path = f"reports/{model_name}_classification_report.txt"
        save_classification_report(y_test, y_pred, model_name, cr_path)
        mlflow.log_artifact(cr_path)

        # Save & log model
        model_path = f"models/artifacts/{model_name}.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, model_name)
        mlflow.log_artifact(model_path)

        print(f"✅ {model_name} logged to MLflow!")

        return metrics


if __name__ == "__main__":

    # Load config
    config = load_config()

    # Load splits
    X_train, X_test, y_train, y_test = load_splits(config)

    # Get all models
    models = get_models()

    # Store results for comparison
    all_results = {}

    # Train and log each model
    for model_name, model in models.items():
        metrics = train_and_log(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            config
        )
        all_results[model_name] = metrics

    # Print final comparison
    print("\n=== Final Model Comparison ===")
    print(f"{'Model':<25} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 73)
    for model_name, metrics in all_results.items():
        print(
            f"{model_name:<25} "
            f"{metrics['accuracy']:<12.4f} "
            f"{metrics['f1_score']:<12.4f} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f}"
        )
    