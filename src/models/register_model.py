import mlflow
import os
import mlflow.sklearn
import yaml
import joblib
from pathlib import Path
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

def load_config(config_path: str = "configs/config.yaml") -> dict:
    load_dotenv()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    mlruns_path = os.path.abspath("mlruns")
    config["mlflow"]["tracking_uri"] = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"file:///{mlruns_path}"
    )
    return config


def get_best_run(config: dict) -> tuple:
    """
    Fetch best run from MLflow experiment
    based on highest F1 score.
    """
    client = MlflowClient(
        tracking_uri=config["mlflow"]["tracking_uri"]
    )

    # Get experiment by name
    experiment = client.get_experiment_by_name(
        config["mlflow"]["experiment_name"]
    )

    if experiment is None:
        raise ValueError(
            f"Experiment '{config['mlflow']['experiment_name']}' not found!"
        )

    # Search all runs — order by F1 score descending
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found in experiment!")

    best_run = runs[0]
    print(f"\n✅ Best Run Found!")
    print(f"Run ID    : {best_run.info.run_id}")
    print(f"Run Name  : {best_run.info.run_name}")
    print(f"F1 Score  : {best_run.data.metrics.get('f1_score', 'N/A'):.4f}")
    print(f"Accuracy  : {best_run.data.metrics.get('accuracy', 'N/A'):.4f}")
    print(f"Precision : {best_run.data.metrics.get('precision', 'N/A'):.4f}")
    print(f"Recall    : {best_run.data.metrics.get('recall', 'N/A'):.4f}")

    return client, best_run


def register_model(client, best_run, model_name: str = "MentalHealthClassifier") -> str:
    """
    Register best run model to MLflow Model Registry.
    Returns registered model version.
    """

    run_id    = best_run.info.run_id
    run_name  = best_run.info.run_name

    # Model URI — points to logged model artifact
    model_uri = f"runs:/{run_id}/{run_name}"

    print(f"\n⏳ Registering model to MLflow Registry...")
    print(f"Model URI : {model_uri}")

    # Register model
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print(f"✅ Model registered!")
    print(f"Name    : {registered_model.name}")
    print(f"Version : {registered_model.version}")

    return registered_model.version


def transition_model_stage(
    client,
    model_name: str,
    version: str,
    stage: str
):
    """
    Set model alias — replaces deprecated stage transitions.
    Aliases: Staging, Production, Archived
    """
    client.set_registered_model_alias(
        name=model_name,
        alias=stage,
        version=version
    )
    print(f"✅ Model '{model_name}' v{version} alias set to '{stage}'")


def add_model_description(
    client,
    model_name: str,
    version: str
):
    """Add description and tags to registered model."""

    # Add version description
    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            "LinearSVC model tuned with Optuna. "
            "F1=96.70%, Accuracy=96.77%. "
            "Trained on mental_health dataset."
        )
    )

    # Add tags
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="tuning",
        value="optuna"
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="dataset",
        value="mental_health"
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="algorithm",
        value="LinearSVC"
    )

    print(f"✅ Model description and tags added!")


# def load_model_from_registry(
#     model_name: str,
#     stage: str,
#     config: dict
# ):
#     """
#     Load model directly from MLflow Registry
#     by stage — Production, Staging etc.
#     """
#     mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
#     # run_id = "34f881b1ecb9403aa2e2daaa465df6ba" 
#     # experiment_id = "546774975895166091"
#     # model_folder_name = "linear_svc_best_tuned"    

#     model_uri = f"models:/{model_name}/{stage}"
#     # model_uri = os.path.abspath(f"mlruns/{experiment_id}/{run_id}/artifacts/{model_folder_name}")
#     print(f"\n⏳ Loading model from registry: {model_uri}")

#     model = mlflow.sklearn.load_model(model_uri)
#     print(f"✅ Model loaded from registry!")

#     return model

# def load_model_from_registry(model_name, stage, config=None):
#     client = mlflow.tracking.MlflowClient()
    
#     print(f"⏳ Dynamically locating latest version for stage '{stage}'...")
#     # Automatically get the latest version details from the model registry
#     latest_versions = client.get_latest_versions(model_name, stages=[stage])
    
#     if not latest_versions:
#         raise RuntimeError(f"No model versions found for '{model_name}' in stage '{stage}'")
        
#     latest_version = latest_versions[0]
#     run_id = latest_version.run_id
    
#     # Safely get the absolute path to your local mlruns directory
#     base_tracking_dir = os.path.abspath("mlruns")
    
#     print(f"🔎 Scanning MLflow storage for Run ID: {run_id}")
    
#     # Walk through mlruns dynamically to locate where this Run ID is stored
#     target_artifact_dir = None
#     for root, dirs, files in os.walk(base_tracking_dir):
#         if run_id in root and "artifacts" in root:
#             target_artifact_dir = root
#             break
            
#     if not target_artifact_dir or not os.listdir(target_artifact_dir):
#         raise FileNotFoundError(
#             f"❌ Critical Error: The artifact directory for Run ID '{run_id}' "
#             f"is missing or completely empty. Your training script did not successfully save the model."
#         )

#     # Dynamically pick up the first logged model folder name inside artifacts
#     subdirs = [d for d in os.listdir(target_artifact_dir) if os.path.isdir(os.path.join(target_artifact_dir, d))]
#     if not subdirs:
#         raise FileNotFoundError(f"❌ No logged model folder found inside: {target_artifact_dir}")
        
#     final_model_path = os.path.join(target_artifact_dir, subdirs[0])
    
#     # Format cleanly to a Windows-compatible file URI scheme
#     model_uri = Path(final_model_path).as_uri()
    
#     print(f"✅ Loading model dynamically from local path: {model_uri}")
#     model = mlflow.sklearn.load_model(model_uri)
#     return model


def load_model_from_registry(
    model_name: str,
    alias: str,
    config: dict
):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    model_uri = f"models:/{model_name}@{alias}"
    print(f"\n⏳ Loading model from registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Model loaded from registry!")
    return model

def compare_all_runs(config: dict):
    """Print comparison table of all MLflow runs."""
    client = MlflowClient(
        tracking_uri=config["mlflow"]["tracking_uri"]
    )

    experiment = client.get_experiment_by_name(
        config["mlflow"]["experiment_name"]
    )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"]
    )

    print("\n=== All MLflow Runs Comparison ===")
    print(f"{'Run Name':<30} {'F1':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<10}")
    print("-" * 74)

    for run in runs:
        metrics = run.data.metrics
        print(
            f"{run.info.run_name:<30} "
            f"{metrics.get('f1_score', 0):<10.4f} "
            f"{metrics.get('accuracy', 0):<12.4f} "
            f"{metrics.get('precision', 0):<12.4f} "
            f"{metrics.get('recall', 0):<10.4f}"
        )


if __name__ == "__main__":

    MODEL_NAME = "MentalHealthClassifier"

    # Load config
    config = load_config()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    # Step 1 — Compare all runs
    compare_all_runs(config)

    # Step 2 — Get best run
    client, best_run = get_best_run(config)

    # Step 3 — Register best model
    version = register_model(client, best_run, MODEL_NAME)

    # Step 4 — Add description & tags
    add_model_description(client, MODEL_NAME, version)

    # Step 5 — Transition to Staging
    transition_model_stage(client, MODEL_NAME, version, "Staging")

    # Step 6 — Transition to Production
    transition_model_stage(client, MODEL_NAME, version, "Production")

    # Step 7 — Load model from Production registry
    model = load_model_from_registry(MODEL_NAME, "Production", config)

    # Step 8 — Quick sanity check
    print("\n⏳ Running sanity check on loaded model...")
    sample_texts = [
        "I feel so hopeless and empty inside",
        "Today was a great day I feel amazing"
    ]
    predictions = model.predict(sample_texts)
    print("\n=== Sanity Check Results ===")
    for text, pred in zip(sample_texts, predictions):
        label = "Depression" if pred == 1 else "Non-Depression"
        print(f"Text  : {text[:50]}")
        print(f"Label : {label}")
        print()

    print("✅ Day 7 Complete — Model registered and in Production!")