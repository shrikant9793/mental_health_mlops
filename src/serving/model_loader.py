import mlflow
import mlflow.sklearn
import yaml
import os
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from yaml file."""
    load_dotenv()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Dynamic tracking URI
    mlruns_path = os.path.abspath("mlruns")
    config["mlflow"]["tracking_uri"] = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"file:///{mlruns_path}"
    )
    return config


def load_model_from_registry(
    model_name: str,
    alias: str,
    config: dict
):
    """
    Load model from MLflow Registry by alias.
    Used at API startup.
    """
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    model_uri = f"models:/{model_name}@{alias}"

    print(f"⏳ Loading model from registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Model loaded successfully!")

    return model


def get_model_info(
    model_name: str,
    alias: str,
    config: dict
) -> dict:
    """
    Get model metadata from MLflow Registry.
    Returns version, tags, description.
    """
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    # Get model version by alias
    model_version = client.get_model_version_by_alias(
        name=model_name,
        alias=alias
    )

    return {
        "model_name"      : model_name,
        "model_version"   : model_version.version,
        "alias"           : alias,
        "description"     : model_version.description,
        "tags"            : model_version.tags
    }