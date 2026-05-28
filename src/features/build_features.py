import pandas as pd
import yaml
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.features.text_preprocessor import clean_text


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    """Load raw dataset."""
    df = pd.read_csv(config["data"]["raw_data_path"])
    print(f"✅ Data loaded: {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply text cleaning to dataset."""
    text_col = config["model"]["text_column"]
    print("⏳ Cleaning text...")
    df[text_col] = df[text_col].apply(clean_text)
    print("✅ Text cleaning done!")
    return df


def split_data(df: pd.DataFrame, config: dict):
    """Split data into train and test sets."""
    text_col = config["model"]["text_column"]
    target_col = config["model"]["target_column"]

    X = df[text_col]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    print(f"✅ Train size : {X_train.shape[0]}")
    print(f"✅ Test size  : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def build_pipeline(config: dict) -> Pipeline:
    """Build sklearn TF-IDF pipeline."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=config["features"]["max_features"],
            ngram_range=tuple(config["features"]["ngram_range"])
        ))
    ])
    return pipeline


def save_splits(X_train, X_test, y_train, y_test, config: dict):
    """Save train and test splits to processed folder."""
    processed_path = config["data"]["processed_data_path"]

    train_df = pd.DataFrame({
        config["model"]["text_column"]: X_train,
        config["model"]["target_column"]: y_train
    })

    test_df = pd.DataFrame({
        config["model"]["text_column"]: X_test,
        config["model"]["target_column"]: y_test
    })

    train_df.to_csv(f"{processed_path}train.csv", index=False)
    test_df.to_csv(f"{processed_path}test.csv", index=False)
    print(f"✅ Train split saved to {processed_path}train.csv")
    print(f"✅ Test split saved to {processed_path}test.csv")


def save_pipeline(pipeline: Pipeline, path: str = "models/artifacts/preprocessor.pkl"):
    """Save preprocessing pipeline."""
    joblib.dump(pipeline, path)
    print(f"✅ Preprocessing pipeline saved to {path}")


if __name__ == "__main__":
    # Load config
    config = load_config()

    # Load data
    df = load_data(config)

    # Preprocess
    df = preprocess_data(df, config)

    # Split
    X_train, X_test, y_train, y_test = split_data(df, config)

    # Build pipeline
    pipeline = build_pipeline(config)

    # Fit pipeline on train data
    pipeline.fit(X_train)
    print("✅ Pipeline fitted on training data!")

    # Save splits
    save_splits(X_train, X_test, y_train, y_test, config)

    # Save pipeline
    save_pipeline(pipeline)