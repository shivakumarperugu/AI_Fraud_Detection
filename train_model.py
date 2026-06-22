from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "spam.csv"
MODEL_PATH = ROOT / "fraud_model.pkl"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("Dataset must contain 'label' and 'text' columns.")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_and_save_model(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> None:
    df = load_data(data_path)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].map({"legit": 0, "fraud": 1})
    if df["label"].isna().any():
        raise ValueError("Dataset contains unknown labels. Use 'legit' or 'fraud'.")

    print(f"Training on {len(df)} examples from {data_path}")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Classification report:\n")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"], zero_division=0))

    joblib.dump(pipeline, model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    train_and_save_model()
