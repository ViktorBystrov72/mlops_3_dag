import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

METRICS_PATH = "model_metrics.json"
TARGET_COLUMN = "target"


def test(model_path: str, test_csv: str) -> str:
    """Тестирование модели на тестовой выборке и сохранение результатов."""
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv)
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    predictions = model.predict(features)
    accuracy = accuracy_score(target, predictions)
    report = classification_report(target, predictions)

    metrics = {"accuracy": accuracy, "report": report}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return METRICS_PATH
