import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "model.pkl"
TARGET_COLUMN = "target"


def train(train_csv: str) -> str:
    """Обучение модели логистической регрессии на тренировочной выборке и сохранение модели."""
    df = pd.read_csv(train_csv)
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    model = LogisticRegression(max_iter=200)
    model.fit(features, target)
    joblib.dump(model, MODEL_PATH)
    return MODEL_PATH
