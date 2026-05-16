from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

DATASET_DIR = Path("dataset")
IRIS_CSV = DATASET_DIR / "iris.csv"
IRIS_TRAIN_CSV = DATASET_DIR / "iris_train.csv"
IRIS_TEST_CSV = DATASET_DIR / "iris_test.csv"

TEST_SIZE_FRACTION = 0.2
TRAIN_TEST_SPLIT_RANDOM_STATE = 42


def load_data() -> str:
    """Загрузка датасета Iris."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(IRIS_CSV, index=False)
    return str(IRIS_CSV)


def prepare_data(csv_path: str) -> list[str]:
    """Чтение загруженного датасета и разделение на train и test выборки."""
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE_FRACTION,
        random_state=TRAIN_TEST_SPLIT_RANDOM_STATE,
    )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(IRIS_TRAIN_CSV, index=False)
    test_df.to_csv(IRIS_TEST_CSV, index=False)
    return [str(IRIS_TRAIN_CSV), str(IRIS_TEST_CSV)]
