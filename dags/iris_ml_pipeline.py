import os
import sys
from pathlib import Path

import pendulum
from airflow.sdk import dag, task

PROJECT_ROOT = Path(os.environ.get("MLOPS_ROOT", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _chdir_to_project() -> None:
    os.chdir(PROJECT_ROOT)


@dag(
    dag_id="iris_ml_pipeline",
    description="Загрузка Iris, подготовка данных, обучение и тестирование модели",
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "iris"],
)
def iris_ml_pipeline():
    @task
    def load_data() -> str:
        _chdir_to_project()
        from data import load_data as load_iris_dataset

        return load_iris_dataset()

    @task(multiple_outputs=True)
    def prepare_data(csv_path: str) -> dict[str, str]:
        _chdir_to_project()
        from data import prepare_data as split_dataset

        train_csv, test_csv = split_dataset(csv_path)
        return {"train_csv": train_csv, "test_csv": test_csv}

    @task
    def train_model(train_csv: str) -> str:
        _chdir_to_project()
        from train import train

        return train(train_csv)

    @task
    def test_model(model_path: str, test_csv: str) -> str:
        _chdir_to_project()
        from test import test as evaluate_model

        return evaluate_model(model_path, test_csv)

    csv_path = load_data()
    splits = prepare_data(csv_path)
    model_path = train_model(splits["train_csv"])
    test_model(model_path, splits["test_csv"])


iris_ml_pipeline()
