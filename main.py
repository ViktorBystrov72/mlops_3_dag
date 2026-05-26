from data import load_data, prepare_data
from test import test
from train import train


def run_pipeline() -> None:
    csv_path = load_data()
    train_csv, test_csv = prepare_data(csv_path)
    model_path = train(train_csv)
    metrics_path = test(model_path, test_csv)
    print("Готово:")
    print(f"  датасет:     {csv_path}")
    print(f"  train:       {train_csv}")
    print(f"  test:        {test_csv}")
    print(f"  модель:      {model_path}")
    print(f"  метрики:     {metrics_path}")


if __name__ == "__main__":
    run_pipeline()
