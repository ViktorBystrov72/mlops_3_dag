# MLOps: пайплайн классификации Iris

Учебный проект: ML-пайплайн на **scikit-learn**, оркестрируемый **Apache Airflow 3**.

## Запуск

```bash
cd mlops
docker compose up -d --build 
```

### UI

- http://localhost:8080
- Логин / пароль: **`admin` / `admin`**

- DAG **`iris_ml_pipeline`** → **Trigger DAG**

После прогона в **корне проекта** `mlops/` появятся:

- `dataset/iris.csv`, `dataset/iris_train.csv`, `dataset/iris_test.csv`
- `model.pkl`
- `model_metrics.json`

### Команды

```bash
docker compose up -d --build   # в фоне
docker compose down            # остановить
docker compose down -v         # остановить и удалить том с БД Airflow
docker compose logs -f         # логи
```

