# Resume notes

## Draft bullets

- Built a PySpark ETL over 38419594 NYC taxi trips (12 months) with data-quality checks, matching DuckDB features on 1.0000 of overlapping zone-hour rows.
- Defined the extract, validate, transform, train, and evaluate steps as an Airflow DAG (5 tasks, statically validated) and ran the same tasks end to end with a local runner.
- Retrained the demand model on 11 months of history at 0.1451 WAPE on held-out April 2024, versus 0.1930 for seasonal-naive.
- Re-ran the parent DuckDB and LightGBM pipeline at 0.1461 April top-zone WAPE on the same held-out month.

## Skills and tools

Python, SQL, PySpark, DuckDB, Airflow DAG definitions, LightGBM, MLflow, pandas, NumPy, pytest. Local-mode Spark ETL, feature parity checks, time-series forecasting, and static DAG validation with a local runner.

## Project summary

This project extends a zone-hour taxi demand model with a local PySpark cleaning and aggregation job over 12 months of TLC yellow trips. An Airflow DAG defines extract, validate, transform, train, and evaluate; a local runner executes those tasks, and the retrained model is scored on held-out April 2024.

## GitHub description

PySpark ETL for a year of NYC taxi trips, checked against DuckDB, with an Airflow DAG and an 11-month demand model.
