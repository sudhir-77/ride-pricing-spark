"""Airflow DAG for the NYC taxi demand pipeline.

This file is the DAG definition. It is validated by parsing the source. On this
Windows machine the task callables run through the local runner, not an Airflow
scheduler, and Spark stays in local mode.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from ride_pricing.tasks import evaluate, extract, train, transform, validate

with DAG(
    dag_id="nyc_taxi_demand",
    schedule="0 6 * * *",
    start_date=datetime(2024, 4, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["taxi", "demand"],
) as dag:
    extract_task = PythonOperator(task_id="extract", python_callable=extract)
    validate_task = PythonOperator(task_id="validate", python_callable=validate)
    transform_task = PythonOperator(task_id="transform", python_callable=transform)
    train_task = PythonOperator(task_id="train", python_callable=train)
    evaluate_task = PythonOperator(task_id="evaluate", python_callable=evaluate)

    extract_task >> validate_task >> transform_task >> train_task >> evaluate_task
