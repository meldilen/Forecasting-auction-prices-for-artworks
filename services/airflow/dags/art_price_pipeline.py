from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'art_pipeline',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    'art_auction_pipeline',
    default_args=default_args,
    description='Art Auction Price Prediction Pipeline - Data Engineering, Model Engineering, Deployment',
    schedule_interval=timedelta(minutes=5),
    catchup=False,
    tags=['art', 'ml', 'pipeline']
)

data_engineering = BashOperator(
    task_id='data_engineering',
    bash_command='cd /opt/airflow/code/datasets && python preprocessing.py',
    dag=dag,
)

model_engineering = BashOperator(
    task_id='model_engineering',
    bash_command='cd /opt/airflow/code/modelsEngs && python train_model.py',
    dag=dag,
)

deployment = BashOperator(
    task_id='deployment',
    bash_command='cd /opt/airflow && docker-compose up -d --build api streamlit',
    dag=dag,
)

data_engineering >> model_engineering >> deployment