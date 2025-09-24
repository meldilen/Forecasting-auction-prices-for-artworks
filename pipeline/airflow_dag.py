from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
import os

default_args = {
    'owner': 'art-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def run_eda():
    """Run EDA pipeline"""
    os.system('cd ../datasets && python EDA.py')

def run_preprocessing():
    """Run preprocessing pipeline"""
    os.system('cd ../datasets && python preprocessing.py')

def run_training():
    """Run model training"""
    os.system('cd ../models && python train_model.py')

def deploy_services():
    """Deploy API and app"""
    os.system('cd ../deployment && docker-compose up -d')

with DAG(
    'art_auction_pipeline',
    default_args=default_args,
    description='Art Auction Price Prediction Pipeline',
    schedule_interval=timedelta(minutes=5),  # Run every 5 minutes
    catchup=False,
    tags=['art', 'ml', 'pipeline']
) as dag:
    
    eda_task = PythonOperator(
        task_id='run_eda',
        python_callable=run_eda
    )
    
    preprocessing_task = PythonOperator(
        task_id='run_preprocessing',
        python_callable=run_preprocessing
    )
    
    training_task = PythonOperator(
        task_id='run_training',
        python_callable=run_training
    )
    
    deploy_task = PythonOperator(
        task_id='deploy_services',
        python_callable=deploy_services
    )
    
    # Define pipeline order
    eda_task >> preprocessing_task >> training_task >> deploy_task