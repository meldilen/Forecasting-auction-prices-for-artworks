from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, '/opt/airflow/project')

default_args = {
    'owner': 'art_pipeline',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'art_auction_pipeline',
    default_args=default_args,
    description='Automated Art Auction Price Prediction Pipeline',
    schedule_interval=timedelta(minutes=5),
    catchup=False,
    tags=['art', 'ml', 'pipeline']
)

def run_data_engineering():
    """Stage 1: Data Engineering"""
    import subprocess
    try:
        result1 = subprocess.run([
            'python', '/opt/airflow/project/code/datasets/EDA.py'
        ], capture_output=True, text=True, cwd='/opt/airflow/project')
        
        if result1.returncode != 0:
            raise Exception(f"EDA failed: {result1.stderr}")
        
        result2 = subprocess.run([
            'python', '/opt/airflow/project/code/datasets/preprocessing.py'
        ], capture_output=True, text=True, cwd='/opt/airflow/project')
        
        if result2.returncode != 0:
            raise Exception(f"Preprocessing failed: {result2.stderr}")
            
        print("Data engineering completed successfully")
        
    except Exception as e:
        print(f"Data engineering error: {e}")
        raise

def run_model_engineering():
    """Stage 2: Model Engineering"""
    import subprocess
    try:
        result = subprocess.run([
            'python', '/opt/airflow/project/code/modelsEngs/train_model.py'
        ], capture_output=True, text=True, cwd='/opt/airflow/project')
        
        if result.returncode != 0:
            raise Exception(f"Model training failed: {result.stderr}")
            
        print("Model engineering completed successfully")
        
    except Exception as e:
        print(f"Model engineering error: {e}")
        raise

def run_deployment():
    """Stage 3: Deployment"""
    import subprocess
    try:
        subprocess.run([
            'docker-compose', 'down'
        ], cwd='/opt/airflow/project/code/deployment', capture_output=True)
        
        result = subprocess.run([
            'docker-compose', 'up', '--build', '-d'
        ], cwd='/opt/airflow/project/code/deployment', capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Deployment failed: {result.stderr}")
            
        print("Deployment completed successfully")
        
    except Exception as e:
        print(f"Deployment error: {e}")
        raise

data_engineering_task = PythonOperator(
    task_id='data_engineering',
    python_callable=run_data_engineering,
    dag=dag,
)

model_engineering_task = PythonOperator(
    task_id='model_engineering',
    python_callable=run_model_engineering,
    dag=dag,
)

deployment_task = PythonOperator(
    task_id='deployment',
    python_callable=run_deployment,
    dag=dag,
)

data_engineering_task >> model_engineering_task >> deployment_task