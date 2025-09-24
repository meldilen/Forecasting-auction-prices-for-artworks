#!/bin/bash

# Initialize Airflow database
docker-compose -f docker-compose-airflow.yml up -d postgres
sleep 10

# Initialize Airflow
docker-compose -f docker-compose-airflow.yml run airflow-webserver airflow db init
docker-compose -f docker-compose-airflow.yml run airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start all services
docker-compose -f docker-compose-airflow.yml up -d

echo "Airflow is starting... Check http://localhost:8080"
echo "Username: admin, Password: admin"