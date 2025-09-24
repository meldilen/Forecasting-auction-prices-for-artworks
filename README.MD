# Art Auction Price Prediction Pipeline 🎨🤖

A fully automated machine learning pipeline for predicting art auction prices, built with modern MLOps tools.

## Overview

This project demonstrates a complete ML pipeline that:
- Processes raw art auction data
- Trains and evaluates a predictive model
- Deploys the model as a REST API
- Provides a web interface for predictions
- Runs automatically every 5 minutes

## Project Structure

```
├── code/
│   ├── datasets/              # Data processing scripts
│   ├── models/               # Model training code
│   └── deployment/
│       ├── api/              # FastAPI backend
│       └── app/              # Streamlit frontend
├── data/
│   ├── raw/                  # Raw dataset files
│   └── processed/            # Processed training/test data
├── models/                   # Trained model files
├── services/
│   └── airflow/
│       └── dags/             # Pipeline automation
├── notebooks/               # Jupyter notebooks for exploration
└── requirements.txt         # Python dependencies
```

## Pipeline Architecture

### Stage 1: Data Engineering
- **Input**: Raw art auction data (CSV files)
- **Processing**: Data cleaning, outlier removal, train-test split
- **Output**: Processed training and testing datasets
- **Tools**: Pandas, Scikit-learn

### Stage 2: Model Engineering
- **Input**: Processed training data
- **Processing**: Feature engineering, model training, evaluation
- **Output**: Trained model file, performance metrics
- **Tools**: MLflow, Scikit-learn, Random Forest, Linear Regression

### Stage 3: Deployment
- **API**: FastAPI REST service for predictions
- **Web App**: Streamlit interface for users
- **Containerization**: Docker and Docker Compose
- **Tools**: FastAPI, Streamlit, Docker

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/meldilen/Forecasting-auction-prices-for-artworks.git
cd art-auction-pipeline
```

2. **Set up the environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run the pipeline with Airflow**
```bash
cd services/airflow
docker-compose up -d
```

4. **Access the applications**
- **Web App**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Airflow UI**: http://localhost:8080 (admin/admin)

## Usage

### Making Predictions

1. Open the Streamlit app at `http://localhost:8501`
2. Fill in the artwork details:
   - Creation year, dimensions, signature status
   - Artist information (lifespan, exhibition history)
   - Art period and movement
3. Click "Predict Auction Price" to get the estimated value

## Automation Features

- **Scheduled Pipeline**: Runs every 5 minutes via Airflow
- **Health Monitoring**: Automatic health checks for all services
- **Model Versioning**: MLflow tracking for experiment management
- **Container Orchestration**: Docker Compose for service management

## Technologies Used

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, MLflow
- **API**: FastAPI, Pydantic
- **Web Interface**: Streamlit
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Apache Airflow
- **Monitoring**: Health checks, logging

## Development

### Adding New Features

1. **Data Processing**: Modify `code/datasets/data_pipeline.py`
2. **Model Training**: Update `code/modelsEngs/train_model.py`
3. **API Endpoints**: Edit `code/deployment/api/main.py`
4. **Web Interface**: Modify `code/deployment/app/app.py`

## Monitoring and Logs

- **Airflow DAGs**: Monitor pipeline execution at `http://localhost:8080`
- **API Logs**: Check container logs with `docker logs art_price_api`
- **App Logs**: Check container logs with `docker logs art_price_app`
- **MLflow Tracking**: Model metrics and version history

## Acknowledgments

- Dataset: Art auction historical data
- Tools: FastAPI, Streamlit, Airflow, MLflow, Docker
- Course: MLOps Lab Assignment

---

**Note**: This is a demonstration project for educational purposes. The prediction model should be validated with real-world data before production use.