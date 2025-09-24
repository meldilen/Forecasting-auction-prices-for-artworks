#!/usr/bin/env python3
import os
import sys
import time
import schedule
import subprocess
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_script(script_name, description):
    logger.info(f"Starting {description}...")
    
    try:
        result = subprocess.run([
            'python', script_name
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            logger.info(f"{description} completed successfully")
            return True
        else:
            logger.error(f"{description} failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running {description}: {e}")
        return False

def data_engineering_stage():
    """Stage 1: Data Engineering"""
    logger.info("=== STAGE 1: Data Engineering ===")
    
    # Run EDA first to combine datasets
    if not run_script('eda.py', 'EDA processing'):
        return False
    
    # Run preprocessing to clean and prepare data
    if not run_script('preprocessing.py', 'Data preprocessing'):
        return False
    
    logger.info("Data engineering stage completed")
    return True

def model_engineering_stage():
    """Stage 2: Model Engineering"""
    logger.info("=== STAGE 2: Model Engineering ===")
    
    if not run_script('train_model.py', 'Model training'):
        return False
    
    logger.info("Model engineering stage completed")
    return True

def deployment_stage():
    """Stage 3: Deployment"""
    logger.info("=== STAGE 3: Deployment ===")
    
    try:
        # Build and run Docker containers
        deployment_dir = os.path.join(os.path.dirname(__file__), 'deployment')
        
        # Stop existing containers
        subprocess.run([
            'docker-compose', 'down'
        ], cwd=deployment_dir, capture_output=True)
        
        # Build and start new containers
        result = subprocess.run([
            'docker-compose', 'up', '--build', '-d'
        ], cwd=deployment_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Deployment stage completed - API and App are running")
            return True
        else:
            logger.error(f"Deployment failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error in deployment stage: {e}")
        return False

def run_pipeline():
    """Run the complete pipeline"""
    logger.info("🚀 Starting automated pipeline execution")
    start_time = datetime.now()
    
    success = True
    
    # Stage 1: Data Engineering
    if not data_engineering_stage():
        success = False
        logger.error("Data engineering stage failed")
    
    # Stage 2: Model Engineering
    if success and not model_engineering_stage():
        success = False
        logger.error("Model engineering stage failed")
    
    # Stage 3: Deployment
    if success and not deployment_stage():
        success = False
        logger.error("Deployment stage failed")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if success:
        logger.info(f"✅ Pipeline completed successfully in {duration:.2f} seconds")
    else:
        logger.error(f"❌ Pipeline failed after {duration:.2f} seconds")
    
    return success

def main():
    logger.info("Art Auction Price Prediction Pipeline Started")
    
    run_pipeline()
    
    schedule.every(5).minutes.do(run_pipeline)
    
    logger.info("Pipeline scheduled to run every 5 minutes")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()