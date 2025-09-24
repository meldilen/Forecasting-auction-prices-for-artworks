#!/usr/bin/env python3
import os
import time
import schedule
from datetime import datetime

def run_stage(stage_name, command):
    """Run a pipeline stage"""
    print(f"\n{'='*50}")
    print(f"Running {stage_name} at {datetime.now()}")
    print(f"{'='*50}")
    
    result = os.system(command)
    if result == 0:
        print(f"✓ {stage_name} completed successfully")
    else:
        print(f"✗ {stage_name} failed")
    return result

def run_full_pipeline():
    """Run the entire pipeline"""
    print("🚀 Starting Art Auction Pipeline...")
    
    # Stage 1: Data Engineering
    run_stage("EDA", "cd datasets && python EDA.py")
    run_stage("Preprocessing", "cd datasets && python preprocessing.py")
    
    # Stage 2: Model Engineering
    run_stage("Model Training", "cd models && python train_model.py")
    
    # Stage 3: Deployment
    run_stage("Deployment", "cd deployment && docker-compose up -d --build")
    
    print("\n🎉 Pipeline completed!")

if __name__ == "__main__":
    # Run immediately
    run_full_pipeline()
    
    # Schedule to run every 5 minutes
    schedule.every(5).minutes.do(run_full_pipeline)
    
    print("⏰ Pipeline scheduled to run every 5 minutes...")
    
    while True:
        schedule.run_pending()
        time.sleep(1)