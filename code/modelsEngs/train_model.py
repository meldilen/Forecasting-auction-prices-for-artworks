import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import json
import os
import warnings
import joblib

warnings.filterwarnings('ignore')

models_dir = "../../models/"
os.makedirs(models_dir, exist_ok=True)

# Load data
final_df = pd.read_csv('../../data/processed/final_art_dataset.csv')

def main():
    """Main function for model engineering pipeline"""
    
    # Step 1: Handle missing target values
    print("Processing data...")
    train_df = final_df.dropna(subset=['price_usd']).copy()
    
    # Step 2: Prepare features and target
    X = train_df.drop('price_usd', axis=1)
    y = train_df['price_usd']
    
    # Step 3: Define feature types
    feature_analysis = []
    for feature in X.columns:
        non_null_count = X[feature].notnull().sum()
        non_null_pct = (non_null_count / len(X)) * 100
        feature_analysis.append({
            'feature': feature,
            'non_null_count': non_null_count,
            'non_null_pct': non_null_pct,
            'dtype': X[feature].dtype
        })
    
    feature_analysis_df = pd.DataFrame(feature_analysis)
    
    # Select features with sufficient data (>50% non-null values)
    usable_numeric_features = feature_analysis_df[
        (feature_analysis_df['dtype'].isin(['float64', 'int64'])) & 
        (feature_analysis_df['non_null_pct'] > 50)
    ]['feature'].tolist()
    
    usable_categorical_features = feature_analysis_df[
        (feature_analysis_df['dtype'] == 'object') & 
        (feature_analysis_df['non_null_pct'] > 50)
    ]['feature'].tolist()
    
    # Remove identifier features
    identifier_features = ['artist_id', 'artist', 'title']
    modeling_numeric_features = [f for f in usable_numeric_features if f not in identifier_features]
    modeling_categorical_features = [f for f in usable_categorical_features if f not in identifier_features]
    
    # Step 4: Create preprocessing pipelines
    if len(modeling_numeric_features) > 0:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    else:
        numeric_transformer = 'drop'
    
    if len(modeling_categorical_features) > 0:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
        ])
    else:
        categorical_transformer = 'drop'
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, modeling_numeric_features),
            ('cat', categorical_transformer, modeling_categorical_features)
        ],
        remainder='drop'
    )
    
    # Step 5: Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    feature_names = []
    if len(modeling_numeric_features) > 0:
        feature_names.extend(modeling_numeric_features)
    
    if len(modeling_categorical_features) > 0:
        cat_transformer = preprocessor.named_transformers_['cat']
        cat_feature_names = cat_transformer.named_steps['onehot'].get_feature_names_out(modeling_categorical_features)
        feature_names.extend(cat_feature_names)
    
    # Step 6: Split data
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed_df, 
        y, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    
    # Apply logarithmic transformation to target variable
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    # Step 7: Setup MLflow
    mlflow.set_tracking_uri("file:../models/mlruns")
    mlflow.set_experiment("Art_Auction_Price_Prediction")
    
    # Step 8: Train models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        "LinearRegression": LinearRegression()
    }
    
    best_model = None
    best_score = float('inf')
    best_model_name = ""
    
    print("Training models...")
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Train model
            model.fit(X_train, y_train_log)
            
            # Make predictions
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", model_name)
            if model_name == "RandomForest":
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 10)
            
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.sklearn.log_model(model, f"{model_name.lower()}_model")
            
            # Update best model
            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_model_name = model_name
    
    # Step 9: Save best model
    print(f"Best model: {best_model_name} with RMSE: ${best_score:,.2f}")
    
    if best_model is not None:
        # Save using joblib
        model_filename = f"art_auction_price_model.pkl"
        model_filepath = os.path.join(models_dir, model_filename)
        joblib.dump(best_model, model_filepath)
        
        # Save metadata
        model_metadata = {
            "model_name": best_model_name,
            "training_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "features_used": list(X_train.columns),
            "performance_metrics": {
                "rmse": float(best_score),
                "r2_score": float(r2_score(y_test, np.expm1(best_model.predict(X_test))))
            },
            "data_dimensions": {
                "training_samples": X_train.shape[0],
                "features_count": X_train.shape[1],
                "test_samples": X_test.shape[0]
            }
        }
        
        metadata_filepath = os.path.join(models_dir, "model_metadata.json")
        with open(metadata_filepath, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Save predictions
        y_pred_log = best_model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        
        results_df = pd.DataFrame({
            'actual_price': y_test.values,
            'predicted_price': y_pred,
            'absolute_error': np.abs(y_test.values - y_pred),
            'percentage_error': (np.abs(y_test.values - y_pred) / y_test.values) * 100
        })
        
        results_filepath = os.path.join(models_dir, "model_predictions.csv")
        results_df.to_csv(results_filepath, index=False)
        
        print(f"Model saved: {model_filepath}")
        print(f"MAE: ${results_df['absolute_error'].mean():,.2f}")
        print(f"R² Score: {model_metadata['performance_metrics']['r2_score']:.4f}")
    
    print("Model engineering completed successfully!")

if __name__ == "__main__":
    main()