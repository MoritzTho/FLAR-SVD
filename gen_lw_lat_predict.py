#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Low-rank Weight Latency Prediction Model
---------------------------------------
This script trains a Random Forest model to predict latency recovery
for low-rank weight operations based on various input features.
"""

import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# import matplotlib.pyplot as plt


def load_data(filepath):
    """Load and prepare the dataset."""
    # df = pd.read_csv(filepath)        # Load a single CSV file

    # List of CSV files
    csv_files = glob.glob(filepath)
    # Read and concatenate all CSV files
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def prepare_features(df):
    """Extract features and target from the dataset."""
    # Define input features and target variable
    X = df[['Batch_dim', 'Patch_dim', 'In_feats', 'Out_feats', 'Rank']]
    y = df['Lat_recovery']
    
    return X, y


def train_model(X, y, test_size=0.2, random_state=42, n_estimators=100):
    """Train a Random Forest regression model."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create and train the model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print performance metrics."""
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print evaluation metrics
    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    return y_pred, mse, r2


def save_model(model, filepath):
    """Save the trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def main():
    """Main execution function."""
    input_file = '/workspace/layer_wise_latency*.csv'
    output_model = '/workspace/rf_model_k_mult_8.pkl'
    
    # Process pipeline
    print("Loading data...")
    df = load_data(input_file)
    
    print("\nPreparing features...")
    X, y = prepare_features(df)
    
    print("\nTraining model...")
    model, _, X_test, _, y_test = train_model(X, y)
    
    print("\nEvaluating model...")
    _ = evaluate_model(model, X_test, y_test)
    
    print("\nSaving model...")
    save_model(model, output_model)
    
    print("\nProcess completed successfully!")


if __name__ == "__main__":
    main()
