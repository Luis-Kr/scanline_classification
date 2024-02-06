import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import utils.logger as lgr
from utils.header_dict import header_dict

# Set the root directory to the project directory (.../pcd_mesh)
root_dir = Path(__file__).resolve().parent.parent.parent
print(f'Root directory: {root_dir}')

 # Set up the logger
logger = lgr.logger_setup('rf_training', 
                            Path(root_dir) / "data/logs/rf_training.log")


def load_data(file_paths):
    """
    Load data from a CSV file or a list of CSV files and return a pandas DataFrame.

    Args:
        file_paths (str or list of str): The path(s) to the CSV file(s).

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded data.
    """
    print(f'Loading data')
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    data_frames = []
    for file_path in file_paths:
        data = pd.read_csv(file_path, delimiter=' ', header=None,
                           names=list(header_dict.keys()))
        data_frames.append(data)

    return pd.concat(data_frames, axis=0)


def train_and_test_model(X_train, y_train, X_test, y_test, model_path, weights_dict, logger):
    # Set up random forest classifier
    logger.info("Setting up random forest classifier...")
    
    # Load the already trained model if it exists
    if model_path.exists():
        rf = joblib.load(model_path)
    else:
        rf = RandomForestClassifier(n_estimators=150, 
                                    criterion='gini',
                                    max_features='sqrt',
                                    class_weight=weights_dict,
                                    n_jobs=-1,
                                    random_state=42)

    # Train the model
    logger.info("Training the model...")
    rf.fit(X_train, y_train)

    # Predict on test set
    logger.info("Predicting on test set...")
    y_pred = rf.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy score: {accuracy}")

    # Save the model
    joblib.dump(rf, model_path)
    logger.info("Model saved successfully.")


def create_class_weights(labels):
    """
    Create class weights for the given labels.

    Args:
        y_train (np.ndarray): The labels to create class weights for.

    Returns:
        dict: A dictionary containing the class weights.
    """
    unique_labels = np.unique(labels)
    class_weights = class_weight.compute_class_weight('balanced', 
                                                      classes=unique_labels, 
                                                      y=labels)
    weights_dict = {class_label: weight for class_label, weight in zip(unique_labels, class_weights)}
    return weights_dict


if __name__ == "__main__":
    # Load training data
    data_dir = Path(root_dir) / "data/06_subsampling/ScanPos_Relocated"
    model_path = Path(root_dir) / "data/models/random_forest_SiteA_ScanPosRelocated_ClassWeights.joblib"
    data_paths = list(data_dir.glob('*.txt'))
    data = load_data(data_paths)
    
    # Split data into features and labels
    X = data.iloc[:, 9:-1].values
    y = data.iloc[:, -1].values
    
    # Create class weights
    weights_dict = create_class_weights(labels=y)
    
    print(f'Class weights: {weights_dict}')

    # Split data into training and testing sets using 5-fold cross validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   
    for index, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        logger.info(f"{index}")
        logger.info(f"X_train shape (training): {X_train.shape}")
        logger.info(f"y_train shape (training): {y_train.shape}")
        logger.info(f"X_test shape (validation): {X_test.shape}")
        logger.info(f"y_test shape (validation): {y_test.shape}")

        train_and_test_model(X_train, y_train, X_test, y_test, model_path, weights_dict, logger)