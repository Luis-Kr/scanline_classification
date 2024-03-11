import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.utils import class_weight
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from pathlib import Path 
import sys
import pickle

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import utils.logger as lgr

# Set the root directory to the project directory (.../pcd_mesh)
root_dir = Path(__file__).resolve().parent.parent.parent
print(f'Root directory: {root_dir}')

 # Set up the logger
logger = lgr.logger_setup('rf_training', 
                          Path(root_dir) / "data/data_for_training/logs/rf_training.log")


def train_model(x_train, y_train, logger):
    # Set up random forest classifier
    logger.info("Setting up random forest classifier...")
    
    # Create the model
    rf = RandomForestClassifier(n_estimators=75, 
                                max_depth=10,
                                min_samples_leaf=1,
                                min_samples_split=10,
                                criterion='log_loss',
                                max_features='sqrt',
                                n_jobs=-1,
                                random_state=42)

    # Train the model
    logger.info("Training the model...")
    rf.fit(x_train, y_train)

    return rf


def evaluate_model(model, x_test, y_test, logger):
    # Evaluate the model
    logger.info("Evaluating the model...")
    
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    logger.info(f"Accuracy: {accuracy:.3f}")

    return accuracy, f1, precision, recall
    

def split_data_into_features_labels(training_data, testing_data, logger, statistic="mean"):
    # Split the data into features and labels
    logger.info("Splitting the data into features and labels...")
    #X = data.drop(columns=['label', 'label_names'])

    # Training data
    stats_columns = training_data.columns[training_data.columns.str.contains('intensity|red|green|blue|nx|ny|nz|slope|curvature|roughness')]
    x_train = training_data[stats_columns]
    #x_train = x_train.drop(columns=['rho_mean', 'nx_xyz_mean', 'ny_xyz_mean', 'nz_xyz_mean', 'slope_mean', 'curvature_mean'])
    #x_train = pd.concat([training_data["z_median_nn"], x_train], axis=1)
    #x_train = training_data.drop(columns=['label', 'label_names'])
    y_train = training_data['label']
    
    # Testing_data
    stats_columns = testing_data.columns[testing_data.columns.str.contains('intensity|red|green|blue|nx|ny|nz|slope|curvature|roughness')]
    x_test = testing_data[stats_columns]
    #x_test = x_test.drop(columns=['rho_mean', 'nx_xyz_mean', 'ny_xyz_mean', 'nz_xyz_mean', 'slope_mean', 'curvature_mean'])
    # x_test = pd.concat([testing_data["z_median_nn"], x_test], axis=1)
    #x_test = testing_data.drop(columns=['label', 'label_names'])    
    y_test = testing_data['label']
    
    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save the column names 
    column_names_path = Path("/DATA/Luis/thesis/scanline_classification/models/rf_training/feature_importance_workflow/column_names.pkl")
    with open(column_names_path, 'wb') as f:
        pickle.dump(x_train.columns, f)
    
    return x_train, x_test, y_train, y_test


def main(training_data_path, testing_data_path, output_dir, logger):
    # Load the data
    logger.info(f"Loading data from {training_data_path}...")
    training_data = pd.read_csv(training_data_path, sep=",", header=0)
    testing_data = pd.read_csv(testing_data_path, sep=",", header=0)

    # Split the data into features and labels
    x_train, x_test, y_train, y_test = split_data_into_features_labels(training_data, testing_data, logger)
    
    # Training the model
    rf_model = train_model(x_train, y_train, logger)
    
    # Evaluate the model
    accuracy, f1, precision, recall = evaluate_model(rf_model, x_test, y_test, logger)
    
    # Save the model
    logger.info("Saving the model...")
    output_dir.mkdir(parents=False, exist_ok=True)
    model_path = output_dir / "rf_model_mean_statistics.joblib"
    joblib.dump(rf_model, model_path)
    
    # Save the evaluation results to a csv file
    logger.info("Saving the evaluation results...")
    evaluation_results = {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
    evaluation_results_path = output_dir / "rf_model_performance_mean_statistics.csv"
    pd.DataFrame(evaluation_results, index=[0]).to_csv(evaluation_results_path, index=False)
    
    
if __name__=="__main__":
    training_data_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training/scanline_subsampling/merged/training_data_merged_frac.csv")
    testing_data_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_validation/merged/validation_data_merged_frac.csv")
    output_dir = Path("/DATA/Luis/thesis/scanline_classification/models/rf_training/feature_importance_workflow")
    
    main(training_data_path, testing_data_path, output_dir, logger)
    