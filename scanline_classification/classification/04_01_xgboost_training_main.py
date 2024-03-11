import logging
import pandas as pd
from xgboost import XGBClassifier
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
import time
import json

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig

import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import utils.logger as lgr

# Set the root directory to the project directory (.../pcd_mesh)
root_dir = Path(__file__).resolve().parent.parent.parent
print(f'Root directory: {root_dir}')


def train_model(x_train, y_train, logger, n_estimators, max_depth, learning_rate):
    # Set up XGBoost classifier
    logger.info("Setting up XGBoost classifier...")
    
    # Create the model
    xgb = XGBClassifier(n_estimators=n_estimators, 
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_jobs=-1,
                        random_state=42)

    # Train the model
    logger.info("Training the model...")
    xgb.fit(x_train, y_train)

    return xgb


def evaluate_model(model, x_test, y_test, logger):
    # Evaluate the model
    logger.info("Evaluating the model...")
    
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    logger.info(f"Accuracy: {accuracy:.3f}")
    logger.info(f"F1-score: {f1:.3f}")

    return accuracy, f1, precision, recall
    

def split_data_into_features_labels(training_data, testing_data, output_dir, logger, statistic="mean"):
    # Split the data into features and labels
    logger.info("Splitting the data into features and labels...")
    #X = data.drop(columns=['label', 'label_names'])

    # Training data
    stats_columns1 = training_data.columns[training_data.columns.str.contains('intensity|red|green|blue|slope|curvature|roughness|vert_angle|nz', case=True) & 
                                           ~training_data.columns.str.contains('nx_xyz|ny_xyz|nz_xyz|nx|ny', case=True)]
    x_train = training_data[stats_columns1]
    # stats_columns2 = x_train.columns[x_train.columns.str.contains('mean|median|perc2nd|perc25th')] #'rho|intensity|red|green|nz|slope|roughness|nz_xyz'
    # x_train = x_train[stats_columns2]
    y_train = training_data['label']
    
    # Testing_data
    stats_column1 = testing_data.columns[testing_data.columns.str.contains('intensity|red|green|blue|slope|curvature|roughness|vert_angle|nz', case=True) & 
                                         ~testing_data.columns.str.contains('nx_xyz|ny_xyz|nz_xyz|nx|ny', case=True)]
    x_test = testing_data[stats_column1]
    # stats_column2 = x_test.columns[x_test.columns.str.contains('mean|median|perc2nd|perc25th')]
    # x_test = x_test[stats_column2]
    y_test = testing_data['label']
    
    # Save the column names 
    #column_names_path = Path("/DATA/Luis/thesis/scanline_classification/models/xgb_training/feature_importance_workflow/column_names.pkl")
    column_names_path = Path(output_dir) / "column_names.pkl"
    with open(column_names_path, 'wb') as f:
        pickle.dump(x_train.columns, f)
        
    column_names_path = Path(output_dir) / "column_names.json"
    with open(column_names_path, 'w') as f:
        for item in list(x_train.columns):
            f.write(json.dumps(item) + '\n')
        
    print(f"Column shape {x_test.columns.shape}")
    print(f"Column names {x_test.columns}")
    
    return x_train, x_test, y_train, y_test


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig):
    # Clear the hydra config cache
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
     # Set up the logger
    logger = lgr.logger_setup('xgb_training', 
                              Path(cfg.training.output_dir) / "logs/xgb_training.log")
    
    # Load the data
    logger.info(f"Loading data from {cfg.training.training_data_path}...")
    training_data = pd.read_csv(cfg.training.training_data_path, sep=",", header=0)
    testing_data = pd.read_csv(cfg.training.testing_data_path, sep=",", header=0)
    
    # # Remove rows where label_name is unclassified
    # training_data = training_data[training_data['label'] != 0]
    # testing_data = testing_data[testing_data['label'] != 0]
    
    # # Decrease the label values by 1
    # training_data['label'] = training_data['label'] - 1
    # testing_data['label'] = testing_data['label'] - 1

    # Split the data into features and labels
    x_train, x_test, y_train, y_test = split_data_into_features_labels(training_data, testing_data, cfg.training.output_dir,logger)
    
    # Training the model
    start_time = time.perf_counter()
    xgb_model = train_model(x_train, y_train, logger, cfg.training.n_estimators, cfg.training.max_depth, cfg.training.learning_rate)
    end_time = time.perf_counter()
    training_time = end_time - start_time
    logger.info(f"Training completed. It took {training_time:.2f} seconds.")
    
    # Evaluate the model
    accuracy, f1, precision, recall = evaluate_model(xgb_model, x_test, y_test, logger)
    
    # Save the model
    logger.info("Saving the model...")
    Path(cfg.training.output_dir).mkdir(parents=False, exist_ok=True)
    model_path = Path(cfg.training.output_dir) / "xgb_model.joblib"
    joblib.dump(xgb_model, model_path)
    
    # Save the evaluation results to a csv file
    logger.info("Saving the evaluation results...")
    evaluation_results = {"accuracy": round(accuracy,4), 
                          "f1": round(f1,4), 
                          "precision": round(precision,4), 
                          "recall": round(recall,4), 
                          "training_time": round(training_time,4), 
                          "n_estimators": cfg.training.n_estimators, 
                          "max_depth": cfg.training.max_depth, 
                          "learning_rate": cfg.training.learning_rate}
    
    evaluation_results_path = Path(cfg.training.output_dir) / "xgb_model_performance.csv"
    
    # check if the file exists
    if evaluation_results_path.exists():
        # append the results to the file
        pd.DataFrame(evaluation_results, index=[0]).to_csv(evaluation_results_path, mode='a', header=False, index=False)
    else:
        # create a new file
        pd.DataFrame(evaluation_results, index=[0]).to_csv(evaluation_results_path, index=False, header=True)


if __name__=="__main__":
    main()