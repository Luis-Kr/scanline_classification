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


def define_hyperparameters():
    bootstrap = [True, False]
    n_estimators = [10, 50, 100, 200, 400, 600, 800, 1000]
    max_features = ["sqrt", "log2", None]
    max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    
    return {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap}


def train_model(x_train, y_train, logger):
    # Set up random forest classifier
    logger.info("Setting up random forest classifier...")
    
    # Create the model
    rf = RandomForestClassifier()
    
    rf_randomized = RandomizedSearchCV(estimator = rf, 
                                       param_distributions = define_hyperparameters(), 
                                       n_iter = 30, 
                                       cv = 3, 
                                       verbose=2, 
                                       random_state=42, 
                                       n_jobs = -1)

    # Train the model
    logger.info("Training the model...")
    rf_randomized.fit(x_train, y_train)

    return rf_randomized


def evaluate_model(model, x_test, y_test, logger):
    # Evaluate the model
    logger.info("Evaluating the model...")
    
    y_pred = model.predict(x_test)
    errors = abs(y_pred - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    logger.info(f"Mean Absolute Percentage Error: {mape:.3f}")
    logger.info(f"Accuracy: {accuracy:.3f}")

    return mape, accuracy, f1, precision, recall
    

def split_data_into_features_labels(data, logger):
    # Split the data into features and labels
    logger.info("Splitting the data into features and labels...")
    X = data.drop(columns=['label', 'label_names'])
    y = data['label']
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test


def main(data_path, output_dir, logger):
    # Load the data
    logger.info(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path, sep=",", header=0)

    # Split the data into features and labels
    x_train, x_test, y_train, y_test = split_data_into_features_labels(data, logger)
    
    # Training the model
    rf_model = train_model(x_train, y_train, logger)
    
    # Evaluate the model
    mape, accuracy, f1, precision, recall = evaluate_model(rf_model.best_estimator_, x_test, y_test, logger)
    
    # Save the model
    logger.info("Saving the model...")
    output_dir.mkdir(parents=False, exist_ok=True)
    model_path = output_dir / "rf_model_RandomizedSearchCV_result.pkl"
    joblib.dump(rf_model, model_path)
    
    # Save the evaluation results to a csv file
    logger.info("Saving the evaluation results...")
    evaluation_results = {"mape": mape, "accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
    evaluation_results_path = output_dir / "evaluation_RandomizedSearchCV_bestmodel.csv"
    pd.DataFrame(evaluation_results, index=[0]).to_csv(evaluation_results_path, index=False)
    
    
if __name__=="__main__":
    data_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training/scanline_subsampling/merged/training_data_merged_frac.csv")
    output_dir = Path("/DATA/Luis/thesis/scanline_classification/models/rf_training/RandomizedSearchCV_bestmodel_allfeatures_frac")
    
    main(data_path, output_dir, logger)
    