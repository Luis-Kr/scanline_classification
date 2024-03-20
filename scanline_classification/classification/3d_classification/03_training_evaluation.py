import logging
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import numpy as np
import joblib
from pathlib import Path 
import sys
import pickle
import time
import json
from numba import njit, prange
import time
import random
from pprint import pprint
import gzip

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

import utils.logger as lgr

def create_class_weights(labels):
    # Get the unique labels
    unique_labels = np.unique(labels)
    
    # Create class weights
    class_weights = class_weight.compute_class_weight('balanced', 
                                                      classes=unique_labels, 
                                                      y=labels)
    
    # Create sample weights
    sample_weights = class_weight.compute_sample_weight('balanced', labels)
    
    # Create a dictionary of class weights
    weights_dict = {class_label: weight for class_label, weight in zip(unique_labels, class_weights)}
    
    return weights_dict, sample_weights


def train_model(x_train, y_train, logger, n_estimators, max_depth, learning_rate, sample_weights):
    # Set up XGBoost classifier
    logger.info("Setting up XGBoost classifier...")
    
    # Create the XGB model
    xgb = XGBClassifier(n_estimators=n_estimators, 
                        max_depth=max_depth,
                        gamma=0.5,
                        learning_rate=learning_rate,
                        sample_weight=sample_weights,
                        n_jobs=-1,
                        random_state=42)

    # Train the model
    logger.info("Training the model...")
    time_start = time.time()
    xgb.fit(x_train, y_train)
    time_end = time.time()
    training_time = time_end - time_start   
    logger.info(f"Training time: {training_time:.2f}s")

    return xgb


def prediction(model, x_test, logger):
    logger.info("Prediction...")
    
    # Make predictions
    time_start = time.time()
    y_pred = model.predict(x_test)
    time_end = time.time()
    prediction_time = time_end - time_start   
    logger.info(f"Prediction time: {prediction_time:.2f}s")
    
    return y_pred


def evaluate_model(y_pred, y_test, logger):
    logger.info("Evaluating the model...")
    
    # Get the confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    # Get the classification report
    cls_report = classification_report(y_test, y_pred, digits=3, target_names=np.array(["unclassified", "man-made objects", "ground", "tree trunk/branches", "leaves", "low vegetation"]), output_dict=True)

    return cnf_matrix, cls_report


def training(cfg, logger):
    output_dir = Path(cfg.cls_3d.output_dir)
    
    # Load the training data
    training_data = pd.read_csv(output_dir / cfg.cls_3d.training.training_data_path, sep=",", header=0)
    
    # Get a list of all files in the directory
    files_siteA = sorted(list((output_dir / cfg.cls_3d.training.training_data_subsampled_dir).glob('SiteA*.gz')))
    files_siteB = sorted(list((output_dir / cfg.cls_3d.training.training_data_subsampled_dir).glob('SiteB*.gz')))
    
    selected_files = [ [files_siteA[4], files_siteB[0]], [files_siteA[5], files_siteB[2]], [files_siteA[1], files_siteB[4]] ]
    
    # Initialize list to store training reports
    cls_training_report_dfs = []
    
    # Log hyperparameters
    logger.info('--------------------------------------------------------------------------------')
    logger.info('--------------------------------------------------------------------------------')
    logger.info(f"n_estimators: {cfg.training.n_estimators}")
    logger.info(f"max_depth: {cfg.training.max_depth}")
    logger.info(f"learning_rate: {cfg.training.learning_rate}")

    # For each fold, train on 10 files and test on 2 files
    for i, (test_indices1, test_indices2) in enumerate(selected_files):
        logger.info(f"::: Fold {i+1} ::::")
        
        # Get testing file names
        file_name1 = test_indices1.stem
        file_name2 = test_indices2.stem
        
        logger.info(f"Input data shape: {training_data.shape}")
        
        # Subset training data
        training_data_subset = training_data[
            (training_data['path'] != file_name1) &
            (training_data['path'] != file_name2)
        ]
       
        # Get training data
        x_train = training_data_subset.drop(columns=['label', 'label_names', 'path'])
        y_train = training_data_subset['label']  
        
        logger.info(f"Training data shape: {x_train.shape}")
        
        # Create class weights
        _, sample_weights = create_class_weights(y_train)
        
        # Train the model
        xgb_model = train_model(x_train, y_train, logger, cfg.training.n_estimators, cfg.training.max_depth, cfg.training.learning_rate, sample_weights)
        
        # Define label names
        label_names={0: "unclassified",
                     1: "man-made objects",
                     2: "ground",
                     3: "tree trunk/branches",
                     4: "leaves",
                     5: "low vegetation"}
        
        # Load attribute statistics
        with open(output_dir / cfg.cls_3d.sampling.attributes_path, 'rb') as f:
                attribute_statistics = pickle.load(f)
        
        # Test the model
        for file_path in selected_files[i]:
            logger.info(":: Prediction on the test data ::")
            logger.info(f"Testing on {file_path}")
            
            # Load test data
            pcd_attributes_file = gzip.GzipFile(file_path, "r")
            pcd_attributes = np.load(pcd_attributes_file)
        
            # Create a DataFrame from the attributes
            test_data_subsampled = pd.DataFrame(pcd_attributes, columns=attribute_statistics)
            filename = Path(file_path).stem
            
            # Map label names
            test_data_subsampled['label_names'] = test_data_subsampled['label'].map(label_names)
            
            # Get test data
            x_test = test_data_subsampled.drop(columns=['x', 'y', 'z', 'label', 'label_names'])
            y_test = test_data_subsampled['label'] 
            
            # Make predictions
            y_pred = prediction(xgb_model, x_test, logger)
            
            # Evaluate model
            cnf_matrix_fr, cls_report_fr = evaluate_model(y_pred=y_pred, 
                                                          y_test=y_test,
                                                          logger=logger)
            
            # Write cls_report_fr to a pandas dataframe 
            cls_report_fr_df = pd.DataFrame(cls_report_fr).transpose()
            cls_report_fr_df['n_estimators'] = cfg.training.n_estimators
            cls_report_fr_df['max_depth'] = cfg.training.max_depth
            cls_report_fr_df['learning_rate'] = cfg.training.learning_rate
            cls_report_fr_df['filename'] = filename
            cls_report_fr_df['fold'] = i+1
            cls_report_fr_df['id'] = cfg.training.id
            
            # Save the confusion matrix
            cnf_matrix_path = output_dir / "confusion_matrices_testing"
            cnf_matrix_path.mkdir(parents=True, exist_ok=True)
            np.savetxt(cnf_matrix_path / f"id{cfg.training.id}__{filename}__nestimators{cfg.training.n_estimators}_maxdepth{cfg.training.max_depth}_learningrate{cfg.training.learning_rate}_confusion_matrix_testing.csv", 
                       cnf_matrix_fr, delimiter=',', fmt='%u')
            
            # Save x,y,z labels and predicted labels as numpy array
            save_classified_pcd = output_dir / "classified_pcd" / "testing"
            save_classified_pcd.mkdir(parents=True, exist_ok=True)
            pcd_out = np.c_[test_data_subsampled['x'], test_data_subsampled['y'], test_data_subsampled['z'], y_test, y_pred]
            
            logger.info(f"Saving classified point cloud to: {save_classified_pcd}")
            np.savetxt(save_classified_pcd / f"{filename}_xyz_labels_predicted_labels.txt", pcd_out, delimiter=' ', fmt='%1.4f %1.4f %1.4f %u %u')
        
            # Append the cls_report_fr_df to the list
            cls_training_report_dfs.append(cls_report_fr_df)
    
    # Concatenate the cls_report_fr_dfs
    cls_training_report = pd.concat(cls_training_report_dfs)
    
    # Save the files
    classification_report_dir = output_dir / "classification_report_testing"
    classification_report_dir.mkdir(parents=True, exist_ok=True)
    cls_training_report.to_csv(classification_report_dir / f"id{cfg.training.id}_nestimators{cfg.training.n_estimators}_maxdepth{cfg.training.max_depth}_learningrate{cfg.training.learning_rate}_cls_report_testing.csv")
    
    
    return cls_training_report, xgb_model, attribute_statistics, label_names



def validation(cfg, model, attribute_statistics, label_names, logger):
    logger.info("::: Validation :::")
    output_dir = Path(cfg.cls_3d.output_dir)
    
    # Initialize list to store validation reports
    cls_validation_report_dfs = []

    # Evaluate the model on the validation data
    for i, file_path in enumerate((output_dir / cfg.cls_3d.training.validation_data_dir).glob('*.gz')):
        
        logger.info(":: Prediction on the validation data ::")
        logger.info(f"Validation on {file_path}")
        
        # Load validation data
        pcd_attributes_file = gzip.GzipFile(file_path, "r")
        pcd_attributes = np.load(pcd_attributes_file)
        
        # Create a DataFrame from the attributes
        validation_data_subsampled = pd.DataFrame(pcd_attributes, columns=attribute_statistics)
        filename = Path(file_path).stem
        
        # Map label names
        validation_data_subsampled['label_names'] = validation_data_subsampled['label'].map(label_names)
        
        # Get test data
        x_validation = validation_data_subsampled.drop(columns=['x', 'y', 'z', 'label', 'label_names'])
        y_validation = validation_data_subsampled['label'] 
        
        # Make predictions
        y_pred = prediction(model, x_validation, logger)
        
        # Evaluate model
        cnf_matrix_validation, cls_report_validation = evaluate_model(y_pred=y_pred, 
                                                                      y_test=y_validation,
                                                                      logger=logger)
        
        # Write cls_report_fr to a pandas dataframe 
        cls_validation_report_df = pd.DataFrame(cls_report_validation).transpose()
        cls_validation_report_df['n_estimators'] = cfg.training.n_estimators
        cls_validation_report_df['max_depth'] = cfg.training.max_depth
        cls_validation_report_df['learning_rate'] = cfg.training.learning_rate
        cls_validation_report_df['filename'] = filename
        cls_validation_report_df['fold'] = i+1
        cls_validation_report_df['id'] = cfg.training.id
    
        # Save the confusion matrix
        cnf_matrix_path = output_dir / "confusion_matrices_validation"
        cnf_matrix_path.mkdir(parents=True, exist_ok=True)
        np.savetxt(cnf_matrix_path / f"id{cfg.training.id}__{filename}__nestimators{cfg.training.n_estimators}_maxdepth{cfg.training.max_depth}_learningrate{cfg.training.learning_rate}_confusion_matrix_validation.csv", 
                    cnf_matrix_validation, delimiter=',', fmt='%u')
        
        # Save x,y,z labels and predicted labels as numpy array
        save_classified_pcd = output_dir / "classified_pcd" / "validation"
        save_classified_pcd.mkdir(parents=True, exist_ok=True)
        pcd_out = np.c_[validation_data_subsampled['x'], validation_data_subsampled['y'], validation_data_subsampled['z'], y_validation, y_pred]
        
        logger.info(f"Saving classified point cloud to: {save_classified_pcd}")
        np.savetxt(save_classified_pcd / f"{filename}_xyz_labels_predicted_labels.txt", pcd_out, delimiter=' ', fmt='%1.4f %1.4f %1.4f %u %u')
        
        # Append the cls_report_fr_df to the list
        cls_validation_report_dfs.append(cls_validation_report_df)
    
    
    # Concatenate the cls_report_fr_dfs
    cls_validation_report_df_out = pd.concat(cls_validation_report_dfs)
    
    # Save the files
    classification_report_dir = output_dir / "classification_report_validation"
    classification_report_dir.mkdir(parents=False, exist_ok=True)
    cls_validation_report_df_out.to_csv(classification_report_dir / f"id{cfg.training.id}_nestimators{cfg.training.n_estimators}_maxdepth{cfg.training.max_depth}_learningrate{cfg.training.learning_rate}_cls_report_validation.csv")
    
    return cls_validation_report_df_out


def write_to_csv(df, path):
    """Write a DataFrame to a CSV file, appending if the file already exists."""
    mode = 'a' if path.exists() else 'w'
    header = not path.exists()
    df.to_csv(path, mode=mode, header=header, index=True)



@hydra.main(version_base=None, config_path="../../../config", config_name="main")
def main(cfg: DictConfig):
    # Clear the hydra config cache
    GlobalHydra.instance().clear()
    output_dir = Path(cfg.cls_3d.output_dir)

    # Set up the logger
    logger = lgr.logger_setup('xgb_training', output_dir / "logs/xgb_training.log")

    # Record the start time
    time_start_main = time.time()

    # Run the training and validation
    cls_report_fr_dfs, model, attribute_statistics, label_names = training(cfg, logger)
    cls_validation_report_df_out = validation(cfg, model, attribute_statistics, label_names, logger)

    # Prepare the output directory
    all_results_dir = output_dir / "all_results"
    all_results_dir.mkdir(parents=True, exist_ok=True)

    # Define the output file paths
    cls_testing_results_path = all_results_dir / "testing_cls_results.csv"
    cls_validation_results_path = all_results_dir / "validation_cls_results.csv"

    # Write the testing and validation results to CSV files
    write_to_csv(pd.DataFrame(cls_report_fr_dfs), cls_testing_results_path)
    write_to_csv(pd.DataFrame(cls_validation_report_df_out), cls_validation_results_path)

    # Record the end time and calculate the total time
    time_end_main = time.time()
    total_time = time_end_main - time_start_main

    # Log the total time
    logger.info(f">>> Total time for the parameter test: {total_time:.2f}s ({total_time / 60:.2f} minutes) <<<")

    
if __name__=="__main__":
    main()