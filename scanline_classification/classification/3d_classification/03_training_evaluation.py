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

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent.absolute()))

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
    # Load the training data
    training_data = pd.read_csv(cfg.training.training_data_path, sep=",", header=0)
    
    # Get a list of all files in the directory
    files_siteA = list(Path(cfg.training.training_data_subsampled_dir).glob('SiteA*.txt')) 
    files_siteB = list(Path(cfg.training.training_data_subsampled_dir).glob('SiteB*.txt')) 

    # Shuffle the files
    np.random.seed(42)
    random.shuffle(files_siteA, random.seed(2))
    random.shuffle(files_siteB, random.seed(4))

    # Select random pairs of files
    selected_files = [[files_siteA[i], files_siteB[i]] for i in np.random.random_integers(0, 5, 3)]
    
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
        
        # Get attributes to consider
        if not cfg.attribute.condsider_all_features_and_stats:
            cols_to_consider = cfg.attribute.best_overall.cols_to_consider
            stats_to_consider = cfg.attribute.best_overall.stats_to_consider
            attributes_to_consider = [f"{i}_{j}" for i in cols_to_consider for j in stats_to_consider]
            
            # Get training data
            x_train = training_data_subset[attributes_to_consider]
            y_train = training_data_subset['label']   
        else:
            cols_to_consider = cfg.attribute.all.cols_to_consider
            stats_to_consider = cfg.attribute.all.stats_to_consider
            attributes_to_consider = [f"{i}_{j}" for i in cols_to_consider for j in stats_to_consider]
            
            # Get training data
            x_train = training_data_subset[attributes_to_consider]
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
        with open(cfg.training.attribute_statistics_path, 'rb') as f:
                attribute_statistics = pickle.load(f)
        
        # Test the model
        for file_path in selected_files[i]:
            logger.info(":: Prediction on the test data ::")
            logger.info(f"Testing on {file_path}")
            
            # Load test data
            test_data_subsampled = pd.read_csv(file_path, sep=" ", header=None, names=attribute_statistics)
            filename = Path(file_path).stem
            segmentation_pcds_dir = Path(cfg.training.segmentation_data_dir)
            file_to_search = "_".join(Path(filename).stem.split("_")[0:3])
            segmentation_pcd = segmentation_pcds_dir.glob(f"{file_to_search}*").__next__()
            
            logger.info(f'Loading full resolution point cloud: {segmentation_pcd}')
            
            # Load full resolution pcd
            pcd_fr = np.loadtxt(segmentation_pcd, delimiter=' ')
            
            # Map label names
            test_data_subsampled['label_names'] = test_data_subsampled['label'].map(label_names)
            
            # Get test data
            x_test = test_data_subsampled[attributes_to_consider]
            
            # Make predictions
            y_pred = prediction(xgb_model, x_test, logger)
            
            # Get indices per segment
            pcd_sorted, indices_per_segment = get_indices_per_segment(cfg, pcd_fr)
            
            # Unfold labels
            fr_predicted_labels = unfold_labels(pcd=pcd_sorted, pcd_subs_predicted_labels=y_pred, indices_per_segment=indices_per_segment)
            
            # Get validation data
            validation_pred = fr_predicted_labels
            validation_test = pcd_sorted[:, cfg.pcd_col.label]
            
            # Evaluate model
            cnf_matrix_fr, cls_report_fr = evaluate_model(y_pred=validation_pred, 
                                                          y_test=validation_test,
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
            cnf_matrix_path = Path(cfg.training.output_dir) / "confusion_matrices_testing"
            cnf_matrix_path.mkdir(parents=False, exist_ok=True)
            np.savetxt(cnf_matrix_path / f"id{cfg.training.id}__{filename}__nestimators{cfg.training.n_estimators}_maxdepth{cfg.training.max_depth}_learningrate{cfg.training.learning_rate}_confusion_matrix_testing.csv", 
                       cnf_matrix_fr, delimiter=',', fmt='%u')
            
            # Append the cls_report_fr_df to the list
            cls_training_report_dfs.append(cls_report_fr_df)
    
    # Concatenate the cls_report_fr_dfs
    cls_training_report = pd.concat(cls_training_report_dfs)
    
    # Save the files
    classification_report_dir = Path(cfg.training.output_dir) / "classification_report_testing"
    classification_report_dir.mkdir(parents=False, exist_ok=True)
    cls_training_report.to_csv(classification_report_dir / f"id{cfg.training.id}_nestimators{cfg.training.n_estimators}_maxdepth{cfg.training.max_depth}_learningrate{cfg.training.learning_rate}_cls_report_testing.csv")
    
    
    return cls_training_report, xgb_model, attributes_to_consider, label_names



def validation(cfg, model, attributes_to_consider, label_names, logger):
    logger.info("::: Validation :::")
    
    # Initialize list to store validation reports
    cls_validation_report_dfs = []
    
    # Load attribute statistics
    with open(cfg.training.attribute_statistics_path, 'rb') as f:
            attribute_statistics = pickle.load(f)


    # Evaluate the model on the validation data
    for i, file_path in enumerate(Path(cfg.training.validation_data_dir).glob('*.txt')):
        
        logger.info(":: Prediction on the validation data ::")
        logger.info(f"Validation data shape: {pd.read_csv(file_path, sep=' ', header=None, names=attribute_statistics).shape}")
        logger.info(f"Validation on {file_path}")
        
        # Load validation data
        validation_data_subsampled = pd.read_csv(file_path, sep=" ", header=None, names=attribute_statistics)
        filename = Path(file_path).stem
        segmentation_pcds_dir = Path(cfg.training.segmentation_data_dir)
        file_to_search = "_".join(Path(filename).stem.split("_")[0:3])
        segmentation_pcd = segmentation_pcds_dir.glob(f"{file_to_search}*").__next__()
        
        logger.info(f'Loading full resolution point cloud: {segmentation_pcd}')
        pcd_fr = np.loadtxt(segmentation_pcd, delimiter=' ')
        
        # Map label names
        validation_data_subsampled['label_names'] = validation_data_subsampled['label'].map(label_names)
        
        # Get test data
        x_validation = validation_data_subsampled[attributes_to_consider]
        
        # Make predictions
        y_pred = prediction(model, x_validation, logger)
        
        # Get indices per segment
        pcd_sorted, indices_per_segment = get_indices_per_segment(cfg, pcd_fr)
        
        # Unfold labels
        fr_predicted_labels = unfold_labels(pcd=pcd_sorted, pcd_subs_predicted_labels=y_pred, indices_per_segment=indices_per_segment)
        
        # Get validation data
        validation_pred = fr_predicted_labels
        validation_test = pcd_sorted[:, cfg.pcd_col.label]
        
        # Evaluate model
        cnf_matrix_validation, cls_report_validation = evaluate_model(y_pred=validation_pred, 
                                                                    y_test=validation_test,
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
        cnf_matrix_path = Path(cfg.training.output_dir) / "confusion_matrices_validation"
        cnf_matrix_path.mkdir(parents=False, exist_ok=True)
        np.savetxt(cnf_matrix_path / f"id{cfg.training.id}__{filename}__nestimators{cfg.training.n_estimators}_maxdepth{cfg.training.max_depth}_learningrate{cfg.training.learning_rate}_confusion_matrix_validation.csv", 
                    cnf_matrix_validation, delimiter=',', fmt='%u')
        
        
        # Append the cls_report_fr_df to the list
        cls_validation_report_dfs.append(cls_validation_report_df)
    
    
    # Concatenate the cls_report_fr_dfs
    cls_validation_report_df_out = pd.concat(cls_validation_report_dfs)
    
    # Save the files
    classification_report_dir = Path(cfg.training.output_dir) / "classification_report_validation"
    classification_report_dir.mkdir(parents=False, exist_ok=True)
    cls_validation_report_df_out.to_csv(classification_report_dir / f"id{cfg.training.id}_nestimators{cfg.training.n_estimators}_maxdepth{cfg.training.max_depth}_learningrate{cfg.training.learning_rate}_cls_report_validation.csv")
    
    return cls_validation_report_df_out


def write_to_csv(df, path):
    """Write a DataFrame to a CSV file, appending if the file already exists."""
    mode = 'a' if path.exists() else 'w'
    header = not path.exists()
    df.to_csv(path, mode=mode, header=header, index=True)



@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig):
    # Clear the hydra config cache
    GlobalHydra.instance().clear()

    # Set up the logger
    logger = lgr.logger_setup('xgb_training', Path(cfg.training.output_dir) / "logs/xgb_training.log")

    # Record the start time
    time_start_main = time.time()

    # Run the training and validation
    cls_report_fr_dfs, model, attributes_to_consider, label_names = training(cfg, logger)
    cls_validation_report_df_out = validation(cfg, model, attributes_to_consider, label_names, logger)

    # Prepare the output directory
    all_results_dir = Path(cfg.training.output_dir) / "all_results"
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