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
from scipy.spatial import cKDTree

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

import utils.logger as lgr


def evaluate_model(y_pred, y_test, logger):
    logger.info("Evaluating the model...")
    
    # Get the confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    # Get the classification report
    cls_report = classification_report(y_test, y_pred, digits=3, target_names=np.array(["unclassified", "man-made objects", "ground", "tree trunk/branches", "leaves", "low vegetation"]), output_dict=True)

    return cnf_matrix, cls_report


def subsample_pcd(original_pcd: np.ndarray,
                  subsampled_pcd: np.ndarray) -> np.ndarray:
    # Create a KDTree
    tree = cKDTree(original_pcd[:, :3])

    # Query both point clouds
    _, indices = tree.query(subsampled_pcd[:, :3], k=1, workers=6)

    # Filter pcd based on the indices
    original_pcd_subsampled = original_pcd[indices, :]
    
    return original_pcd_subsampled



def evaluate_testing_results(cfg, logger):
    output_dir = Path(cfg.cls_3d.output_dir)
    
    testing_results_dir_3d = output_dir / cfg.cls_3d.evaluation.testing_3d
    testing_results_dir_2d = Path(cfg.cls_3d.evaluation.testing_2d) 
    
    cls_training_report_dfs = []

    for i, test_files in enumerate(zip(sorted(testing_results_dir_3d.glob('*.txt')), sorted(testing_results_dir_2d.glob('*.txt')))):
        logger.info(f"::: Fold {i+1} ::::")
        
        # Get testing file names
        test_file_3d_path = test_files[0]
        test_file_2d_fr_path = test_files[1]
        filename = '_'.join(str(test_files[0]).split('/')[-1].split('_')[0:3])
        
        logger.info(f"Import the data: {str(test_file_3d_path)}")
        
        # Load the testing data
        test_file_3d = np.loadtxt(test_file_3d_path, delimiter=' ')
        test_file_fr_2d = np.loadtxt(test_file_2d_fr_path, delimiter=' ')
        
        # Subsample the point cloud containing the scanline classification result
        test_file_2d = subsample_pcd(test_file_fr_2d, test_file_3d)
        
        # Evaluate the 3D model
        cnf_matrix_3d, cls_report_3d = evaluate_model(y_pred=test_file_3d[:, -1], 
                                                      y_test=test_file_3d[:, -2],
                                                      logger=logger)
        
        # Evaluate the 2D model
        cnf_matrix_2d, cls_report_2d = evaluate_model(y_pred=test_file_2d[:, -1], 
                                                      y_test=test_file_2d[:, -2],
                                                      logger=logger)
        
        # Write the reports to a pandas dataframe
        cls_report_3d_df = pd.DataFrame(cls_report_3d).transpose()
        cls_report_3d_df['filename'] = filename
        cls_report_3d_df['id'] = "3d"
        
        cls_report_2d_df = pd.DataFrame(cls_report_2d).transpose()
        cls_report_2d_df['filename'] = filename
        cls_report_2d_df['id'] = "2d"
        
        # Save the confusion matrix
        cnf_matrix_path = output_dir / cfg.cls_3d.evaluation.output_dir / "confusion_matrices_testing"
        cnf_matrix_path.mkdir(parents=True, exist_ok=True)
        np.savetxt(cnf_matrix_path / f"{filename}_3d_confusion_matrix_testing.csv", 
                   cnf_matrix_3d, delimiter=',', fmt='%u')
        np.savetxt(cnf_matrix_path / f"{filename}_2d_confusion_matrix_testing.csv", 
                   cnf_matrix_2d, delimiter=',', fmt='%u')
        
        cls_training_report_dfs.append(cls_report_3d_df)
        cls_training_report_dfs.append(cls_report_2d_df)
    
    # Concatenate the cls_report_fr_dfs
    cls_training_report = pd.concat(cls_training_report_dfs)
    
    # Save the files
    classification_report_dir = output_dir / cfg.cls_3d.evaluation.output_dir / "classification_report_testing"
    classification_report_dir.mkdir(parents=False, exist_ok=True)
    cls_training_report.to_csv(classification_report_dir / f"cls_report_testing_3d_2d.csv")



def evaluate_validation_results(cfg, logger):
    output_dir = Path(cfg.cls_3d.output_dir)
    
    validation_results_dir_3d = output_dir / cfg.cls_3d.evaluation.validation_3d
    validation_results_dir_2d = Path(cfg.cls_3d.evaluation.validation_2d)
    
    cls_validation_report_dfs = []
    
    for i, validation_files in enumerate(zip(sorted(validation_results_dir_3d.glob('*.txt')), sorted(validation_results_dir_2d.glob('*.txt')))):
        logger.info(f"::: Fold {i+1} ::::")
        
        # Get validation file names
        validation_file_3d_path = validation_files[0]
        validation_file_2d_fr_path = validation_files[1]
        filename = '_'.join(str(validation_files[0]).split('/')[-1].split('_')[0:3])
        
        logger.info(f"Import the data: {str(validation_file_3d_path)}")
        
        # Load the validation data
        validation_file_3d = np.loadtxt(validation_file_3d_path, delimiter=' ')
        validation_file_fr_2d = np.loadtxt(validation_file_2d_fr_path, delimiter=' ')
        
        # Subsample the point cloud containing the scanline classification result
        validation_file_2d = subsample_pcd(validation_file_fr_2d, validation_file_3d)
        
        # Evaluate the 3D model
        cnf_matrix_3d, cls_report_3d = evaluate_model(y_pred=validation_file_3d[:, -1], 
                                                      y_test=validation_file_3d[:, -2],
                                                      logger=logger)
        
        # Evaluate the 2D model
        cnf_matrix_2d, cls_report_2d = evaluate_model(y_pred=validation_file_2d[:, -1], 
                                                      y_test=validation_file_2d[:, -2],
                                                      logger=logger)
        
        # Write the reports to a pandas dataframe
        cls_report_3d_df = pd.DataFrame(cls_report_3d).transpose()
        cls_report_3d_df['filename'] = filename
        cls_report_3d_df['id'] = "3d"
        
        cls_report_2d_df = pd.DataFrame(cls_report_2d).transpose()
        cls_report_2d_df['filename'] = filename
        cls_report_2d_df['id'] = "2d"
        
        # Save the confusion matrix
        cnf_matrix_path = output_dir / cfg.cls_3d.evaluation.output_dir / "confusion_matrices_validation"
        cnf_matrix_path.mkdir(parents=True, exist_ok=True)
        np.savetxt(cnf_matrix_path / f"{filename}_3d_confusion_matrix_validation.csv", cnf_matrix_3d, delimiter=',', fmt='%u')
        np.savetxt(cnf_matrix_path / f"{filename}_2d_confusion_matrix_validation.csv", cnf_matrix_2d, delimiter=',', fmt='%u')
        
        cls_validation_report_dfs.append(cls_report_3d_df)
        cls_validation_report_dfs.append(cls_report_2d_df)
    
    # Concatenate the cls_report_fr_dfs
    cls_validation_report = pd.concat(cls_validation_report_dfs)
    
    # Save the files
    classification_report_dir = output_dir / cfg.cls_3d.evaluation.output_dir / "classification_report_validation"
    classification_report_dir.mkdir(parents=False, exist_ok=True)
    cls_validation_report.to_csv(classification_report_dir / f"cls_report_validation_3d_2d.csv")



@hydra.main(version_base=None, config_path="../../../config", config_name="main")
def main(cfg: DictConfig):
    # Clear the hydra config cache
    GlobalHydra.instance().clear()

    # Set up the logger
    logger = lgr.logger_setup('xgb_training', Path(cfg.cls_3d.output_dir) / "logs/cls_comparison_3d_2d.log")

    # Record the start time
    time_start_main = time.time()
    
    # Generate output dir
    output_dir = Path(cfg.cls_3d.evaluation.output_dir)
    output_dir.mkdir(parents=False, exist_ok=True)

    # Run the training and validation
    evaluate_testing_results(cfg, logger)
    evaluate_validation_results(cfg, logger)

    # Record the end time and calculate the total time
    time_end_main = time.time()
    total_time = time_end_main - time_start_main

    # Log the total time
    logger.info(f">>> Total time for the parameter test: {total_time:.2f}s ({total_time / 60:.2f} minutes) <<<")

    
if __name__=="__main__":
    main()