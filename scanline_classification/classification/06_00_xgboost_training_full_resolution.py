import logging
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
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
from numba import njit, prange

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


def train_model(x_train, y_train, logger, n_estimators, max_depth, learning_rate, sample_weights):
    # Set up XGBoost classifier
    logger.info("Setting up XGBoost classifier...")
    
    # Create the model
    xgb = XGBClassifier(n_estimators=n_estimators, 
                        max_depth=max_depth,
                        gamma=0.5,
                        learning_rate=learning_rate,
                        sample_weight=sample_weights,
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
    cnf_matrix = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, digits=3, target_names=np.array(["unclassified", "man-made objects", "ground", "tree trunk/branches", "leaves", "low vegetation"]), output_dict=True)

    return cnf_matrix, cls_report, y_pred


def evaluate_model_fr(y_pred, y_test, logger):
    # Evaluate the model
    logger.info("Evaluating the model...")
    
    cnf_matrix = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, digits=3, target_names=np.array(["unclassified", "man-made objects", "ground", "tree trunk/branches", "leaves", "low vegetation"]), output_dict=True)

    return cnf_matrix, cls_report
    

def split_data_into_features_labels(cfg, training_data, testing_data, output_dir, logger, attributes_to_consider):
    # Split the data into features and labels
    logger.info("Splitting the data into features and labels...")

    # Training data
    if cfg.attribute.condsider_all_features_and_stats:
        x_train = training_data.drop(columns=['z_median_nn', 'z_perc2nd_nn', 'z_perc98th_nn', "label", "label_names"])
    else:
        x_train = training_data[attributes_to_consider]
    
    y_train = training_data['label']
    
    _, sample_weights = create_class_weights(y_train)
    
    # Testing_data
    if cfg.attribute.condsider_all_features_and_stats:
        x_test = testing_data.drop(columns=['z_median_nn', 'z_perc2nd_nn', 'z_perc98th_nn', "label", "label_names"])
    else:
        x_test = testing_data[attributes_to_consider]
        
    y_test = testing_data['label']

    # # Save the column names 
    # column_names_path = Path(output_dir) / "column_names.pkl"
    # with open(column_names_path, 'wb') as f:
    #     pickle.dump(x_train.columns, f)
        
    # column_names_path = Path(output_dir) / "column_names.json"
    # with open(column_names_path, 'w') as f:
    #     for item in list(x_train.columns):
    #         f.write(json.dumps(item) + '\n')
        
    print(f"Column shape {x_train.columns.shape}")
    print(f"Column names {x_train.columns}")
    
    return x_train, x_test, y_train, y_test, sample_weights


def validation_data_evaluation(cfg: DictConfig,
                               file_paths_dir: str,
                               segmentation_pcds_dir: str,
                               attribute_statistics_path: str,
                               xgb_model,
                               x_test_frac,
                               y_test_frac,
                               output_dir,
                               attributes_to_consider,
                               logger):
    logger.info("Preparing the data for validation...")
    
    with open(attribute_statistics_path, 'rb') as f:
        attribute_statistics = pickle.load(f)
        
    drop_cols = ["x_median_nn", "y_median_nn", "x_perc2nd_nn", "y_perc2nd_nn", "x_perc98th_nn", "y_perc98th_nn", "segment_id"]
    
    cls_report_subs_frac_dfs = []
    cls_report_subs_dfs = []
    cls_report_fr_dfs = []
    
    for file_path in Path(file_paths_dir).glob('*.txt'):
        logger.info(f'Loading {file_path}')
        data = pd.read_csv(file_path, delimiter=' ', header=None, names=attribute_statistics)
        filename = Path(file_path).stem
        segmentation_pcds_dir = Path(segmentation_pcds_dir)
        file_to_search = "_".join(Path(filename).stem.split("_")[0:3])
        segmentation_pcd = segmentation_pcds_dir.glob(f"{file_to_search}*").__next__()
        
        logger.info(f'Loading {segmentation_pcd}')
        pcd_fr = np.loadtxt(segmentation_pcd, delimiter=' ')
        
        label_names={0: "unclassified",
                     1: "man-made objects",
                     2: "ground",
                     3: "tree trunk/branches",
                     4: "leaves",
                     5: "low vegetation"}
        
        logger.info(f"Shape of the input data: {data.shape}")
        
        # # Remove rows where label is 0 (unclassified)
        # data = data[data['label'] != 0]
        
        # # Decrease the label values by 1
        # data['label'] = data['label'] - 1
        
        data['label_names'] = data['label'].map(label_names)
        class_weights, _ = create_class_weights(data['label'])
        validation_data = data.drop(columns=drop_cols)
        
        print(validation_data['label_names'].value_counts())
        print(f"Class weights: {class_weights}")
        
        # Validation data
        logger.info(f"Validation data shape: {validation_data.shape}")
        if cfg.attribute.condsider_all_features_and_stats:
            x_test_subsampled = validation_data.drop(columns=['z_median_nn', 'z_perc2nd_nn', 'z_perc98th_nn',"label", "label_names"])
        else:
            x_test_subsampled = validation_data[attributes_to_consider]
        
        y_test_subsampled = validation_data['label']
        
        # Evaluate the model for the samples from the subsampled validation data
        logger.info("Evaluating the model for the samples from the subsampled validation data...")
        cnf_matrix_subs_frac, cls_report_subs_frac, _ = evaluate_model(model=xgb_model, 
                                                                        x_test=x_test_frac, 
                                                                        y_test=y_test_frac, 
                                                                        logger=logger)
        
        cls_report_subs_frac_df = pd.DataFrame(cls_report_subs_frac).transpose()
        cls_report_subs_frac_df['filename'] = filename
        classification_report_dir = Path(cfg.training.output_dir) / "classification_report"
        classification_report_dir.mkdir(parents=False, exist_ok=True)
        cls_report_subs_frac_df.to_csv(classification_report_dir / f"{filename}_cls_report_subs_frac.csv")
        
        cls_report_subs_frac_dfs.append(cls_report_subs_frac_df)
        
        # Evaluate the model for the subsampled validation data
        logger.info("Evaluating the model for the subsampled validation data...")
        cnf_matrix_subs, cls_report_subs, y_pred = evaluate_model(model=xgb_model, 
                                                                  x_test=x_test_subsampled, 
                                                                  y_test=y_test_subsampled, 
                                                                  logger=logger)
        
        # Write cls_report_subs to a pandas dataframe and save as csv
        cls_report_subs_df = pd.DataFrame(cls_report_subs).transpose()
        cls_report_subs_df['filename'] = filename
        cls_report_subs_df.to_csv(classification_report_dir / f"{filename}_cls_report_subs.csv")
        
        cls_report_subs_dfs.append(cls_report_subs_df)
        
        # Evaluate the model for the full resolution validation data
        logger.info("Evaluating the model for the full resolution validation data...")
        pcd_sorted, indices_per_segment = get_indices_per_segment(cfg, pcd_fr)
        fr_predicted_labels = unfold_labels(pcd=pcd_sorted, pcd_subs_predicted_labels=y_pred, indices_per_segment=indices_per_segment)
        fr_pcd_with_prediction = assign_labels(cfg=cfg, pcd=pcd_sorted, predicted_labels=fr_predicted_labels)
        
        yy_test = pcd_sorted[:, cfg.pcd_col.label]
        yy_pred = fr_predicted_labels
        
        cnf_matrix_fr, cls_report_fr = evaluate_model_fr(y_pred=yy_pred, 
                                                         y_test=yy_test,
                                                         logger=logger)
        
        # Write cls_report_fr to a pandas dataframe and save as csv
        cls_report_fr_df = pd.DataFrame(cls_report_fr).transpose()
        cls_report_fr_df['filename'] = filename
        cls_report_fr_df.to_csv(classification_report_dir / f"{filename}_cls_report_fr.csv")
        
        cls_report_fr_dfs.append(cls_report_fr_df)
        
        logger.info(f"Saving the results for {filename}...")
        
        cnf_matrix_path = Path(cfg.training.output_dir) / "confusion_matrices"
        cnf_matrix_path.mkdir(parents=False, exist_ok=True)
        
        # Save the confusion matrix for the sampled subsampled pcd
        np.savetxt(cnf_matrix_path / f"{filename}_confusion_matrix_sampled_subs_pcd.csv", cnf_matrix_subs_frac, delimiter=',', fmt='%u')
        np.savetxt(cnf_matrix_path / f"{filename}_confusion_matrix_subs_pcd.csv", cnf_matrix_subs, delimiter=',', fmt='%u')
        np.savetxt(cnf_matrix_path / f"{filename}_confusion_matrix_fr_pcd.csv", cnf_matrix_fr, delimiter=',', fmt='%u')
        
        # # Save the full resolution pcd with the predicted labels
        # pcd_with_prediction_path = Path(output_dir) / "pcd_with_prediction"
        # pcd_with_prediction_path.mkdir(parents=False, exist_ok=True)
        # np.savetxt(pcd_with_prediction_path / f"{filename}_pcd_with_prediction.txt", fr_pcd_with_prediction, delimiter=' ', fmt='%1.4f %1.4f %1.4f %u %u')

    return pd.concat(cls_report_subs_frac_dfs), pd.concat(cls_report_subs_dfs), pd.concat(cls_report_fr_dfs)


def create_class_weights(labels):
    unique_labels = np.unique(labels)
    class_weights = class_weight.compute_class_weight('balanced', 
                                                      classes=unique_labels, 
                                                      y=labels)
    sample_weights = class_weight.compute_sample_weight('balanced', labels)
    weights_dict = {class_label: weight for class_label, weight in zip(unique_labels, class_weights)}
    
    return weights_dict, sample_weights


# Get the number of points in each segment
def get_indices_per_segment(cfg, pcd):
    # Sort the pcd by vertical angle and segment id
    sort_idx = np.lexsort(np.rot90(pcd[:,(cfg.pcd_col.scanline_id, cfg.pcd_col.vert_angle)]))
    
    pcd_sorted = pcd[sort_idx]
    
    # # Remove labels with value 0 (unclassified)
    # pcd_sorted = pcd_sorted[pcd_sorted[:,cfg.pcd_col.label] != 0]
    
    # # Decrease the label values by 1
    # pcd_sorted[:,cfg.pcd_col.label] -= 1
    
    # Get the counts of the individual segment ids
    _, counts = np.unique(pcd_sorted[:,cfg.pcd_col.segment_ids], return_counts=True)
    
    # Sort the point cloud by segment id
    sorted_indices = np.argsort(pcd_sorted[:,cfg.pcd_col.segment_ids])
    
    # Split the sorted indices into segments
    indices_per_segment = np.split(sorted_indices, np.cumsum(counts[:-1]))
    
    return pcd_sorted, indices_per_segment


@njit(parallel=True)
def unfold_labels(pcd: np.ndarray, 
                  pcd_subs_predicted_labels: np.ndarray,
                  indices_per_segment):
    predicted_labels = np.zeros(pcd.shape[0])

    for i in prange(pcd_subs_predicted_labels.shape[0]):
        segment_indices = indices_per_segment[i]
        predicted_labels[segment_indices] = pcd_subs_predicted_labels[i]

    return predicted_labels


def assign_labels(cfg: DictConfig,
                  pcd: np.ndarray, 
                  predicted_labels: np.ndarray):
    return np.c_[pcd[:, :3], pcd[:, cfg.pcd_col.label], predicted_labels]


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
    testing_data = pd.read_csv(cfg.training.validation_file_frac, sep=",", header=0)
    
    # # Remove rows where label_name is unclassified
    # training_data = training_data[training_data['label'] != 0]
    # testing_data = testing_data[testing_data['label'] != 0]
    
    # # Decrease the label values by 1
    # training_data['label'] = training_data['label'] - 1
    # testing_data['label'] = testing_data['label'] - 1
    
    #---------------------------------------------------
    cols_to_consider = cfg.attribute.best_overall.cols_to_consider
    print(f"Columns to consider: {cols_to_consider}")
    stats_to_consider = cfg.attribute.best_overall.stats_to_consider
    attributes_to_consider = [f"{i}_{j}" for i in cols_to_consider for j in stats_to_consider]
    #---------------------------------------------------
 
    # Split the data into features and labels
    x_train, x_test, y_train, y_test, sample_weights = split_data_into_features_labels(cfg, training_data, testing_data, cfg.training.output_dir,logger, attributes_to_consider)
    
    # Training the model
    start_time = time.perf_counter()
    xgb_model = train_model(x_train, y_train, logger, cfg.training.n_estimators, cfg.training.max_depth, cfg.training.learning_rate, sample_weights)
    end_time = time.perf_counter()
    training_time = end_time - start_time
    logger.info(f"Training completed. It took {training_time:.2f} seconds.")
    
    # Evaluate the model
    #accuracy, f1, precision, recall = evaluate_model(xgb_model, x_test, y_test, logger)
    cls_report_subs_frac_dfs, cls_report_subs_dfs, cls_report_fr_dfs = validation_data_evaluation(cfg=cfg,
                                                                                                  file_paths_dir=cfg.training.validation_data_dir,
                                                                                                  segmentation_pcds_dir=cfg.training.segmentation_data_dir,
                                                                                                  attribute_statistics_path=cfg.training.attribute_statistics_path,
                                                                                                  xgb_model=xgb_model,
                                                                                                  x_test_frac=x_test,
                                                                                                  y_test_frac=y_test,
                                                                                                  output_dir=cfg.training.output_dir,
                                                                                                  attributes_to_consider=attributes_to_consider,
                                                                                                  logger=logger)
    
    # Save the model
    logger.info("Saving the model...")
    Path(cfg.training.output_dir).mkdir(parents=False, exist_ok=True)
    model_path = Path(cfg.training.output_dir_model) / "xgb_model.joblib"
    joblib.dump(xgb_model, model_path)
    
    # Save the evaluation results to a csv file
    logger.info("Saving the evaluation results...")
    all_results_dir = Path(cfg.training.output_dir) / "all_results"
    all_results_dir.mkdir(parents=False, exist_ok=True)
    cls_results_subs_frac_path = all_results_dir / "cls_results_subs_frac.csv"
    cls_results_subs_path = all_results_dir / "cls_results_subs.csv"
    cls_results_fr_path = all_results_dir / "cls_results_fr.csv"
    
    # Check if one of the files exists
    if cls_results_fr_path.exists():
        mode = 'a'  # append if file exists
        header = False  # don't write header if appending
    else:
        mode = 'w'  # write if file doesn't exist
        header = True  # write header if writing

    # Write or append to the files
    pd.DataFrame(cls_report_subs_frac_dfs).to_csv(cls_results_subs_frac_path, mode=mode, header=header, index=True)
    pd.DataFrame(cls_report_subs_dfs).to_csv(cls_results_subs_path, mode=mode, header=header, index=True)
    pd.DataFrame(cls_report_fr_dfs).to_csv(cls_results_fr_path, mode=mode, header=header, index=True)


if __name__=="__main__":
    main()
    
    

