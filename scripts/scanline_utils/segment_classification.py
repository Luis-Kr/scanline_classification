from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
from numba import njit, prange
import numba
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig

def evaluate_classifier(rf_model, features, labels):
    predicted_labels = rf_model.predict(features)
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels, average='weighted')
    recall = recall_score(labels, predicted_labels, average='weighted')
    f1 = f1_score(labels, predicted_labels, average='weighted')
    cnf_matrix = confusion_matrix(labels, predicted_labels)
    
    return predicted_labels, accuracy, precision, recall, f1, cnf_matrix


def segment_classification(cfg: DictConfig,
                           pcd_subsampled: np.ndarray, 
                           model_filepath: str,
                           metrics_output_filepath: str, 
                           cnfmatrix_output_path: str,
                           pcd_subsampled_classified_path: str):    
    features = pcd_subsampled[:, 9:-2] # ignore the first 9 columns (position) and the last 2 columns (segment ids, labels)
    labels = pcd_subsampled[:, -1]
    
    rf_model = joblib.load(model_filepath)
    
    predicted_labels, acc, prec, rec, f1, cnf_matrix = evaluate_classifier(rf_model, features, labels)
    
    with metrics_output_filepath.open('a') as f:
        f.write("accuracy,precision,recall,f1,filename\n")
        f.write(f"{acc},{prec},{rec},{f1},{metrics_output_filepath.name}\n")
    
    np.savetxt(cnfmatrix_output_path, cnf_matrix, delimiter=',', fmt='%u')
    
    pcd_subsampled_classified = np.c_[pcd_subsampled[:, :3], pcd_subsampled[:, -1], predicted_labels]
    
    if cfg.output_compressed:
        np.savez_compressed(str(pcd_subsampled_classified_path) + "_subsampled_classified.npz", pcd_subsampled_classified)
    else:
        fmt = "%1.4f %1.4f %1.4f %u %u"
        np.savetxt(str(pcd_subsampled_classified_path) + "_subsampled_classified.txt", pcd_subsampled_classified, delimiter=' ', fmt=fmt)
    
    return predicted_labels


@njit(parallel=True)
def unfold_labels(pcd: np.ndarray, 
                  pcd_subs_predicted_labels: np.ndarray,
                  indices_per_class):
    predicted_labels = np.zeros(pcd.shape[0])

    for i in prange(pcd_subs_predicted_labels.shape[0]):
        segment_indices = indices_per_class[i]
        predicted_labels[segment_indices] = pcd_subs_predicted_labels[i]

    return predicted_labels

def assign_labels(pcd: np.ndarray, 
                  predicted_labels: np.ndarray):
    return np.c_[pcd[:, :12], predicted_labels]