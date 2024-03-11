import subprocess
from itertools import product

training_data = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/training/merged/training_data_merged_frac.csv"
testing_data = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/validation/merged/validation_data_merged_frac.csv"
output_dir_coarse = "/DATA/Luis/thesis/scanline_classification/models/xgb_training/feature_importance"
output_dir_fine = "/DATA/Luis/thesis/scanline_classification/models/xgb_training/feature_importance_fine"
output_dir_stats = "/DATA/Luis/thesis/scanline_classification/models/xgb_training/feature_importance_stats"

coarse=False
statistics_feat_importance=True

statistics = ["mean","var","std","median","perc2nd","perc98th","perc25th","perc75th","skewness"]
drop_cols_coarse = ["none","z","intensity","red","green","blue","rho","vert_angle","slope","curvature","roughness","nx_xyz","ny_xyz","nz_xyz","nx","ny","nz"]
drop_cols_fine = ['none','intensity','red','green','vert_angle','slope','curvature','roughness','nz']

statistics_feature_importance = ["all","mean","var","std","median","perc2nd","perc98th","perc25th","perc75th","skewness"]

n_estimators = 200
max_depth = 4
learning_rate = 0.1 

if coarse:
    drop_cols = drop_cols_coarse
    output_dir = output_dir_coarse
else:
    drop_cols = drop_cols_fine
    output_dir = output_dir_fine
    
if statistics_feat_importance:
    drop_cols = drop_cols_fine

# Print the combinations
if not statistics_feat_importance:
    for i, stat in enumerate(statistics):
        for v, drop_col in enumerate(drop_cols):
            print("---------------------------------------------------")
            print(f'Drop col: {drop_col} --- Statistic: {stat} --- Iteration: {i+1} --- {v+1} of {len(drop_cols)}')
            print("---------------------------------------------------")
            
            print(f"drops: {drop_cols}")
                        
            command = (
            f"python scanline_classification/classification/05_01_xgboost_training_feat_import.py "
            f"training.training_data_path={training_data} "
            f"training.testing_data_path={testing_data} "
            f"training.output_dir={output_dir} "
            f"training.n_estimators={n_estimators} "
            f"training.max_depth={max_depth} "
            f"training.learning_rate={learning_rate} "
            f"training.drop_col={drop_col} "
            f"training.cols_to_consider='{drop_cols}' "
            f"training.statistics={stat} "
            f"training.coarse={coarse} "
            )
            
            # Run the command
            subprocess.run(command, shell=True)
else:
    for i, stat in enumerate(statistics_feature_importance):
        print("---------------------------------------------------")
        print(f'Statistic: {stat} --- Iteration: {i+1} of {len(statistics_feature_importance)}')
        print("---------------------------------------------------")
        
        if i == 0:
            drop_col = "none"
        else:
            drop_col = "skip"
            
        output_dir = output_dir_stats
            
        command = (
        f"python scanline_classification/classification/05_01_xgboost_training_feat_import.py "
        f"training.training_data_path={training_data} "
        f"training.testing_data_path={testing_data} "
        f"training.output_dir={output_dir} "
        f"training.n_estimators={n_estimators} "
        f"training.max_depth={max_depth} "
        f"training.learning_rate={learning_rate} "
        f"training.drop_col={drop_col} "
        f"training.cols_to_consider='{drop_cols}' "
        f"training.statistics={stat} "
        f"training.statistics_feat_importance={statistics_feat_importance} "
        f"training.coarse={coarse} "
        )
        
        # Run the command
        subprocess.run(command, shell=True)


