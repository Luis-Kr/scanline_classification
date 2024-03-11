import subprocess

output_dir = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/feature_importance_stats"

columns = ["z","intensity","red","green","blue","rho","vert_angle","slope","curvature","roughness","nx_xyz","ny_xyz","nz_xyz","nx","ny","nz"]
statistics = ["none","mean","var","std","median","perc2nd","perc98th","perc25th","perc75th","skewness"]

n_estimators = 150
max_depth = 6
learning_rate = 0.3

# Print the combinations
for i, col in enumerate(columns):
    for v, drop_stat in enumerate(statistics):
        print("---------------------------------------------------")
        print(f'Statistics to drop: {drop_stat} --- Attribute: {col} --- Iteration: {i+1} --- {v+1} of {len(columns)}')
        print("---------------------------------------------------")
        
        print(f"Columns that will be dropped: {drop_stat}")
                  
        command = (
        f"python scanline_classification/classification/06_04_01_xgboost_full_resolution_feature_importance_statistics.py  "
        f"training.output_dir={output_dir} "
        f"training.n_estimators={n_estimators} "
        f"training.max_depth={max_depth} "
        f"training.learning_rate={learning_rate} "
        f"training.drop_col={drop_stat} "
        f"training.statistics={drop_stat} "
        f"training.col_to_consider={col} "
        )
        
        # Run the command
        subprocess.run(command, shell=True)