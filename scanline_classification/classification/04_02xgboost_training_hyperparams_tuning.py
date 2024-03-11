import subprocess
from itertools import product

training_data = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/training/merged/training_data_merged_frac.csv"
testing_data = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/validation/merged/validation_data_merged_frac.csv"
output_dir = "/DATA/Luis/thesis/scanline_classification/models/xgb_training/hyperparameter_tuning"

n_estimators = [50, 100, 150, 200, 300, 400, 500]
max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
learning_rate = [1, 0.1, 0.01, 0.001, 0.0001] 

# Generate all combinations
combinations = list(product(n_estimators, max_depth, learning_rate))

# Print the combinations
for i, (n_estimators, max_depth, learning_rate) in enumerate(combinations):
    print("---------------------------------------------------")
    print(f'Combination --- n_estimators:{n_estimators} | max_depth:{max_depth} | learning_rate:{learning_rate} --- {i+1}/{len(combinations)}')
    print("---------------------------------------------------")
                
    command = (
    f"python scanline_classification/classification/xgboost_training_main.py "
    f"training.training_data_path={training_data} "
    f"training.testing_data_path={testing_data} "
    f"training.output_dir={output_dir} "
    f"training.n_estimators={n_estimators} "
    f"training.max_depth={max_depth} "
    f"training.learning_rate={learning_rate} "
    )
    
    # Run the command
    subprocess.run(command, shell=True)


