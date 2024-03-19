import subprocess
from itertools import product

training_data_path = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/training/merged/training_data_merged_frac_test.csv"
training_data_subsampled_dir = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/training"
validation_data_dir = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/validation"
segmentation_data_dir = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/scanline_segmentation"
attribute_statistics_path = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/scanline_subsampling/attribute_statistics/SiteA_RHV_01_Labeled.pkl"
#output_dir = "/DATA/Luis/thesis/scanline_classification/data/final_results/1_best_overall_performance"
output_dir = "/DATA/Luis/thesis/scanline_classification/data/final_results/2_hyperparameter_testing/fine"

n_estimators = [1, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400]
max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9]  
learning_rate = [0.3] 

condsider_all_features_and_stats = False

# Generate all combinations
combinations = list(product(n_estimators, max_depth, learning_rate))

# Print the combinations
for i, (n_estimators, max_depth, learning_rate) in enumerate(combinations):
    print("---------------------------------------------------")
    print(f'Combination --- n_estimators:{n_estimators} | max_depth:{max_depth} | learning_rate:{learning_rate} --- {i+1}/{len(combinations)}')
    print("---------------------------------------------------")
                
    command = (
    f"python scanline_classification/classification/07_01_hyperparameter_tuning_full_resolution_validation.py "
    f"training.training_data_path={training_data_path} "
    f"training.validation_data_dir={validation_data_dir} "
    f"training.segmentation_data_dir={segmentation_data_dir} "
    f"training.attribute_statistics_path={attribute_statistics_path} "
    f"training.training_data_subsampled_dir={training_data_subsampled_dir} "
    f"training.output_dir={output_dir} "
    f"training.n_estimators={n_estimators} "
    f"training.max_depth={max_depth} "
    f"training.learning_rate={learning_rate} "
    f"training.id={i} "
    f"attribute.condsider_all_features_and_stats={condsider_all_features_and_stats} "
    )
    
    # Run the command
    subprocess.run(command, shell=True)