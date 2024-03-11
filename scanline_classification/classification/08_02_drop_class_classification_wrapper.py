import subprocess
from itertools import product

training_data_path = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/training/merged/training_data_merged_frac_test.csv"
training_data_subsampled_dir = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/training"
validation_data_dir = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/validation"
segmentation_data_dir = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/scanline_segmentation"
attribute_statistics_path = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/scanline_subsampling/attribute_statistics/SiteA_RHV_01_Labeled.pkl"
output_dir = "/DATA/Luis/thesis/scanline_classification/data/final_results/3_pairwise_classification"

n_estimators = 100
max_depth = 3
learning_rate = 0.3

# Generate all combinations
classes_to_drop = [0, 1, 2, 3, 4, 5]

# Print the combinations
for i, class_to_drop in enumerate(classes_to_drop):
    print("---------------------------------------------------")
    print(f'Class that will be dropped: {class_to_drop} --- {i+1}/{len(classes_to_drop)}')
    print("---------------------------------------------------")
                
    command = (
    f"python scanline_classification/classification/08_01_drop_class_classification.py "
    f"training.training_data_path={training_data_path} "
    f"training.validation_data_dir={validation_data_dir} "
    f"training.segmentation_data_dir={segmentation_data_dir} "
    f"training.attribute_statistics_path={attribute_statistics_path} "
    f"training.training_data_subsampled_dir={training_data_subsampled_dir} "
    f"training.output_dir={output_dir} "
    f"training.n_estimators={n_estimators} "
    f"training.max_depth={max_depth} "
    f"training.learning_rate={learning_rate} "
    f"training.class_to_drop={class_to_drop} "
    f"training.id={i} "
    )
    
    # Run the command
    subprocess.run(command, shell=True)