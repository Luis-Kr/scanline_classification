import subprocess
from pathlib import Path
from itertools import product

pcd_dir = Path("/DATA/Luis/thesis/scanline_classification/data/03_labeled/SiteD_RHV_01_Labeled.txt")
dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/final_results/4_performance_test/cls_3d")
rf_model_path = Path("/DATA/Luis/thesis/scanline_classification/models/xgb_training/model_60_features/xgb_model.joblib")

nghb_list = [0.5]
voxel_size_list = [0.2, 0.1, 0.05, 0.03]
 
# Generate all combinations
combinations = list(product(nghb_list, voxel_size_list))

# Print the combinations
for i, (nghb, voxel_size) in enumerate(combinations):
    
    command = (
    f"python scanline_classification/classification/3d_classification/3D_pointcloud_classification_main.py "
    f"cls_3d.input_file_path={pcd_dir} "
    f"cls_3d.output_dir={dst_dir} "
    f"cls_3d.model_path={rf_model_path} "
    f"cls_3d.nghb_search_radius={nghb} "
    f"cls_3d.voxel_size={voxel_size} "
    )
    
    # Run the command
    subprocess.run(command, shell=True)