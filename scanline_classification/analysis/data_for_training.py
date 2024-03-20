import subprocess
from pathlib import Path

pcd_dir = Path("/DATA/Luis/thesis/scanline_classification/data/03_labeled")
dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/final_results/4_performance_test/scanline_approach")
rf_model_path = Path("/DATA/Luis/thesis/scanline_classification/models/xgb_training/model_60_features/xgb_model.joblib")

# Print the combinations
for i, path in enumerate(pcd_dir.glob("*.txt")):
    
    if i == 1:
        break
    
    command = (
    f"python scanline_classification/scanline_classification_main.py "
    f"pcd_path={path} "
    f"dst_dir={dst_dir} "
    f"run_classification=True "
    f"paths.rf_model={rf_model_path} "
    f"sgcl.save_pcd=False "
    )
    
    # Run the command
    subprocess.run(command, shell=True)