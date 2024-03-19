import subprocess
from pathlib import Path

pcd_dir = Path("/DATA/Luis/thesis/scanline_classification/data/03_labeled")
dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02")

# Print the combinations
for i, path in enumerate(pcd_dir.glob("*.txt")):
                
    command = (
    f"python scanline_classification/scanline_classification_main.py "
    f"pcd_path={path} "
    f"dst_dir={dst_dir} "
    )
    
    # Run the command
    subprocess.run(command, shell=True)