import subprocess
from pathlib import Path

input_files_dir= "/Users/luiskremer/Code/Uni/Code_Master_RSIV/019_scanline_segmentation/data/03_labeled/SiteA_Scans_Global_I_RGB_RHV"
output_dir= "/Users/luiskremer/Code/Uni/Code_Master_RSIV/019_scanline_segmentation/data/07_classification/3d_classification" #radius25cm_voxel2cm

nghb_search_radius= 0.5
voxel_size= 0.1

# Print the combinations
for i, file in enumerate(Path(input_files_dir).rglob('*.txt')):
    print("---------------------------------------------------")
    print(f'Processing file: {file} --- {i+1}/{len(list(Path(input_files_dir).rglob("*.txt")))}')
    print("---------------------------------------------------")

    command = (
    f"python scanline_classification/classification/3d_classification/3D_pointcloud_classification_main.py "
    f"cls_3d.input_file_path={file} "
    f"cls_3d.output_dir={output_dir} "
    f"cls_3d.nghb_search_radius={nghb_search_radius} "
    f"cls_3d.voxel_size={voxel_size} "
    )
    
    # Run the command
    subprocess.run(command, shell=True)