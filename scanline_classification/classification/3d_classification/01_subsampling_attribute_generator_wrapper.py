import subprocess
from pathlib import Path

input_files_dir= "/DATA/Luis/thesis/scanline_classification/data/03_labeled"
output_dir= "/DATA/Luis/thesis/scanline_classification/data/final_results/5_comparison_3d_scanline_approach/cls_3d"

nghb_search_radius=0.5
voxel_size=0.05

output_dir = output_dir + "/" + f"radius{nghb_search_radius}_voxel{voxel_size}"
print(f"Output directory: {output_dir}")

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