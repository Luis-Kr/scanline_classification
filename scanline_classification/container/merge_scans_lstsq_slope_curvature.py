import numpy as np
from pathlib import Path
import gzip

def extract_values(directory: str) -> np.ndarray:
    """
    Extracts the first three columns from each .npy file in a directory.

    Args:
        directory (str): The directory to search.

    Returns:
        np.ndarray: The extracted values.
    """
    # Create a Path object for the directory
    directory_path = Path(directory)
    
    array_init = False

    # Loop through the files in the directory
    for file_path in sorted(directory_path.iterdir()):
        if file_path.suffix == '.txt':
            # check if Path(directory_output) / 'slope_curvature_sphcoords.npy' is empty
            if not array_init:
                print(f'Loading file: {file_path}')
                print('Initialising array...')
                # Load the data from the file
                data = np.loadtxt(file_path, delimiter=' ')
                # Extract the first three columns
                extracted_values = data[:, [0, 1, 2, 7, 8, 9, 11, 14, 16, 17]]
                # Stack the extracted values column-wise in the new array
                values = extracted_values
                array_init = True
            else:
                print(f'Loading file: {file_path}')
                # Load the data from the file
                data = np.loadtxt(file_path, delimiter=' ')
                # Extract the first three columns
                extracted_values = data[:, [16, 17]]
                # Stack the extracted values column-wise in the new array
                values = np.hstack((values, extracted_values)) 

    return values

# Use the function
root_dir = Path(__file__).parent.parent.parent
directory_input = "data/05_segmentation/SiteD_Scans_Global_I_RGB_RHV"  
directory_output = "data/meetings/slope_curvature_lstsq_sphcoords"  
values = extract_values(directory_input)

print(f'values.shape: {values.shape}')

# Scanline extraction
scanlines = [1, 100, 1000, 2000, 3000, 4000]
selected_values = values[np.isin(values[:, 7], scanlines)]

print(f'selected_values.shape: {selected_values.shape}')

# Save the array
f = gzip.GzipFile(Path(directory_output) / 'slope_curvature_sphcoords.npy.gz', "w")
np.save(file=f, arr=values)
f.close()

f = gzip.GzipFile(Path(directory_output) / 'scanline_subset_slope_curvature_sphcoords.npy.gz', "w")
np.save(file=f, arr=selected_values)
f.close()