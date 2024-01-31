import numpy as np
from pathlib import Path

def process_files(input_dir: str, 
                  output_dir: str) -> None:
    """
    Function to assign a unique value to each point in a scan session.

    Parameters:
    input_dir (str): The directory containing the input .asc files.
    output_dir (str): The directory where the processed files will be saved.
    """
    # Initialize start_id to 0
    start_id = 0

    # Loop through each file in the directory
    for _, file in enumerate(sorted(Path(input_dir).glob("*.asc"))):
        print(f'Loading {file}')
        # Load the data from the file
        data = np.loadtxt(file, delimiter=" ")

        # Calculate np.arange(data.shape[0]) and append it to the data
        point_number_range = np.arange(start_id, start_id + data.shape[0])
        data = np.c_[data, point_number_range]
        start_id += data.shape[0]
        
        print(f'Saving {file}')
        # Define the format for saving the data
        fmt = "%1.4f %1.4f %1.4f %1.6f %u %u %u %1.4f %1.6f %1.6f %u"

        # Create a new filename by adding "_Counter" to the stem of the original filename
        new_filename = file.stem + "_Counter.asc"

        # Save the processed data to the output directory
        np.savetxt(Path(output_dir) / new_filename, data, delimiter=" ", fmt=fmt)
        
        # Delete the data to free up memory
        del data


if __name__ == "__main__":
    input_dir = "/Users/luiskremer/Code/Uni/Code_Master_RSIV/019_scanline_segmentation/data/01_raw/SiteB_Scans_Global_I_RGB_RHV"
    output_dir = "/Users/luiskremer/Code/Uni/Code_Master_RSIV/019_scanline_segmentation/data/02_number_of_points_counter/SiteB_Scans_Global_I_RGB_RHV"
    process_files(input_dir, output_dir)