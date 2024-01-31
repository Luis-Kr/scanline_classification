import pandas as pd 
import numpy as np
from pathlib import Path


def load_data_asc(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path, delimiter=' ', header=None,
                       names=['x', 
                              'y', 
                              'z', 
                              'intensity', 
                              'red', 
                              'green', 
                              'blue', 
                              'rho', 
                              'horizontal_angle', 
                              'vertical_angle',
                              'point_id'])
    return data


def load_data_asc_labeled_all_scans(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path, delimiter=' ', header=None,
                       names=['x', 
                              'y', 
                              'z', 
                              'red', 
                              'green', 
                              'blue', 
                              'intensity', 
                              'rho', 
                              'horizontal_angle', 
                              'vertical_angle',
                              'point_id',
                              'label'])
    return data


def load_data_labeled_all_scans_extra_attributes(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path, delimiter=' ', header=None,
                       names=['x', 
                              'y', 
                              'z', 
                              'red', 
                              'green', 
                              'blue',
                              'intensity', 
                              'blank1', 'blank2', 'blank3', 'blank4', 'blank5', 'blank6', 'blank7', 'blank8', 'blank9', 'blank10', 'blank11','blank12', 'blank13', 'blank14','label'])
    return data


def main(input_dir: Path, output_dir: Path, all_scans_file: Path):
    # Load data
    scans_merged_classified = load_data_asc_labeled_all_scans(all_scans_file)
    scans_merged_classified = scans_merged_classified[['x','y','z','red','green','blue','label']]
    scans_merged_classified = scans_merged_classified.drop_duplicates(subset=['x','y','z','red','green','blue'])
    
    # scans_merged_classified = load_data_labeled_all_scans_extra_attributes(all_scans_file)
    # scans_merged_classified = scans_merged_classified[['x','y','z','intensity','label']]
    # scans_merged_classified = scans_merged_classified.drop_duplicates(subset=['x', 'y', 'z','intensity'])
    
    for input_path in list(sorted(input_dir.glob("*.asc"))):
        # Load the scan
        print(f'Loading scan: {input_path}')
        
        scan = load_data_asc(input_path)
        
        print(f'Number of points in scan: {scan.shape[0]}')
        
        # The problem here was that CC corrupted the point_id column when saving the .asc files
        # X,Y,Z,Red,Green,Blue remained the same, so these values were used to merge the dataframes
        scan_df_classes = pd.merge(scan, 
                                   scans_merged_classified, 
                                   on=['x','y','z','red','green','blue'], 
                                   how='left')
        
        # count the number of nan values in the label column
        nan_count = scan_df_classes['label'].isna().sum()
        print(f'Number of nan values in label column: {nan_count}')
        print(scan_df_classes)
        
        # Fill the nan values with 0
        scan_df_classes['label'] = scan_df_classes['label'].fillna(0)
        
        # Save the scan01_classes_np as a .txt file and define the data format
        fmt = "%1.4f %1.4f %1.4f %1.6f %u %u %u %1.4f %1.6f %1.6f %u %u"
        output_file = output_dir / str(input_path.stem).replace("_Counter", "_Labeled.txt")
        print(output_file)
        np.savetxt(output_file, scan_df_classes.to_numpy(), delimiter=' ', fmt=fmt)


if __name__ == '__main__':
    root_dir = Path(__file__).parent.parent.parent
    # input_dir = root_dir / "data/02_number_of_points_counter/SiteD_Scans_Global_I_RGB_RHV"
    # output_dir = root_dir / "data/03_labeled/SiteD_Scans_Global_I_RGB_RHV"
    # all_scans_file = root_dir / "data/Labels/SiteD_Scans_Global_I_RGB_RHV/SiteD_RHV_Counter_Labeled_PCD_ALLSCANS.txt"
    
    # input_dir = root_dir / "data/02_number_of_points_counter/SiteA_Scans_Global_I_RGB_RHV"
    # output_dir = root_dir / "data/03_labeled/SiteA_Scans_Global_I_RGB_RHV"
    # all_scans_file = root_dir / "data/Labels/SiteA_Scans_Global_I_RGB_RHV/SiteA_RHV_Counter_Labeled_PCD_ALLSCANS.txt"
    
    input_dir = root_dir / "data/02_number_of_points_counter/SiteB_Scans_Global_I_RGB_RHV"
    output_dir = root_dir / "data/03_labeled/SiteB_Scans_Global_I_RGB_RHV"
    all_scans_file = root_dir / "data/Labels/SiteB_Scans_Global_I_RGB_RHV/SiteB_RHV_Counter_Labeled_PCD_ALLSCANS.txt"
    
    main(input_dir, output_dir, all_scans_file)