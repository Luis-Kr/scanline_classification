from pathlib import Path
import pandas as pd
import numpy as np
import pickle

def load_data(file_paths: str,
              attribute_statistics_path: str,
              drop_cols: list,
              scs: bool, 
              scs_colnames: list):
    """
    Load data from a CSV file or a list of CSV files and return a pandas DataFrame.

    Args:
        file_paths (str or list of str): The path(s) to the CSV file(s).

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded data.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
        
    with open(attribute_statistics_path, 'rb') as f:
        attribute_statistics = pickle.load(f)

    data_frames = []
    for file_path in file_paths:
        print(f'Loading {file_path}')
        if not scs:
            data = pd.read_csv(file_path, delimiter=' ', header=None, names=attribute_statistics)
        else:
            data = pd.read_csv(file_path, delimiter=' ', header=None, names=scs_colnames)
        print(f'Shape of the input data: {data.shape}')
        
        label_names={0: "unclassified",
                     1: "man-made objects",
                     2: "ground",
                     3: "tree trunk/branches",
                     4: "leaves",
                     5: "low vegetation"}
        
        data['label_names'] = data['label'].map(label_names)
        data['path'] = file_path.stem

        print(data['path'].value_counts)
        print(f"Shape of the input data: {data.shape}")
        
        if not scs:
            pcd_df_frac = data.drop(columns=drop_cols)
        else:
            pcd_df_frac = data
        
        print(pcd_df_frac['label_names'].value_counts())

        pcd_df_frac = pcd_df_frac.groupby('label_names').apply(lambda x: x.sample(min(len(x), 10000), random_state=42))       
        data_frames.append(pcd_df_frac)
        
        print(pcd_df_frac['label_names'].value_counts())

    return pd.concat(data_frames, axis=0)


def save_data(data: pd.DataFrame, dst_dir: str):
    """
    Save data to a CSV file.

    Args:
        data (pandas.DataFrame): The data to save.
        dst_dir (str): The directory where the data will be saved.
    """
    print(data['label_names'].value_counts())
    
    print(f'Saving data to {dst_dir}')
    dst_dir.mkdir(parents=False, exist_ok=True)
    print(f'Shape of data: {data.shape}')
    data.to_csv(dst_dir / 'training_data_merged_frac_test.csv', index=False)
    

if __name__=="__main__":
    # pcd_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training/scanline_subsampling")
    # file_paths = list(pcd_dir.glob('*.txt'))
    # dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training/scanline_subsampling/merged")
    # attribute_statistics_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training/scanline_subsampling/attribute_statistics/SiteA_RHV_01_Labeled.pkl")

    # pcd_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_validation")
    # file_paths = list(pcd_dir.glob('*.txt'))
    # dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_validation/merged")
    # attribute_statistics_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training/scanline_subsampling/attribute_statistics/SiteA_RHV_01_Labeled.pkl")
    
    # pcd_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/training")
    # file_paths = list(pcd_dir.glob('*.txt'))
    # dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/training/merged")
    # attribute_statistics_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/scanline_subsampling/attribute_statistics/SiteA_RHV_01_Labeled.pkl")
    
    # pcd_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/validation")
    # file_paths = list(pcd_dir.glob('*.txt'))
    # dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/validation/merged")
    # attribute_statistics_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/scanline_subsampling/attribute_statistics/SiteA_RHV_01_Labeled.pkl")
    
    pcd_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/training")
    file_paths = list(pcd_dir.glob('*.txt'))
    dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/training/merged")
    attribute_statistics_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/scanline_subsampling/attribute_statistics/SiteA_RHV_01_Labeled.pkl")
    
    # pcd_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/validation")
    # file_paths = list(pcd_dir.glob('*.txt'))
    # dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/validation/merged")
    # attribute_statistics_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/scanline_subsampling/attribute_statistics/SiteA_RHV_01_Labeled.pkl")
    
    # pcd_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/scanline_segmentation")
    # file_paths = list(pcd_dir.glob('*.txt'))
    # dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/scanline_segmentation/merged")
    # attribute_statistics_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/scanline_subsampling/attribute_statistics/SiteA_RHV_01_Labeled.pkl")
    
    drop_cols = ["x_median_nn", "y_median_nn", "x_perc2nd_nn", "y_perc2nd_nn", "x_perc98th_nn", "y_perc98th_nn", "segment_id"]
    
    scs = False
    scs_colnames = ["X","Y","Z","Intensity","Red","Green","Blue","Rho","Phi","Vert_angle","Point_Counter","label","Expected_value","Expected_value_std","Scanline_id","Rho_diff","Slope","Curvature","Roughness","Segment_ids","nx_xyz","ny_xyz","nz_xyz","nx","ny","nz"]
    
    data = load_data(file_paths, attribute_statistics_path, drop_cols, scs, scs_colnames)
    save_data(data, dst_dir)