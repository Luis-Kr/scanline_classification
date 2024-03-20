from pathlib import Path
import pandas as pd
import numpy as np
import pickle
# Hydra and OmegaConf imports
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
import gzip


def load_data(cfg: DictConfig):
    """
    Load data from a CSV file or a list of CSV files and return a pandas DataFrame.

    Args:
        file_paths (str or list of str): The path(s) to the CSV file(s).

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded data.
    """    
    with open(Path(cfg.cls_3d.output_dir) / cfg.cls_3d.sampling.attributes_path, 'rb') as file:
        attribute_statistics = pickle.load(file)

    data_frames = []
    for file_path in (Path(cfg.cls_3d.output_dir) / cfg.cls_3d.sampling.files_dir).glob('*.gz'):
        print(f'Loading {file_path}')
        pcd_attributes_file = gzip.GzipFile(file_path, "r")
        pcd_attributes = np.load(pcd_attributes_file)
        
        # Create a DataFrame from the attributes
        pcd_attributes_df = pd.DataFrame(pcd_attributes, columns=attribute_statistics)
        
        print(f'Shape of the input data: {pcd_attributes_df.shape}')
        
        label_names={0: "unclassified",
                     1: "man-made objects",
                     2: "ground",
                     3: "tree trunk/branches",
                     4: "leaves",
                     5: "low vegetation"}
        
        pcd_attributes_df['label_names'] = pcd_attributes_df['label'].map(label_names)
        pcd_attributes_df['path'] = file_path.stem

        print(pcd_attributes_df['path'].value_counts)
        print(f"Shape of the input data: {pcd_attributes_df.shape}")
        
        pcd_attributes_frac = pcd_attributes_df.drop(columns=['x', 'y', 'z'])
        
        print(pcd_attributes_frac['label_names'].value_counts())

        pcd_attributes_frac = pcd_attributes_frac.groupby('label_names').apply(lambda x: x.sample(min(len(x), 10000), random_state=42))       
        data_frames.append(pcd_attributes_frac)
        
        print(pcd_attributes_frac['label_names'].value_counts())

    return pd.concat(data_frames, axis=0)


def save_data(cfg: DictConfig, 
              data: pd.DataFrame):
    print(data['label_names'].value_counts())
    
    output_dir = Path(cfg.cls_3d.output_dir) / cfg.cls_3d.sampling.files_dir / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Saving data to {output_dir}')
    print(f'Shape of data: {data.shape}')
    
    if 'training' in cfg.cls_3d.sampling.files_dir:
        data.to_csv(output_dir / 'training_data_merged_frac.csv', index=False)
    else:
        data.to_csv(output_dir / 'validation_data_merged_frac.csv', index=False)
    
    
@hydra.main(version_base=None, config_path="../../../config", config_name="main")
def main(cfg: DictConfig):
    dfs = load_data(cfg)
    save_data(cfg, dfs)
    

if __name__=="__main__":
    
    main()