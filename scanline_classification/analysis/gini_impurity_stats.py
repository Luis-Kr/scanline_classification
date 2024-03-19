import pandas as pd 
from pathlib import Path
import numpy as np

# Define column names globally
col_names = ["X", "Y", "Z", "point_counts", "gini_impurity"]

def calculate_statistics(file_txt: Path):
    """Calculate statistics for a single file."""
    
    if file_txt.name.count(".") > 1:
        new_filename_txt = file_txt.name.replace(".", "_", 1)
        old_filename_csv = file_txt.with_suffix(".csv")
        new_filename_csv = old_filename_csv.name.replace(".", "_", 1)
        print(f'Renaming {file_txt.name} to {new_filename_txt}')
        file_txt.rename(file_txt.parent / new_filename_txt)
        old_filename_csv.rename(file_txt.parent / new_filename_csv)
        
        file_txt = file_txt.parent / new_filename_txt
    
    print(f'Processing {file_txt.name}')
    
    file_csv = file_txt.with_suffix(".csv")
    
    pcd_df = pd.read_csv(file_txt, sep=" ", names=col_names)
    params = pd.read_csv(file_csv, sep=",")

    # Add parameters to dataframe
    for param in ['std_multiplier', 'curvature_threshold', 'neighborhood_multiplier']:
        pcd_df[param] = params[param].values[0]
    
    # Combine parameters into a single string
    pcd_df['params'] = "SM" + pcd_df['std_multiplier'].astype(str) + "_CT" + pcd_df['curvature_threshold'].astype(str) + "_NM" + pcd_df['neighborhood_multiplier'].astype(str)

    # Calculate statistics
    stats = {
        'number_segments': pcd_df.shape[0],
        'number_segments_low_points5': pcd_df[pcd_df['point_counts'] < 5].shape[0],
        'number_segments_low_points10': pcd_df[pcd_df['point_counts'] < 10].shape[0],
        'impurity_mean': pcd_df['gini_impurity'].mean(),
        'impurity_std': pcd_df['gini_impurity'].std(),
        'impurity_variance': pcd_df['gini_impurity'].var(),
        'impurity_ptp': np.ptp(pcd_df['gini_impurity']),
        'impurity_2percentile': np.percentile(pcd_df['gini_impurity'], 2),
        'impurity_98percentile': np.percentile(pcd_df['gini_impurity'], 98),
        'std_multiplier': pcd_df['std_multiplier'].values[0],
        'curvature_threshold': pcd_df['curvature_threshold'].values[0],
        'neighborhood_multiplier': pcd_df['neighborhood_multiplier'].values[0],
        'params': pcd_df['params'].values[0]
    }

    # Calculate percentages
    for key in ['number_segments_low_points5', 'number_segments_low_points10']:
        stats[f'percentage_{key}'] = stats[key] / stats['number_segments'] * 100

    return stats

def process_directory(dir: Path):
    """Process all .txt files in a directory."""
    return (calculate_statistics(file_txt) for file_txt in dir.glob("*.txt"))

def main():
    # Define input and output paths
    #input_path = Path('/DATA/Luis/thesis/scanline_classification/data/gini_impurity_coarse/scanline_subsampling/gini_impurity')
    input_path = Path('/DATA/Luis/thesis/scanline_classification/data/gini_impurity_fine/scanline_subsampling/gini_impurity')
    output_file = input_path.parent / 'gini_impurity_stats' / 'gini_impurity_stats.csv'

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process files and save results
    stats_df = pd.DataFrame(process_directory(input_path))
    stats_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()























# import pandas as pd 
# from pathlib import Path
# import numpy as np


# def calculate_statistics(dir: Path):
    
#     for file_txt in dir.glob("*.txt"):
#         print(f'Processing {file_txt.name}')
        
#         file_csv = file_txt.with_suffix(".csv")
#         pcd_df = pd.read_csv(file_txt, sep=" ", names=col_names)
#         params = pd.read_csv(file_csv, sep=",")

#         pcd_df['std_multiplier'] = params['std_multiplier'].values[0]
#         pcd_df['curvature_threshold'] = params['curvature_threshold'].values[0]
#         pcd_df['neighborhood_multiplier'] = params['neighborhood_multiplier'].values[0]
#         pcd_df['params'] = "SM" + pcd_df['std_multiplier'].astype(str) + "_CT" + pcd_df['curvature_threshold'].astype(str) + "_NM" + pcd_df['neighborhood_multiplier'].astype(str)

#         number_segments = pcd_df.shape[0]
#         number_segments_low_points5 = pcd_df[pcd_df['point_counts'] < 5].shape[0]
#         number_segments_low_points10 = pcd_df[pcd_df['point_counts'] < 10].shape[0]
#         percentage_segments_low_points5 = number_segments_low_points5 / number_segments * 100
#         percentage_segments_low_points10 = number_segments_low_points10 / number_segments * 100

#         impurity_mean = pcd_df['gini_impurity'].mean()
#         impurity_std = pcd_df['gini_impurity'].std()
#         impurity_variance = pcd_df['gini_impurity'].var()
#         impurity_ptp = np.ptp(pcd_df['gini_impurity'])
#         impurity_2percentile = np.percentile(pcd_df['gini_impurity'], 2)
#         impurity_98percentile = np.percentile(pcd_df['gini_impurity'], 98)

#         yield {'number_segments': number_segments, 
#                 'number_segments_low_points5': number_segments_low_points5,
#                 'number_segments_low_points10': number_segments_low_points10,
#                 'percentage_segments_low_points5': percentage_segments_low_points5,
#                 'percentage_segments_low_points10': percentage_segments_low_points10,
#                 'impurity_mean': impurity_mean,
#                 'impurity_std': impurity_std,
#                 'impurity_variance': impurity_variance,
#                 'impurity_ptp': impurity_ptp,
#                 'impurity_2percentile': impurity_2percentile,
#                 'impurity_98percentile': impurity_98percentile,
#                 'std_multiplier': pcd_df['std_multiplier'].values[0],
#                 'curvature_threshold': pcd_df['curvature_threshold'].values[0],
#                 'neighborhood_multiplier': pcd_df['neighborhood_multiplier'].values[0],
#                 'params': pcd_df['params'].values[0]}
        
# # Use the function
# col_names = ["X", "Y", "Z", "point_counts", "gini_impurity"]
# input_path = Path('/DATA/Luis/thesis/scanline_classification/data/gini_impurity/scanline_subsampling/gini_impurity')
# output_file = input_path.parent / 'gini_impurity_stats' / 'gini_impurity_stats.csv'

# if not output_file.parent.exists():
#     output_file.mkdir(parents=True)
    
# stats_df = pd.DataFrame(calculate_statistics(input_path))

# # Save the dataframe
# stats_df.to_csv(output_file, index=False)