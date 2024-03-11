import pandas as pd
from pathlib import Path
from tqdm import tqdm

def concatenate_csv_files(dir):
    dfs = []  # an empty list to store the data frames
    for _, file in enumerate(tqdm(list(Path(dir).glob('*_fr.csv')))):  # loop through csv files in the directory
        
        df = pd.read_csv(file)  # read each csv file as a data frame
        dfs.append(df)  # append the data frame to the list
        
    # concatenate all the data frames in the list.
    # ignore_index=True reindexes the final dataframe so the index is continuous
    df = pd.concat(dfs, ignore_index=True)
    return df


dir = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/classification_training_sample10000/classification_report"
df = concatenate_csv_files(dir)

df = df.rename(columns={df.columns[0]: "attribute"})
df = df.drop(columns=["support"])
output_dir = "/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3_v02/classification_training_sample10000/all_results"
df.to_csv(f"{output_dir}/feature_importance_results_full_res.csv", index=False)

