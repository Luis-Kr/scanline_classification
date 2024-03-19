import pandas as pd
from pathlib import Path
import psutil
import time
import threading
import matplotlib.pyplot as plt

def track_memory(func):
    def wrapper(*args, **kwargs):
        # Lists to store memory usage and timestamps
        mem_usage = []
        timestamps = []

        # Flag to indicate whether to continue tracking memory usage
        tracking = True

        # Function to track memory usage
        def track_memory():
            while tracking:
                mem_usage.append(psutil.virtual_memory().used / 1024**3)
                timestamps.append(time.time())
                time.sleep(1)  # Wait for one second

        # Start a separate thread to track memory usage
        tracker = threading.Thread(target=track_memory)
        tracker.start()

        result = func(*args, **kwargs)

        # Stop tracking memory usage
        tracking = False
        tracker.join()

        # Create a plot of memory usage over time
        plt.plot(timestamps, mem_usage)
        plt.xlabel('Time')
        plt.ylabel('Memory usage (GB)')
        #plt.show()
        
        # Save the plot
        plot_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training/scanline_subsampling/merged/memory_usage.png")
        #plt.savefig(plot_path)

        return result

    return wrapper

@track_memory
def process_data(pcds_path, drop_cols, dst_dir):
    label_names={0: "unclassified",
                 1: "man-made objects",
                 2: "ground",
                 3: "tree trunk/branches",
                 4: "leaves",
                 5: "low vegetation"}

    print(f"Loading {pcds_path}")
    
    pcd_df = pd.read_csv(pcds_path, sep=",", header=0)
    pcd_df['label_names'] = pcd_df['label'].map(label_names)

    print(f"Shape of the input data: {pcd_df.shape}")
    print(f"Memory usage: {psutil.virtual_memory().used / 1024**3:.2f} GB")
    
    pcd_df_frac = pcd_df.drop(columns=drop_cols)
    
    print(pcd_df_frac['label_names'].value_counts())

    pcd_df_frac = pcd_df_frac.groupby('label_names').sample(n=50000, replace=False, random_state=42)

    # Save the data
    print(f"Saving data to {dst_dir}")
    dst_dir.mkdir(parents=False, exist_ok=True)
    pcd_df_frac.to_csv(dst_dir / 'training_data_merged_frac.csv', index=False)

# pcds_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training/scanline_subsampling/merged/training_data_merged_all.csv")
# drop_cols = ["x_median_nn", "y_median_nn", "x_perc2nd_nn", "y_perc2nd_nn", "x_perc98th_nn", "y_perc98th_nn", "segment_id"]
# dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training/scanline_subsampling/merged")

# pcds_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_validation/merged/validation_data_merged_all.csv")
# drop_cols = ["x_median_nn", "y_median_nn", "x_perc2nd_nn", "y_perc2nd_nn", "x_perc98th_nn", "y_perc98th_nn", "segment_id"]
# dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_validation/merged")

# pcds_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/training/merged/training_data_merged_all.csv")
# drop_cols = ["x_median_nn", "y_median_nn", "x_perc2nd_nn", "y_perc2nd_nn", "x_perc98th_nn", "y_perc98th_nn", "segment_id"]
# dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/training/merged")

# pcds_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/training/merged/training_data_merged_all.csv")
# drop_cols = ["x_median_nn", "y_median_nn", "x_perc2nd_nn", "y_perc2nd_nn", "x_perc98th_nn", "y_perc98th_nn", "segment_id"]
# dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/training/merged")

# pcds_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/validation/merged/validation_data_merged_all.csv")
# drop_cols = ["x_median_nn", "y_median_nn", "x_perc2nd_nn", "y_perc2nd_nn", "x_perc98th_nn", "y_perc98th_nn", "segment_id"]
# dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_z_normalization/validation/merged")

pcds_path = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3/training/merged/training_data_merged_all.csv")
drop_cols = ["x_median_nn", "y_median_nn", "x_perc2nd_nn", "y_perc2nd_nn", "x_perc98th_nn", "y_perc98th_nn", "segment_id"]
dst_dir = Path("/DATA/Luis/thesis/scanline_classification/data/data_for_training_final_SM2CT20NM3/training/merged")

process_data(pcds_path, drop_cols, dst_dir)