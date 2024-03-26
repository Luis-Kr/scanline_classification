import numpy as np
from pathlib import Path
import logging
import scanline_utils.scanline_extraction as sce
import scanline_utils.scanline_segmentation as scs
import scanline_utils.scanline_subsampling as scsb
import scanline_utils.segment_classification as sgc
import utils.logger as lgr
import utils.data_validation as dv
from typing import List, Tuple
import sys
import numba
from numba import set_num_threads
import pandas as pd
import time
import csv
import os
from datetime import datetime
import pytz
import psutil
import threading
import multiprocessing
import platform
from cpuinfo import get_cpu_info
import open3d as o3d

# Hydra and OmegaConf imports
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig

import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# get the path of the current file
root_dir = Path(__file__).parent.parent.absolute()


def pcd_preprocessing(cfg: DictConfig, 
                      fmt_sce: str,
                      logger: logging.Logger):
    # Read the pcd file
    logger.info(f'Reading the point cloud: {str(root_dir / cfg.pcd_path)}')
    pcd = np.loadtxt(root_dir / cfg.pcd_path, delimiter=' ')
    
    logger.info('Adjusting theta and phi values...')
    pcd[:, cfg.pcd_col.horiz_angle], pcd[:, cfg.pcd_col.vert_angle] = sce.adjust_angles(phi_zf=pcd[:, cfg.pcd_col.horiz_angle],
                                                                                        theta_zf=pcd[:, cfg.pcd_col.vert_angle])
    
    # Compute the knickpoints
    logger.info('Computing knickpoints...')
    pcd, knickpoints = sce.find_knickpoints(pcd=pcd, 
                                            threshold=cfg.sce.threshold, 
                                            horiz_angle=cfg.pcd_col.horiz_angle,
                                            vert_angle=cfg.pcd_col.vert_angle)
    
    # Extract the scanlines
    logger.info('Extracting scanlines...')
    
    n = pcd.shape[0]
    scanlines = np.zeros(n, dtype=np.float64)
    scanlines = sce.scanline_extraction(n, 
                                        scanlines, 
                                        knickpoints)
    
    ## Calculate the mean point-to-point distances as expected values for the segmentation
    # Create a KDTree and calculate mean distances
        
    # max_distances, pcd_xyz_scanpos_centered, normals, scanner_pos = sce.kdtree_maxdist_normals(cfg=cfg,
    #                                                                               pcd=pcd[:, (cfg.pcd_col.x,
    #                                                                                           cfg.pcd_col.y,
    #                                                                                           cfg.pcd_col.z)],
    #                                                                               num_nearest_neighbors=cfg.sce.k_nn)
    
    # normals_xyz, normals = sce.align_normals_with_scanner_pos(cfg=cfg,
    #                                                           pcd=pcd_xyz_scanpos_centered, 
    #                                                           normals=normals)

    # # Bin the data
    # bins, binned_pcd = sce.bin_data(data=pcd[:, cfg.pcd_col.rho],
    #                                 bin_size=cfg.sce.bin_size)

    # # Calculate binned distances
    # binned_distances, binned_distances_std = sce.calculate_binned_distances(max_distances=max_distances, 
    #                                                                         binned_data=binned_pcd, 
    #                                                                         bins=bins)

    # # Interpolate distances
    # binned_distances_interpolated = sce.interpolate_distances(binned_distances)
    # binned_distances_std_interpolated = sce.interpolate_distances(binned_distances_std)

    # Add expected value distance to the point cloud data
    # logger.info('Adding the expected value of distance to the point cloud...')
    # pcd = sce.add_expected_value_distance(pcd=pcd, 
    #                                       binned_pcd=binned_pcd, 
    #                                       binned_distance_interp=binned_distances_interpolated,
    #                                       binned_distances_interp_std=binned_distances_std_interpolated)

    # Append the scanlines to the pcd
    pcd = sce.append_scanlines(pcd, scanlines)
    
    if cfg.sce.save_pcd:
        dv.check_path(cfg.dst_dir / cfg.paths.sce.dst_dir)
        if cfg.output_compressed:
            logger.info(f'Saving the scanlines: {str(cfg.dst_dir / cfg.paths.sce.dst_dir / cfg.filename) + "_scnln.npz"}')
            np.savez_compressed(str(cfg.dst_dir / cfg.paths.sce.dst_dir / cfg.filename) + "_scnln.npz", pcd)
        else:
            logger.info(f'Saving the scanlines: {str(cfg.dst_dir / cfg.paths.sce.dst_dir / cfg.filename) + "_scnln.txt"}')
            np.savetxt(str(cfg.dst_dir / cfg.paths.sce.dst_dir / cfg.filename) + "_scnln.txt", pcd, fmt=fmt_sce, delimiter=' ')
            
    return pcd


def scanline_segmentation(cfg: DictConfig, 
                          fmt_scs: str,
                          pcd: np.ndarray, 
                          logger: logging.Logger):    
    logger.info('Calculating segmentation metrics rho_diff, slope, curvature, roughness and normals...')
    
    # Sort the pcd by the vertical angle
    pcd_sorted, sort_indices = scs.sort_scanline(cfg=cfg, pcd=pcd)
    
    scanline_intervals_dict, scanline_id_arr = scs.get_scanline_intervals(pcd=pcd_sorted, scanline_id_col=cfg.pcd_col.scanline_id)
    
    scanner_LOS = scs.compute_scanner_LOS(pcd_sorted)
    
    rho_diff, slope_D_Z, curvature_D_Z, roughness, mean_dist, std_dist, density, curvature_scanline_3D, zenith_angle_scanline_3D = scs.calculate_segmentation_metrics(pcd=pcd_sorted, 
                                                                                                                                                                        scanline_intervals=scanline_intervals_dict,
                                                                                                                                                                        scanline_id_arr=scanline_id_arr,
                                                                                                                                                                        x_col=cfg.pcd_col.x,
                                                                                                                                                                        y_col=cfg.pcd_col.y,
                                                                                                                                                                        z_col=cfg.pcd_col.z,
                                                                                                                                                                        rho_col=cfg.pcd_col.rho,
                                                                                                                                                                        horiz_angle_col=cfg.pcd_col.horiz_angle,
                                                                                                                                                                        scanner_LOS=scanner_LOS,
                                                                                                                                                                        scanline_3D_attributes=cfg.scs.scanline_3D_attributes)
                    
    # Add the segmentation metrics to the point cloud data
    logger.info('Sorting the PCD...')
    #pcd_sorted = np.c_[pcd_sorted, density, rho_diff, curvature_phi_rho, slope_D_Z, curvature_D_Z, roughness]
    

    
    logger.info('Scanline segmentation...')
    # segment_ids = scs.scanline_segmentation(rho_diff=rho_diff,
    #                                         curvature=curvature_phi_rho,
    #                                         mean_dist=mean_dist,
    #                                         std_dist=std_dist,
    #                                         std_multiplier=cfg.scs.std_multiplier,
    #                                         curvature_threshold=cfg.scs.curvature_threshold)
    
    # Replace the slope_lstsq and curvature_lstsq with the original values
    
    # # Split pcd_sorted into two parts
    # if not cfg.sce.calculate_normals:
    #     pcd_sorted_left = pcd_sorted[:, :-3]
    #     pcd_sorted_right = pcd_sorted[:, -3:]
    # else:
    #     pcd_sorted_left = pcd_sorted[:, :-6]
    #     pcd_sorted_right = pcd_sorted[:, -6:]

    # Concatenate pcd_sorted_left, segment_ids, and pcd_sorted_right
    #pcd_segmented = np.c_[pcd_sorted, segment_ids, rho_diff, slope_D_Z, curvature_D_Z, roughness]
    #pcd_segmented = np.c_[pcd_sorted[:,:3], rho_diff, slope_D_Z, curvature_D_Z, roughness, curvature_scanline_3D, roughness_scanline_3D]
    pcd_segmented = np.c_[pcd_sorted[:,:3], density, density*3, curvature_scanline_3D, zenith_angle_scanline_3D, scanner_LOS[:,2]]
    #pcd_segmented = np.c_[pcd_sorted[:,:3], scanner_LOS[sort_indices,:], scanner_LOS_zenith[sort_indices]]
    
    # Create open3d plot
    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcd_xyz_centered[:,:3][::20])
    # pcd_o3d.normals = o3d.utility.Vector3dVector(scanner_LOS[::20] / 20)
    # o3d.visualization.draw_geometries([pcd_o3d])
    
    
    # if cfg.scs.save_pcd:
    #     dv.check_path(cfg.dst_dir / cfg.paths.scs.dst_dir)
        
    #     segmentation_filename = str(cfg.filename) + "_Segmentation"
    
    #     if cfg.output_compressed:
    #         if not cfg.sce.calculate_normals:
    #             logger.info(f'Saving the pcd with segmentation metrics: {str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".npz"}')
    #             np.savez_compressed(str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".npz", pcd_segmented)
    #         else:
    #             logger.info(f'Saving the pcd with segmentation metrics: {str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".npz"}')
    #             np.savez_compressed(str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".npz", pcd_segmented)
        
    #     else:
    #         if not cfg.sce.calculate_normals:
    #             logger.info(f'Saving the pcd with segmentation metrics: {str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".txt"}')
    #             np.savetxt(str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".txt", pcd_segmented, fmt=fmt_scs.rsplit(' ', 3)[0], delimiter=' ')
    #         else:
    #             logger.info(f'Saving the pcd with segmentation metrics: {str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".txt"}')
    #             np.savetxt(str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".txt", pcd_segmented, fmt=fmt_scs, delimiter=' ')
    
    
    if cfg.scs.save_pcd:
        dv.check_path(cfg.dst_dir / cfg.paths.scs.dst_dir)
        
        segmentation_filename = str(cfg.filename) + "_Segmentation"
    
        if cfg.output_compressed:
            if not cfg.sce.calculate_normals:
                logger.info(f'Saving the pcd with segmentation metrics: {str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".npz"}')
                np.savez_compressed(str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".npz", pcd_sorted)
            else:
                logger.info(f'Saving the pcd with segmentation metrics: {str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".npz"}')
                np.savez_compressed(str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".npz", pcd_sorted)
        
        else:
            logger.info(f'Saving the pcd with segmentation metrics: {str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".txt"}')
            np.savetxt(str(cfg.dst_dir / cfg.paths.scs.dst_dir / segmentation_filename) + ".txt", pcd_segmented, fmt="%1.4f", delimiter=' ')
            
    
    sys.exit()
    
    return pcd_segmented, pcd_sorted



def scanline_subsampling(cfg: DictConfig, 
                         fmt_scsb: str,
                         column_indices: List[int],
                         pcd: np.ndarray, 
                         logger: logging.Logger): 
    logger.info('Scanline subsampling: Calculating the segment attributes...')
    
    # Get the unique segment classes
    segment_classes = np.array(list(set(pcd[:,cfg.pcd_col.segment_ids])))
    
    # Initialize the processed segments array
    processed_segments = np.zeros((segment_classes.shape[0], fmt_scsb.count("%")))
    
    # Get the number of points in each segment
    _, counts = np.unique(pcd[:,cfg.pcd_col.segment_ids], return_counts=True)
    
    # Sort the point cloud by segment id
    sorted_indices = np.argsort(pcd[:,cfg.pcd_col.segment_ids])
    
    # Split the sorted indices into segments
    indices_per_class = np.split(sorted_indices, np.cumsum(counts[:-1]))
    
    # Calculate the segment attributes
    pcd_processed_segments, indices_per_class, gini_impurity = scsb.process_segments(pcd=pcd, 
                                                                                    segment_classes=segment_classes, 
                                                                                    processed_segments=processed_segments, 
                                                                                    counts=counts,
                                                                                    x_col=cfg.pcd_col.x,
                                                                                    y_col=cfg.pcd_col.y,
                                                                                    z_col=cfg.pcd_col.z,
                                                                                    column_indices=numba.typed.List(column_indices),
                                                                                    segment_id_col=cfg.pcd_col.segment_ids,
                                                                                    label_col=cfg.pcd_col.label,
                                                                                    segment_ids_col=cfg.pcd_col.segment_ids)
    
    logger.info(f'Subsampled pcd shape: {pcd_processed_segments.shape}')
    
    if cfg.scsb.save_pcd:
        dv.check_path(cfg.dst_dir / cfg.paths.scsb.dst_dir)
        
        subsampled_filename = str(cfg.filename) + "_Subsampled"
        
        if cfg.output_compressed:
            logger.info(f'Saving the subsampled pcd: {str(cfg.dst_dir / cfg.paths.scsb.dst_dir / subsampled_filename) + ".npz"}')
            np.savez_compressed(str(cfg.dst_dir / cfg.paths.scsb.dst_dir / subsampled_filename) + ".npz", pcd_processed_segments)
        else:
            logger.info(f'Saving the subsampled pcd: {str(cfg.dst_dir / cfg.paths.scsb.dst_dir / subsampled_filename) + ".txt"}')
            np.savetxt(str(cfg.dst_dir / cfg.paths.scsb.dst_dir / subsampled_filename) + ".txt", pcd_processed_segments, fmt=fmt_scsb, delimiter=' ')


    if cfg.scsb.save_gini_impurity:
        dv.check_path(cfg.dst_dir / cfg.paths.scsb.gini_impurity)
        filename = str(cfg.filename) + "_GiniImpurity"
        
        df = pd.DataFrame({
            'std_multiplier': [cfg.scs.std_multiplier],
            'curvature_threshold': [cfg.scs.curvature_threshold],
            'neighborhood_multiplier': [cfg.scs.neighborhood_multiplier]
        })

        # Write the DataFrame to a CSV file
        nm_string = str(cfg.scs.neighborhood_multiplier).replace('.', '_')
        df.to_csv(str(cfg.dst_dir / cfg.paths.scsb.gini_impurity / filename) + f"_StdM{cfg.scs.std_multiplier}_CurvT{cfg.scs.curvature_threshold}_NghM{nm_string}.csv", index=False)
        
        if cfg.output_compressed:
            logger.info(f'Saving the gini impurity values: {str(cfg.dst_dir / cfg.paths.scsb.gini_impurity / filename) + f"_StdM{cfg.scs.std_multiplier}_CurvT{cfg.scs.curvature_threshold}_NghM{nm_string}.npz"}')
            np.savez_compressed(str(cfg.dst_dir / cfg.paths.scsb.gini_impurity / filename) + f"_StdM{cfg.scs.std_multiplier}_CurvT{cfg.scs.curvature_threshold}_NghM{nm_string}.npz", gini_impurity)
        else:
            logger.info(f'Saving the gini impurity values: {str(cfg.dst_dir / cfg.paths.scsb.gini_impurity / filename) + f"_StdM{cfg.scs.std_multiplier}_CurvT{cfg.scs.curvature_threshold}_NghM{nm_string}.txt"}')
            np.savetxt(str(cfg.dst_dir / cfg.paths.scsb.gini_impurity / filename) + f"_StdM{cfg.scs.std_multiplier}_CurvT{cfg.scs.curvature_threshold}_NghM{nm_string}.txt", gini_impurity, fmt='%1.4f', delimiter=' ')
    
    sys.exit()
            
    return pcd_processed_segments, indices_per_class


def track_performance(cfg):
    performance_metrics_path = Path(cfg.dst_dir) / "performance_report" / 'performance_metrics.csv'
    performance_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Record the start time
    start_time = time.time()
    
    # if the file does not exist, create it and write the header
    if not performance_metrics_path.exists():
        with open(performance_metrics_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(["Timestamp", "CPU Usage (%)", "Memory Usage (GB)", "Method", "File", "Seconds_Since_Start"])

    # Open the file in append mode
    with open(performance_metrics_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        
        # Write the header only if the file is empty
        if file.tell() == 0:
            writer.writerow(["Timestamp", "CPU Usage (%)", "Memory Usage (GB)", "Method", "File", "Seconds_Since_Start"])

        while True:
            berlin_tz = pytz.timezone('Europe/Berlin')
            berlin_time = datetime.now(berlin_tz)
            timestamp = berlin_time.strftime("%Y-%m-%d %H:%M:%S")
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage_gb = psutil.Process().memory_info().rss / (1024 ** 3)
            seconds_since_start = time.time() - start_time  # Calculate the seconds since the start

            #print(f'Timestamp: {timestamp}, CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage_gb:.3f} GB')
            writer.writerow([timestamp, cpu_usage, memory_usage_gb, "scanline_approach", Path(cfg.pcd_path).stem, round(seconds_since_start,0)])

            file.flush()  # Flush the file buffer
            os.fsync(file.fileno())  # Ensure it's written to disk
            
            time.sleep(0.01)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    # Clear the hydra config cache
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    threading.Thread(target=track_performance, args=(cfg,), daemon=True).start()
    start_time = time.time()
    
    # Check if cfg.output_dir is None (if yes, set it to the current working directory)
    if cfg.dst_dir == "None":
        cfg.dst_dir = Path.cwd()
    
    if not isinstance(cfg.dst_dir, Path):
        cfg.dst_dir = Path(cfg.dst_dir)
        
    if not isinstance(cfg.pcd_path, Path):
        cfg.pcd_path = Path(cfg.pcd_path)
        
    dv.check_path(cfg.dst_dir)
    dv.check_path(cfg.dst_dir / cfg.paths.scsb.attribute_stats)
    
    # Set up the logger
    logger = lgr.logger_setup('main', cfg.dst_dir / cfg.paths.logger.dst_dir / "main.log")
    
    # Clear the log file
    if cfg.clear_logs:
        with open(cfg.dst_dir / cfg.paths.logger.dst_dir / "main.log", 'w'):
            pass
        
    cfg.filename = Path(cfg.pcd_path).stem
    
    logger.info('Preparing the attributes...')
    
    fmt_sce, fmt_scs, fmt_scsb, column_indices, fmt_pcd_classified = dv.prepare_attributes_and_format(cfg=cfg)
    
    dv.check_attributes_and_normals(cfg=cfg)
    
    # PCD preprocessing
    pcd=pcd_preprocessing(cfg=cfg, fmt_sce=fmt_sce, logger=logger)
    
    # PCD scanline segmentation
    pcd_segmented, pcd_sorted=scanline_segmentation(cfg=cfg, 
                                                    fmt_scs=fmt_scs,
                                                    pcd=pcd, 
                                                    logger=logger)
    
    # PCD scanline subsampling
    pcd_processed_segments, indices_per_class=scanline_subsampling(cfg=cfg, 
                                                                   fmt_scsb=fmt_scsb,
                                                                   column_indices=column_indices,
                                                                   pcd=pcd_segmented, 
                                                                   logger=logger)
    
    if cfg.run_classification:
        logger.info('Segment classification...')
        
        dv.check_path(cfg.dst_dir / cfg.paths.segcl.dst_dir_metrics)
        dv.check_path(cfg.dst_dir / cfg.paths.segcl.dst_dir_cnfmat)
        dv.check_path(cfg.dst_dir / cfg.paths.segcl.dst_dir_subsampled_pcd)
        dv.check_path(cfg.dst_dir / cfg.paths.segcl.dst_dir_subsampled_pcd)
        
        # PCD segment classification
        predicted_labels_subs = sgc.segment_classification(cfg=cfg,
                                                           pcd_subsampled=pcd_processed_segments,
                                                           model_filepath=root_dir / cfg.paths.rf_model,
                                                           metrics_output_filepath=cfg.dst_dir / cfg.paths.segcl.dst_dir_metrics / (str(cfg.filename) + "_metrics.csv"),
                                                           cnfmatrix_output_path=cfg.dst_dir / cfg.paths.segcl.dst_dir_cnfmat / (str(cfg.filename) + "_cnfmatrix.txt"),
                                                           pcd_subsampled_classified_path=cfg.dst_dir / cfg.paths.segcl.dst_dir_subsampled_pcd / cfg.filename)
        
        logger.info('Unfolding and assigning labels...')
        predicted_labels = sgc.unfold_labels(pcd=pcd_sorted, 
                                             pcd_subs_predicted_labels=predicted_labels_subs,
                                             indices_per_class=numba.typed.List(indices_per_class))
        
        logger.info('Assigning labels to the full resolution PCD...')
        pcd_classified = sgc.assign_labels(pcd=pcd_sorted, predicted_labels=predicted_labels)
        
        if cfg.sgcl.save_pcd:
            dv.check_path(cfg.dst_dir / cfg.paths.segcl.dst_dir_pcd_classified)
            
            if cfg.output_compressed:
                logger.info(f'Saving the classified pcd: {str(cfg.dst_dir / cfg.paths.segcl.dst_dir_pcd_classified / (str(cfg.filename) + "_classified.npz"))}')
                np.savez_compressed(str(cfg.dst_dir / cfg.paths.segcl.dst_dir_pcd_classified / (str(cfg.filename) + "_classified.npz")), pcd_classified)
            else:
                logger.info(f'Saving the classified pcd: {str(cfg.dst_dir / cfg.paths.segcl.dst_dir_pcd_classified / (str(cfg.filename) + "_classified.txt"))}')
                np.savetxt(str(cfg.dst_dir / cfg.paths.segcl.dst_dir_pcd_classified / (str(cfg.filename) + "_classified.txt")), pcd_classified, fmt=fmt_pcd_classified, delimiter=' ')

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time of main is: ", execution_time, "seconds")
    
    # Save the execution time
    time_df = pd.DataFrame(data={"execution_time (s)": [execution_time],
                                "file": [cfg.pcd_path.stem]})

    csv_file = Path(cfg.dst_dir) / "performance_report" / "execution_time.csv"
    #time_df_out_path.mkdir(parents=True, exist_ok=True)

    # If the file exists, append without writing the header again, else write with header
    if csv_file.exists():
        time_df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        time_df.to_csv(csv_file, mode='w', header=True, index=False)
    
    
    #-------------------------------------------------------------------------------------------------------------------
    # Get system details
    machine_info = platform.machine()
    platform_info = platform.platform()
    cpu_name = get_cpu_info()['brand_raw']
    cpu_info = platform.processor()
    ram_info = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB
    num_cores_used = int(multiprocessing.cpu_count() - 1)

    # Create a DataFrame with the system details
    system_details_df = pd.DataFrame(data={
        "Machine Info": [machine_info],
        "Platform Info": [platform_info],
        "CPU Name": [cpu_name],
        "CPU Info": [cpu_info],
        "RAM Info (GB)": [ram_info],
        "Number of Cores Used": [num_cores_used]
    })

    # Define the output path
    system_details_path = Path(cfg.dst_dir) / "performance_report" / "system_details.csv"

    # If the file exists, append without writing the header again, else write with header
    if system_details_path.exists():
        pass
    else:
        print("Writing the system details to a CSV file...")
        system_details_df.to_csv(system_details_path, mode='w', header=True, index=False)

if __name__=='__main__':
    # Set the numba threads
    set_num_threads(int(multiprocessing.cpu_count() - 1))
    main()