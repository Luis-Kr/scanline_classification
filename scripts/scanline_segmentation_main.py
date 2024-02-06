import numpy as np
from pathlib import Path
import logging
import utils.scanline_extraction as sce
import utils.scanline_segmentation as scs
import utils.scanline_subsampling as scsb
import utils.logger as lgr

# Hydra and OmegaConf imports
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig


# get the path of the current file
root_dir = Path(__file__).parent.parent.absolute()


def pcd_preprocessing(cfg: DictConfig, logger: logging.Logger):
    # Read the pcd file
    logger.info(f'Reading the point cloud: {Path(root_dir) / cfg.pcd_path}')
    
    pcd = np.loadtxt(Path(root_dir) / cfg.pcd_path, delimiter=' ')
    
    # Compute the knickpoints
    logger.info('Computing the knickpoints...')
    
    pcd, knickpoints = sce.find_knickpoints(pcd=pcd, 
                                            threshold=cfg.sce.threshold, 
                                            horiz_angle=cfg.pcd_col.horiz_angle,
                                            vert_angle=cfg.pcd_col.vert_angle)
    
    # Extract the scanlines
    logger.info('Extracting the scanlines...')
    
    n = pcd.shape[0]
    scanlines = np.zeros(n, dtype=np.float64)
    scanlines = sce.scanline_extraction(n, 
                                        scanlines, 
                                        knickpoints)
    
    ## Calculate the mean point-to-point distances as expected values for the segmentation
    # Create a KDTree and calculate mean distances
    if cfg.sce.calculate_normals:
        logger.info('Calculating the max point-to-point distances and the normals...')
    else:
        logger.info('Calculating the max point-to-point distances...')
        
    max_distances, pcd_xyz_scanpos_centered, normals = sce.kdtree_maxdist_normals(cfg=cfg,
                                                                                  pcd=pcd[:, (cfg.pcd_col.x,
                                                                                              cfg.pcd_col.y,
                                                                                              cfg.pcd_col.z)],
                                                                                  num_nearest_neighbors=cfg.sce.k_nn)
    
    normals_xyz, normals = sce.align_normals_with_scanner_pos(cfg=cfg,
                                                              pcd=pcd_xyz_scanpos_centered, 
                                                              normals=normals)

    # Bin the data
    bins, binned_pcd = sce.bin_data(data=pcd[:, cfg.pcd_col.rho],
                                    bin_size=cfg.sce.bin_size)

    # Calculate binned distances
    binned_distances, binned_distances_std = sce.calculate_binned_distances(max_distances=max_distances, 
                                                                            binned_data=binned_pcd, 
                                                                            bins=bins)

    # Interpolate distances
    binned_distances_interpolated = sce.interpolate_distances(binned_distances)
    binned_distances_std_interpolated = sce.interpolate_distances(binned_distances_std)

    # Add expected value distance to the point cloud data
    logger.info('Adding the expected value of distance to the point cloud data...')
    pcd = sce.add_expected_value_distance(pcd=pcd, 
                                          binned_pcd=binned_pcd, 
                                          binned_distance_interp=binned_distances_interpolated,
                                          binned_distances_interp_std=binned_distances_std_interpolated)

    # Append the scanlines to the pcd
    pcd = sce.append_scanlines(pcd, scanlines)
    
    if cfg.sce.save_pcd:
        # Save the scanlines
        logger.info(f'Saving the scanlines: {Path(root_dir) / "data/raw_plus_scanline_extraction/SiteA_Scans_Global_I_RGB_RHV/Scan01_with_scanlineID.asc"}')
        np.savetxt(Path(root_dir) / 'data/04_scanline_extraction/SiteA_Scans_Global_I_RGB_RHV/Scan01_with_scanlineID.asc', 
                   pcd, fmt=cfg.sce.fmt, delimiter=' ')
    
    return pcd, pcd_xyz_scanpos_centered, normals_xyz, normals


def scanline_segmentation(cfg: DictConfig, 
                          pcd: np.ndarray, 
                          pcd_xyz_scanpos_centered: np.ndarray,
                          logger: logging.Logger, 
                          normals_xyz: np.ndarray, 
                          normals: np.ndarray):    
    logger.info('Calculating the segmentation metrics rho_diff, slope, curvature, roughness and normals...')
    
    # Sort the pcd by the vertical angle
    pcd_sorted, sort_indices = scs.sort_scanline(pcd=pcd, 
                                                 scanline_id_col=cfg.pcd_col.scanline_id, 
                                                 vert_angle_col=cfg.pcd_col.vert_angle)
    
    scanline_intervals = scs.get_scanline_intervals(pcd=pcd_sorted, 
                                                    scanline_id_col=cfg.pcd_col.scanline_id)
    
    rho_diff, slope, curvature, roughness = scs.calculate_segmentation_metrics(pcd=pcd_sorted, 
                                                                               scanline_intervals=scanline_intervals,
                                                                               x_col=cfg.pcd_col.x,
                                                                               y_col=cfg.pcd_col.y,
                                                                               z_col=cfg.pcd_col.z,
                                                                               expected_value_col=cfg.pcd_col.expected_value,
                                                                               rho_col=cfg.pcd_col.rho,
                                                                               least_squares_method=cfg.scs.least_squares_method)
    
    # Add the segmentation metrics to the point cloud data
    logger.info('Sorting the PCD...')
    if not cfg.sce.calculate_normals:
        pcd_sorted = np.c_[pcd_sorted, rho_diff, slope, curvature, roughness, normals_xyz[sort_indices]]
    else:
        pcd_sorted = np.c_[pcd_sorted, rho_diff, slope, curvature, roughness, normals_xyz[sort_indices], normals[sort_indices]]
    
    logger.info('Scanline segmentation...')
    segment_ids = scs.scanline_segmentation(pcd_sorted,
                                            expected_value_col=cfg.pcd_col.expected_value,
                                            expected_value_std_col=cfg.pcd_col.expected_value_std,
                                            std_multiplier=cfg.scs.std_multiplier,
                                            rho_diff_col=cfg.pcd_col.rho_diff,
                                            slope_col=cfg.pcd_col.slope,
                                            curvature_col=cfg.pcd_col.curvature,
                                            expected_value_factor=cfg.scs.expected_value_factor,
                                            slope_threshold=cfg.scs.slope_threshold,
                                            curvature_threshold=cfg.scs.curvature_threshold)
    
    # Split pcd_sorted into two parts
    if not cfg.sce.calculate_normals:
        pcd_sorted_left = pcd_sorted[:, :-3]
        pcd_sorted_right = pcd_sorted[:, -3:]
    else:
        pcd_sorted_left = pcd_sorted[:, :-6]
        pcd_sorted_right = pcd_sorted[:, -6:]

    # Concatenate pcd_sorted_left, segment_ids, and pcd_sorted_right
    pcd_segmented = np.c_[pcd_sorted_left, segment_ids, pcd_sorted_right]
    
    if cfg.sce.relocate_origin:
        pcd_segmented = scs.recalculate_rho(cfg=cfg, 
                                            pcd=pcd_segmented, 
                                            pcd_xyz_scanpos_centered=pcd_xyz_scanpos_centered)

    
    if cfg.scs.save_pcd:
        
        segmentation_path = Path(*[part if part != '03_labeled' else '05_segmentation' for part in Path(cfg.pcd_path).parts])
        
        if not cfg.sce.relocate_origin:
            segmentation_path = segmentation_path.with_stem(segmentation_path.stem + "_Segmentation")
        else:
            segmentation_path = segmentation_path.with_stem(segmentation_path.stem + f"_Segmentation_ScanPosRelocated{cfg.sce.z_offset}m")
        
        if not cfg.sce.calculate_normals:
            logger.info(f'Saving the pcd with segmentation metrics: {Path(root_dir) / segmentation_path}')
            np.savetxt(Path(root_dir) / segmentation_path, 
                       pcd_segmented, fmt=cfg.scs.fmt, delimiter=' ')
        else:
            logger.info(f'Saving the pcd with segmentation metrics: {Path(root_dir) / segmentation_path}')
            np.savetxt(Path(root_dir) / segmentation_path, 
                    pcd_segmented, fmt=cfg.scs.fmt_normals, delimiter=' ')
    
    return pcd_segmented



def scanline_subsampling(cfg: DictConfig, pcd: np.ndarray, logger: logging.Logger): 
    logger.info('Scanline subsampling: Calculating the segment attributes...')
    
    # Get the unique segment classes
    segment_classes = np.array(list(set(pcd[:,cfg.pcd_col.segment_ids])))
    
    # Initialize the processed segments array
    processed_segments = np.zeros((segment_classes.shape[0], 106))
    
    # Get the number of points in each segment
    _, counts = np.unique(pcd[:,cfg.pcd_col.segment_ids], return_counts=True)
    
    # Calculate the segment attributes
    pcd_processed_segments = scsb.process_segments(pcd=pcd, 
                                                   segment_classes=segment_classes, 
                                                   processed_segments=processed_segments, 
                                                   counts=counts,
                                                   x_col=cfg.pcd_col.x,
                                                   y_col=cfg.pcd_col.y,
                                                   z_col=cfg.pcd_col.z,
                                                   height_col=cfg.pcd_col.z,
                                                   intensity_col=cfg.pcd_col.intensity,
                                                   red_col=cfg.pcd_col.red,
                                                   green_col=cfg.pcd_col.green,
                                                   blue_col=cfg.pcd_col.blue,
                                                   rho_col=cfg.pcd_col.rho,
                                                   label_col=cfg.pcd_col.label,
                                                   slope_col=cfg.pcd_col.slope,
                                                   curvature_col=cfg.pcd_col.curvature,
                                                   roughness_col=cfg.pcd_col.roughness,
                                                   segment_ids_col=cfg.pcd_col.segment_ids,
                                                   normals_xyz_col=np.array([cfg.pcd_col.nx_xyz,
                                                                             cfg.pcd_col.ny_xyz,
                                                                             cfg.pcd_col.nz_xyz]),
                                                   normals_col=np.array([cfg.pcd_col.nx,
                                                                         cfg.pcd_col.ny,
                                                                         cfg.pcd_col.nz]))
    
    print(pcd_processed_segments.shape)
    
    if not cfg.sce.calculate_normals:
        # remove the last 21 values from the processed segments
        pcd_processed_segments = np.hstack((pcd_processed_segments[:, :-22],  pcd_processed_segments[:, -1:]))
    
    if cfg.scsb.save_pcd:
        
        segmentation_path = Path(*[part if part != '03_labeled' else '06_subsampling' for part in Path(cfg.pcd_path).parts])
        
        if not cfg.sce.relocate_origin:
            segmentation_path = segmentation_path.with_stem(segmentation_path.stem + "_Subsampling")
        else:
            segmentation_path = segmentation_path.with_stem(segmentation_path.stem + f"_Subsampling_ScanPosRelocated{cfg.sce.z_offset}m")
        
        if cfg.sce.calculate_normals:
            logger.info(f'Saving the pcd with segmentation metrics: {Path(root_dir) / segmentation_path}')
            print(pcd_processed_segments.shape)
            np.savetxt(Path(root_dir) / segmentation_path, 
                    pcd_processed_segments, fmt=cfg.scsb.fmt_normals, delimiter=' ')
        else:
            logger.info(f'Saving the pcd with segmentation metrics: {Path(root_dir) / segmentation_path}')
            print(pcd_processed_segments.shape)
            np.savetxt(Path(root_dir) / segmentation_path, 
                    pcd_processed_segments, fmt=cfg.scsb.fmt, delimiter=' ')



@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    # Clear the hydra config cache
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # Set up the logger
    logger = lgr.logger_setup('functionality_logger', 
                              Path(root_dir) / "data/logs/module_functionality.log")
    
    # Clear the log file
    with open(Path(root_dir) / "data/logs/module_functionality.log", 'w'):
        pass
    
    # PCD preprocessing
    pcd, pcd_xyz_scanpos_centered, normals_xyz, normals=pcd_preprocessing(cfg=cfg, 
                                                                          logger=logger)
    
    # PCD scanline segmentation
    pcd_segmented=scanline_segmentation(cfg=cfg, 
                                        pcd=pcd, 
                                        pcd_xyz_scanpos_centered=pcd_xyz_scanpos_centered,
                                        logger=logger, 
                                        normals_xyz=normals_xyz, 
                                        normals=normals)
    
    # PCD scanline subsampling
    scanline_subsampling(cfg=cfg, 
                         pcd=pcd_segmented, 
                         logger=logger)


if __name__=='__main__':
    main()