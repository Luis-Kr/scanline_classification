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
    logger.info('Calculating the mean point-to-point distances...')
    mean_distances, _ = sce.create_kdtree(points=pcd[:, (cfg.pcd_col.x,
                                                         cfg.pcd_col.y,
                                                         cfg.pcd_col.z)])

    # Bin the data
    bins, binned_pcd = sce.bin_data(data=pcd[:, cfg.pcd_col.rho],
                                    bin_size=cfg.sce.bin_size)

    # Calculate binned distances
    binned_distances = sce.calculate_binned_distances(mean_distances=mean_distances, 
                                                      binned_data=binned_pcd, 
                                                      bins=bins)

    # Interpolate distances
    binned_distances_interpolated = sce.interpolate_distances(binned_distances=binned_distances)

    # Add expected value distance to the point cloud data
    logger.info('Adding the expected value of distance to the point cloud data...')
    pcd = sce.add_expected_value_distance(pcd=pcd, 
                                          binned_pcd=binned_pcd, 
                                          binned_distance_interp=binned_distances_interpolated)

    # Append the scanlines to the pcd
    pcd = sce.append_scanlines(pcd, scanlines)
    
    if cfg.sce.save_pcd:
        # Save the scanlines
        logger.info(f'Saving the scanlines: {Path(root_dir) / "data/raw_plus_scanline_extraction/SiteA_Scans_Global_I_RGB_RHV/Scan01_with_scanlineID.asc"}')
        np.savetxt(Path(root_dir) / 'data/02_raw_plus_scanline_extraction/SiteA_Scans_Global_I_RGB_RHV/Scan01_with_scanlineID.asc', 
                   pcd, fmt=cfg.sce.fmt, delimiter=' ')
    
    return pcd


def scanline_segmentation(cfg: DictConfig, pcd: np.ndarray, logger: logging.Logger):    
    logger.info('Calculating the segmentation metrics rho_diff, slope and curvature...')
    rho_diff, slope, curvature, pcd_sorted = scs.calculate_segmentation_metrics(pcd, 
                                                                                x_col=cfg.pcd_col.x,
                                                                                y_col=cfg.pcd_col.y,
                                                                                z_col=cfg.pcd_col.z,
                                                                                sort_col=cfg.pcd_col.vert_angle,
                                                                                scanline_id_col=cfg.pcd_col.scanline_id,
                                                                                rho_col=cfg.pcd_col.rho)
    
    # Add the segmentation metrics to the point cloud data
    logger.info('Sorting the PCD...')
    pcd_sorted = np.c_[pcd_sorted, rho_diff, slope, curvature]
    pcd_sorted = pcd_sorted[np.lexsort(np.rot90(pcd_sorted[:,(cfg.pcd_col.scanline_id,
                                                              cfg.pcd_col.vert_angle)]))]
    
    logger.info('Scanline segmentation...')
    segment_ids = scs.scanline_segmentation(pcd_sorted,
                                            expected_value_col=cfg.pcd_col.expected_value,
                                            rho_diff_col=cfg.pcd_col.rho_diff,
                                            slope_col=cfg.pcd_col.slope,
                                            curvature_col=cfg.pcd_col.curvature,
                                            expected_value_factor=cfg.scs.expected_value_factor,
                                            slope_threshold=cfg.scs.slope_threshold,
                                            curvature_threshold=cfg.scs.curvature_threshold)
    
    pcd_segmented = np.c_[pcd_sorted, segment_ids]
    
    if cfg.scs.save_pcd:
        logger.info(f'Saving the pcd with segmentation metrics: {Path(root_dir) / "data/raw_plus_scanline_plus_segmentation/SiteA_Scans_Global_I_RGB_RHV/Scan01_ScanlineID_Segmentation.asc"}')
        np.savetxt(Path(root_dir) / 'data/03_raw_plus_scanline_plus_segmentation/SiteA_Scans_Global_I_RGB_RHV/Scan01_ScanlineID_Segmentation.asc', 
                   pcd_segmented, fmt=cfg.scs.fmt, delimiter=' ')
        
    return pcd_segmented



def scanline_subsampling(cfg: DictConfig, pcd: np.ndarray, logger: logging.Logger): 
    logger.info('Scanline subsampling: Calculating the segment attributes...')
    
    # Get the unique segment classes
    segment_classes = np.array(list(set(pcd[:,cfg.pcd_col.segment_ids])))
    
    # Initialize the processed segments array
    processed_segments = np.zeros((segment_classes.shape[0], 57))
    
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
                                                   slope_col=cfg.pcd_col.slope,
                                                   curvature_col=cfg.pcd_col.curvature,
                                                   segment_ids_col=cfg.pcd_col.segment_ids)
    
    if cfg.scsb.save_pcd:
        logger.info(f'Saving the pcd with segmentation metrics: {Path(root_dir) / "data/04_raw_plus_scanline_plus_segmentation_plus_subsampling/SiteA_Scans_Global_I_RGB_RHV/Scan01_ScanlineID_Segmentation_Subsampling.asc"}')
        np.savetxt(Path(root_dir) / "data/04_raw_plus_scanline_plus_segmentation_plus_subsampling/SiteA_Scans_Global_I_RGB_RHV/Scan01_ScanlineID_Segmentation_Subsampling.asc", 
                   pcd_processed_segments, fmt=cfg.scsb.fmt, delimiter=' ')
    
    print(pcd_processed_segments.shape)



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
    pcd=pcd_preprocessing(cfg=cfg, logger=logger)
    
    # PCD scanline segmentation
    pcd_segmented=scanline_segmentation(cfg=cfg, pcd=pcd, logger=logger)
    
    # PCD scanline subsampling
    scanline_subsampling(cfg=cfg, pcd=pcd_segmented, logger=logger)


if __name__=='__main__':
    main()