import numpy as np
from pathlib import Path
import logging
import utils.scanline_extraction as sce
import utils.logger as lgr

# Hydra and OmegaConf imports
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig


# get the path of the current file
root_dir = Path(__file__).parent.parent.absolute()


@hydra.main(version_base=None, config_path="../config", config_name="main")
def pcd_preprocessing(cfg: DictConfig):
    # Set up the logger
    logger = lgr.logger_setup('functionality_logger', 
                              Path(root_dir) / "data/logs/module_functionality.log")
    
    with open(Path(root_dir) / "data/logs/module_functionality.log", 'w'):
        pass
    
    # Read the pcd file
    logger.info(f'Reading the point cloud: {Path(root_dir) / cfg.pcd_path}')
    
    pcd = np.loadtxt(Path(root_dir) / cfg.pcd_path, delimiter=' ')
    
    # Compute the knickpoints
    logger.info('Computing the knickpoints...')
    
    pcd, knickpoints = sce.find_knickpoints(pcd=pcd, 
                                            threshold=cfg.sce.threshold, 
                                            horiz_angle=cfg.col_pcd.horiz_angle,
                                            vert_angle=cfg.col_pcd.vert_angle)
    
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
    mean_distances, _ = sce.create_kdtree(points=pcd[:, (0,1,2)])

    # Bin the data
    bins, binned_pcd = sce.bin_data(data=pcd[:, -3],
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
    pcd_with_scanlines = sce.append_scanlines(pcd, scanlines)
    
    if cfg.save_pcd:
        # Save the scanlines
        logger.info(f'Saving the scanlines: {Path(root_dir) / "data/raw_plus_scanline_extraction/SiteA_Scans_Global_I_RGB_RHV/Scan01_with_scanlineID.asc"}')
        np.savetxt(Path(root_dir) / 'data/raw_plus_scanline_extraction/SiteA_Scans_Global_I_RGB_RHV/Scan01_with_scanlineID.asc', 
                   pcd_with_scanlines, fmt=cfg.sce.fmt, delimiter=' ')


if __name__=='__main__':
    pcd_preprocessing()