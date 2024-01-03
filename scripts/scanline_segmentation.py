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
    logger.info(f'Reading the point cloud: {Path(root_dir) / "data/raw/SiteA_Scans_Global_I_RGB_RHV/SiteA_RHV_01.asc"}')
    
    pcd = np.loadtxt(Path(root_dir) / "data/raw/SiteA_Scans_Global_I_RGB_RHV/SiteA_RHV_01.asc", 
                     delimiter=' ')
    
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
    
    # Append the scanlines to the pcd
    pcd_with_scanlines = sce.append_scanlines(pcd, scanlines)
    
    if cfg.save_pcd:
        # Save the scanlines
        logger.info(f'Saving the scanlines: {Path(root_dir) / "data/raw_plus_scanline_extraction/SiteA_Scans_Global_I_RGB_RHV/Scan01_with_scanlineID.asc"}')
        np.savetxt(Path(root_dir) / 'data/raw_plus_scanline_extraction/SiteA_Scans_Global_I_RGB_RHV/Scan01_with_scanlineID.asc', 
                   pcd_with_scanlines, fmt=cfg.sce.fmt, delimiter=' ')


if __name__=='__main__':
    pcd_preprocessing()