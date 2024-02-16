import numpy as np
from pathlib import Path
import logging
import utils.scanline_extraction as sce
import utils.scanline_segmentation as scs
import utils.scanline_subsampling as scsb
import utils.logger as lgr
from typing import List, Dict, Tuple
import sys
import numba

# Hydra and OmegaConf imports
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig


# get the path of the current file
root_dir = Path(__file__).parent.parent.absolute()


def prepare_attributes_and_format(cfg: DictConfig, 
                                  logger: logging.Logger) -> Tuple[str, str, str, List[int], List[str]]:
    
    logger.info('Preparing the attributes...')
    
    def get_column_indices(attributes: List[str], 
                           pcd_col: Dict[str, int], 
                           pcd_col_fmt: Dict[str, str]) -> Tuple[List[int], str]:
        try:
            column_indices = [pcd_col[attribute.lower()] for attribute in attributes]
            column_fmt = [pcd_col_fmt[attribute.lower()] for attribute in attributes]
            return column_indices, column_fmt
        except KeyError as e:
            raise KeyError(f"The attribute '{e.args[0]}' is not found in the dictionary. Please check the attribute names.")

    # fmt scanline extraction
    fmt_sce = " ".join(fmt for fmt in list(cfg.pcd_col_fmt.values())[:15])
    
    # fmt scanline segmentation
    fmt_scs = " ".join(fmt for fmt in list(cfg.pcd_col_fmt.values()))
    
    # fmt scanline subsampling
    column_indices, column_fmt = get_column_indices(attributes=cfg.attributes, 
                                                    pcd_col=cfg.pcd_col,
                                                    pcd_col_fmt=cfg.pcd_col_fmt)
    
    fmt_scsb = " ".join(fmt for fmt in column_fmt for _ in range(7)) + " " + "%u" #7 because of the number of statistics
    fmt_scsb = " ".join(["%1.4f"] * len(cfg.xyz_attributes)) + " " + fmt_scsb

    attribute_statistics = [f"{attribute}_{statistic}" for attribute in cfg.attributes for statistic in cfg.statistics]
    attribute_statistics = cfg.xyz_attributes + attribute_statistics + ["label"]

    return fmt_sce, fmt_scs, fmt_scsb, column_indices, attribute_statistics


def check_attributes_and_normals(cfg: DictConfig):
    if cfg.sce.calculate_normals == False and all(x in cfg.attributes for x in ["nx", "ny", "nz"]):
        sys.exit("""Error: The attributes contain 'nx', 'ny', and 'nz'. 
                 However, the calculate_normals is set to False. 
                 Please set the calculate_normals to True or remove 'nx', 'ny', and 'nz' from the attributes.""")
    elif cfg.sce.calculate_normals == True and not all(x in cfg.attributes for x in ["nx", "ny", "nz"]):
        sys.exit("""Error: The attributes do not contain 'nx', 'ny', and 'nz'. \n
                    However, the calculate_normals is set to True. \n
                    Please set the calculate_normals to False or add 'nx', 'ny', and 'nz' to the attributes.""")


def pcd_preprocessing(cfg: DictConfig, 
                      fmt_sce: str,
                      logger: logging.Logger):
    # Read the pcd file
    logger.info(f'Reading the point cloud: {Path(root_dir) / cfg.pcd_path}')
    
    pcd = np.loadtxt(Path(root_dir) / cfg.pcd_path, delimiter=' ')
    
    logger.info('Adjusting the theta and phi values...')
    pcd[:, cfg.pcd_col.horiz_angle], pcd[:, cfg.pcd_col.vert_angle] = sce.adjust_angles(phi_zf=pcd[:, cfg.pcd_col.horiz_angle],
                                                                                        theta_zf=pcd[:, cfg.pcd_col.vert_angle])
    
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
        sce_path = Path(*[part if part != '03_labeled' else '04_scanline_extraction' for part in Path(cfg.pcd_path).parts])
        logger.info(f'Saving the scanlines: {Path(root_dir) / sce_path}')
        np.savetxt(Path(root_dir) / sce_path, pcd, fmt=fmt_sce, delimiter=' ')
    
    return pcd, pcd_xyz_scanpos_centered, normals_xyz, normals


def scanline_segmentation(cfg: DictConfig, 
                          fmt_scs: str,
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
                                                                               horiz_angle_col=cfg.pcd_col.horiz_angle,
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
            np.savetxt(Path(root_dir) / segmentation_path, pcd_segmented, fmt=fmt_scs.rsplit(' ', 3)[0], delimiter=' ')
        else:
            logger.info(f'Saving the pcd with segmentation metrics: {Path(root_dir) / segmentation_path}')
            np.savetxt(Path(root_dir) / segmentation_path, pcd_segmented, fmt=fmt_scs, delimiter=' ')
    
    return pcd_segmented



def scanline_subsampling(cfg: DictConfig, 
                         fmt_scsb: str,
                         column_indices: List[int],
                         attribute_statistics: List[str],
                         pcd: np.ndarray, 
                         logger: logging.Logger): 
    logger.info('Scanline subsampling: Calculating the segment attributes...')
    
    # Get the unique segment classes
    segment_classes = np.array(list(set(pcd[:,cfg.pcd_col.segment_ids])))
    
    # Initialize the processed segments array
    processed_segments = np.zeros((segment_classes.shape[0], fmt_scsb.count("%")))
    
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
                                                   column_indices=numba.typed.List(column_indices),
                                                   label_col=cfg.pcd_col.label,
                                                   segment_ids_col=cfg.pcd_col.segment_ids)
    
    if cfg.scsb.save_pcd:
        
        segmentation_path = Path(*[part if part != '03_labeled' else '06_subsampling' for part in Path(cfg.pcd_path).parts])
        
        if not cfg.sce.relocate_origin:
            segmentation_path = segmentation_path.with_stem(segmentation_path.stem + "_Subsampling")
        else:
            segmentation_path = segmentation_path.with_stem(segmentation_path.stem + f"_Subsampling_ScanPosRelocated{cfg.sce.z_offset}m")
        
        if cfg.sce.calculate_normals:
            logger.info(f'Saving the pcd with segmentation metrics: {Path(root_dir) / segmentation_path}')
            print(pcd_processed_segments.shape)
            np.savetxt(Path(root_dir) / segmentation_path, pcd_processed_segments, fmt=fmt_scsb, delimiter=' ')
        else:
            logger.info(f'Saving the pcd with segmentation metrics: {Path(root_dir) / segmentation_path}')
            print(pcd_processed_segments.shape)
            np.savetxt(Path(root_dir) / segmentation_path, pcd_processed_segments, fmt=fmt_scsb, delimiter=' ')



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
    
    fmt_sce, fmt_scs, fmt_scsb, column_indices, attribute_statistics = prepare_attributes_and_format(cfg=cfg,
                                                                                                     logger=logger)
    
    check_attributes_and_normals(cfg=cfg)
    
    # PCD preprocessing
    pcd, pcd_xyz_scanpos_centered, normals_xyz, normals=pcd_preprocessing(cfg=cfg, 
                                                                          fmt_sce=fmt_sce,
                                                                          logger=logger)
    
    # PCD scanline segmentation
    pcd_segmented=scanline_segmentation(cfg=cfg, 
                                        fmt_scs=fmt_scs,
                                        pcd=pcd, 
                                        pcd_xyz_scanpos_centered=pcd_xyz_scanpos_centered,
                                        logger=logger, 
                                        normals_xyz=normals_xyz, 
                                        normals=normals)
    
    # PCD scanline subsampling
    scanline_subsampling(cfg=cfg, 
                         fmt_scsb=fmt_scsb,
                         column_indices=column_indices,
                         attribute_statistics=attribute_statistics,
                         pcd=pcd_segmented, 
                         logger=logger)


if __name__=='__main__':
    main()