from pathlib import Path
import sys
from typing import List, Dict, Tuple
import scanline_utils.scanline_subsampling as scsb

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig


def check_path(inp_path):
    """
    Check if a given path exists, and create it if it doesn't.

    Parameters
    ----------
    inp_path : str
        The path to check/create.

    Returns
    -------
    None
    """
    
    obj = Path(inp_path)
    
    # Check if the path exists (if not, create it)
    if not obj.exists():
        obj.mkdir(parents=True)
        

def prepare_attributes_and_format(cfg: DictConfig) -> Tuple[str, str, str, List[int], List[str]]:
    
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
    fmt_sce = " ".join(fmt for fmt in list(cfg.pcd_col_fmt.values())[:13])
    
    # fmt scanline segmentation
    fmt_scs = " ".join(fmt for fmt in list(cfg.pcd_col_fmt.values()))
    
    # fmt scanline subsampling
    column_indices, column_fmt = get_column_indices(attributes=cfg.attributes, 
                                                    pcd_col=cfg.pcd_col,
                                                    pcd_col_fmt=cfg.pcd_col_fmt)
    
    fmt_scsb = " ".join(fmt for fmt in column_fmt for _ in range(9)) + " " + "%u" + " " + "%u" #9 because of the number of statistics
    fmt_scsb = " ".join(["%1.4f"] * len(cfg.xyz_attributes)) + " " + fmt_scsb

    attribute_statistics = [f"{attribute}_{statistic}" for attribute in cfg.attributes for statistic in cfg.statistics]
    attribute_statistics = cfg.xyz_attributes + attribute_statistics + ["segment_id"] + ["label"]
    
    # Save attribute statistics as pickle file and json file
    scsb.save_attribute_statistics(file_path=cfg.dst_dir / cfg.paths.scsb.attribute_stats / cfg.filename,
                                   attribute_statistics=list(attribute_statistics))
    
    fmt_pcd_classified = " ".join(fmt for fmt in list(cfg.pcd_col_fmt.values())[:12] + ["%u"])

    return fmt_sce, fmt_scs, fmt_scsb, column_indices, fmt_pcd_classified
        
        
def check_attributes_and_normals(cfg: DictConfig):
    if cfg.sce.calculate_normals == False and any(x in cfg.attributes for x in ["nx", "ny", "nz"]):
        sys.exit("""Error: The attributes contain 'nx', 'ny', and 'nz'. 
                 However, the calculate_normals is set to False. 
                 Please set the calculate_normals to True or remove 'nx', 'ny', and 'nz' from the attributes.""")
    elif cfg.sce.calculate_normals == True and not any(x in cfg.attributes for x in ["nx", "ny", "nz"]):
        sys.exit("""Error: The attributes do not contain 'nx', 'ny', and 'nz'. \n
                    However, the calculate_normals is set to True. \n
                    Please set the calculate_normals to False or add 'nx', 'ny', and 'nz' to the attributes.""")

