import numpy as np
from numba import njit, jit, prange
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logger as lgr
from scipy.linalg import svd

# get the path of the current file
root_dir = Path(__file__).parent.parent.parent.absolute()


def center_pcd(pcd: np.ndarray) -> np.ndarray:
    # Compute the centroid of the point cloud
    centroid = np.median(pcd[:,:3], axis=0)

    # Center the point cloud
    pcd_centered = pcd[:,:3] - centroid

    return pcd_centered


@njit(parallel=True)
def compute_curvature(pcd_centered: np.ndarray, 
                      radius: float) -> np.ndarray:
    
    curvature = np.zeros(pcd_centered.shape[0])
    
    for i in prange(pcd_centered.shape[0]):
        # Select the a point from the point cloud
        pt = pcd_centered[i,:3]
        dist_pt = np.zeros(pcd_centered.shape[0])
        
        # Compute the distance between the point and all other points in the point cloud
        for j in prange(pcd_centered.shape[0]):
            dist_pt[j] = np.sqrt(np.sum((pcd_centered[j,:3] - pt)**2))
            
        # Select the points within the given radius
        pt_nbh = pcd_centered[dist_pt < radius]
        
        # Compute the covariance matrix of the selected points
        if pt_nbh.shape[0] > 1:
            eigenvalues, _ = np.linalg.eigh(np.cov(pt_nbh.T))
            curvature[i] = np.min(eigenvalues) / np.sum(eigenvalues)
        else:
            curvature[i] = 0
        
        # Compute the curvature as the ratio of the smallest eigenvalue to the sum of all eigenvalues
        curvature[i] = np.min(eigenvalues) / np.sum(eigenvalues)
        
    return curvature


@njit(parallel=True)
def compute_roughness(pcd_centered: np.ndarray, radius: float) -> np.ndarray:
    roughness = np.zeros(pcd_centered.shape[0])
    
    for i in prange(pcd_centered.shape[0]):
        # Select a point from the point cloud
        pt = pcd_centered[i,:3]
        dist_pt = np.zeros(pcd_centered.shape[0])
        
        # Compute the distance between the point and all other points in the point cloud
        for j in prange(pcd_centered.shape[0]):
            dist_pt[j] = np.sqrt(np.sum((pcd_centered[j,:3] - pt)**2))
            
        # Select the points within the given radius
        pt_nbh = pcd_centered[dist_pt < radius]
        
        # Compute the best fitting plane through the neighborhood
        if pt_nbh.shape[0] > 2:
            # Compute the covariance matrix of the neighborhood points
            C = np.cov(pt_nbh[:,:3], rowvar=False)
            
            # Compute the eigenvalues and eigenvectors of the covariance matrix
            eigenvalues, eigenvectors = np.linalg.eig(C)
            
            # The normal of the plane is the eigenvector corresponding to the smallest eigenvalue
            normal = eigenvectors[:, np.argmin(eigenvalues)]
            
            # Compute the mean of the neighborhood points along axis 0 manually
            mean_pt_nbh = np.sum(pt_nbh[:,:3], axis=0) / pt_nbh.shape[0]
            
            # Compute the distance from the point to the plane
            d = np.abs(np.dot(pt - mean_pt_nbh, normal))
            roughness[i] = d
        else:
            roughness[i] = 0
            
    return roughness


def column_stack(cfg: DictConfig,
                 pcd: np.ndarray, 
                 curvature: np.ndarray) -> np.ndarray:
    return np.c_[pcd[:,(cfg.pcd_col.x, 
                        cfg.pcd_col.y, 
                        cfg.pcd_col.z, 
                        cfg.pcd_col.curvature)], curvature]


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig) -> None:
    # Clear the hydra config cache
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # Set up the logger
    logger = lgr.logger_setup('curvature_analysis', 
                              Path(root_dir) / "data/logs/analysis/a01_curvature.log")
    
    # Load the point cloud data
    logger.info(f"Loading the point cloud: {cfg.a01_curvature.pcd_path}")
    pcd = np.loadtxt(root_dir / cfg.a01_curvature.pcd_path, delimiter=' ')
    
    pcd = pcd[::30]
    
    # Center the point cloud
    logger.info("Centering the point cloud")
    pcd_centered = center_pcd(pcd)
    
    # Compute the curvature of the point cloud
    for i, radius in enumerate(cfg.a01_curvature.radius):
        logger.info(f"Computing the curvature of the point cloud with radius: {radius}m")
        curvature = compute_roughness(pcd_centered, radius)
        
        # Add the curvature to the point cloud data
        logger.info("Adding the curvature to the point cloud data")
        if i == 0:
            pcd_curvature = column_stack(cfg, pcd, curvature)
        else:
            pcd_curvature = np.c_[pcd_curvature, curvature]
    
    # Save the curvature to a file
    logger.info(f"Saving the curvature to: {cfg.a01_curvature.output_path}")
    np.savetxt(cfg.a01_curvature.output_path, pcd_curvature, delimiter=" ", fmt="%1.6f")
    
    
if __name__ == "__main__":
    main()
