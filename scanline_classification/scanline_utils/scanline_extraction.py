import numpy as np 
from typing import Tuple
from numba import njit, prange, int64, float64
from scipy.spatial import cKDTree


def adjust_angles(phi_zf: np.ndarray,
                  theta_zf: np.ndarray) -> tuple:
    """
    Adjusts theta and phi values for a Z+F point cloud.

    Args:
        theta_zf (np.ndarray): Array of theta values.
        phi_zf (np.ndarray): Array of phi values.

    Returns:
        tuple: Adjusted theta and phi values.
    """
    # Flip theta values
    theta_adjusted = 360 - theta_zf

    # Get indices of theta values > 180
    theta_adjusted_idx = np.where(theta_adjusted > 180)[0]

    # Flip and shift theta values > 180
    theta_adjusted[theta_adjusted_idx] *= -1
    theta_adjusted[theta_adjusted_idx] += 360

    # Adjust corresponding phi values
    phi_zf[theta_adjusted_idx] -= 180

    return phi_zf, theta_adjusted


def sort_pcd(pcd: np.ndarray, 
             col: int = -2) -> np.ndarray:
    return pcd[pcd[:, col].argsort()]


def find_knickpoints(pcd: np.ndarray, 
                     threshold: int = 100, 
                     horiz_angle: int = -2,
                     vert_angle: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the knickpoints in the point cloud data.

    Knickpoints are points where the vertical angle difference exceeds a given threshold.

    Parameters:
    pcd (np.ndarray): The point cloud data.
    threshold (int): The threshold for the vertical angle difference.
    horiz_angle (int): The column index for the horizontal angle in the pcd.
    vert_angle (int): The column index for the vertical angle in the pcd.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The sorted point cloud data and the indices of the knickpoints.
    """
    # Sort the pcd based on the second last column
    pcd = sort_pcd(pcd, col=horiz_angle)
    
    # Calculate the absolute difference of the last column
    vertical_ang_diff = abs(np.diff(pcd[:, vert_angle]))
    
    # Append the last value of vertical_ang_diff to itself
    vertical_ang_diff = np.append(vertical_ang_diff, vertical_ang_diff[-1])
    
    # Find the indices where vertical_ang_diff is greater than the threshold
    knickpoints = np.where(vertical_ang_diff > threshold)[0]
    
    return pcd, knickpoints


@njit([(int64, float64[:], int64[:])], parallel=True)
def scanline_extraction(n, scanlines, knickpoints):
   for i in prange(n):
      # For each i, find the index in the sorted knickpoints array where i should be inserted to maintain sorted order.
      # 'side=left' means that the first suitable location is given.
      scanlines[i] = np.searchsorted(knickpoints, i, side='left')
   
   # Increment all elements in scanlines by 1
   scanlines += 1
   
   return scanlines


def append_scanlines(pcd: np.ndarray, 
                     scanlines: np.ndarray) -> np.ndarray:
    # Use np.c_ to concatenate pcd and scanlines along the second axis (columns)
    return np.c_[pcd, scanlines]


def create_kdtree(points: np.ndarray, 
                  k: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a KDTree from the given points and queries the tree for the k nearest neighbors of each point.

    Parameters:
    points (np.ndarray): The points to create the KDTree from.
    k (int): The number of nearest neighbors to query for.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The mean distances and indices of the k nearest neighbors.
    """
    # Create KDTree from points
    tree = cKDTree(points)
    
    # Query for k nearest neighbors
    distances, indices = tree.query(points, workers=-1, k=k)
    
    # Calculate max of 4 nearest distances, excluding self (index 0)
    max_distances = np.max(distances[:, 1:k], axis=1)
    
    # Replace zero distances with a small number
    #max_distances = np.where(max_distances == 0, 0.000001, max_distances)
    
    return max_distances, indices


def bin_data(data: np.ndarray, 
             bin_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bins the given data into bins of the given size.

    Parameters:
    data (np.ndarray): The data to bin.
    bin_size (int): The size of the bins.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The bins and the binned data.
    """
    # Create bins from 0 to max of data with step size as bin_size
    bins = np.arange(0, np.max(data), bin_size)
    
    # Assign each data point to a bin
    binned_data = np.digitize(data, bins)
    
    return bins, binned_data


def calculate_binned_distances(max_distances: np.ndarray, 
                               binned_data: np.ndarray, 
                               bins: np.ndarray) -> np.ndarray:
    # Initialize array for binned distances
    binned_distances = np.zeros(bins.shape[0])
    binned_distances_std = np.zeros(bins.shape[0])
    
    # For each bin, calculate mean distance
    for i in range(bins.shape[0]):
        bin_distance = max_distances[binned_data == i]
        if len(bin_distance) > 0:
            binned_distances[i] = np.max(bin_distance)
            binned_distances_std[i] = np.std(bin_distance)
    
    # Replace zero distances with NaN
    binned_distances[binned_distances == 0] = np.nan
    binned_distances_std[binned_distances_std == 0] = np.nan
    
    return binned_distances, binned_distances_std


def interpolate_distances(binned_distances: np.ndarray) -> np.ndarray:
    """
    Interpolates the binned distances to fill in any missing values.

    Parameters:
    binned_distances (np.ndarray): The binned distances to interpolate.

    Returns:
    np.ndarray: The interpolated distances.
    """
    
    indices = np.arange(binned_distances.shape[0])
    
    # Get indices of non-NaN values
    non_nan_indices = indices[~np.isnan(binned_distances)]
    
    # Interpolate to fill NaN values
    interpolated = np.interp(indices, non_nan_indices, binned_distances[non_nan_indices])
    
    return interpolated


def add_expected_value_distance(pcd: np.ndarray, 
                                binned_pcd: np.ndarray, 
                                binned_distance_interp: np.ndarray,
                                binned_distances_interp_std: np.ndarray) -> np.ndarray:
    """
    Adds the expected value of distance to the point cloud data.

    Parameters:
    pcd (np.ndarray): The point cloud data.
    binned_pcd (np.ndarray): The binned point cloud data.
    max_distance (np.ndarray): The maximum distances for each bin.

    Returns:
    np.ndarray: The point cloud data with the expected value of distance added.
    """
    # Calculate the expected value of distance for each point
    expected_value_distance = binned_distance_interp[binned_pcd-1]
    expected_value_distance_std = binned_distances_interp_std[binned_pcd-1]

    # Add the expected value of distance to the point cloud data
    pcd = np.c_[pcd, expected_value_distance, expected_value_distance_std]
    
    return pcd


@njit(parallel=True)
def compute_normals_numba(indices, point_clouds):
    # Initialize an empty array to store the normals
    normals = np.empty((indices.shape[0],3))

    # Loop over all indices in parallel
    for i in prange(indices.shape[0]):
        # Select the i-th point cloud from point_clouds
        point_cloud = point_clouds[i]

        # Compute the covariance matrix of point_cloud and find its eigenvectors
        _, eigenvectors = np.linalg.eigh(np.cov(point_cloud.T))
        
        # The first eigenvector (corresponding to the smallest eigenvalue) is the normal of the point cloud
        normals[i] = eigenvectors[:, 0]          

    return normals


def kdtree_maxdist_normals(cfg, pcd, num_nearest_neighbors=4):
    scanner_pos = np.mean(pcd[:,:3], axis=0)
    pcd_xyz_centered = pcd[:,:3] - scanner_pos
    
    # Build a k-d tree from point_clouds for efficient nearest neighbor search
    kdtree = cKDTree(pcd_xyz_centered)

    # Query the k-d tree for the num_nearest_neighbors nearest neighbors of each point in point_clouds
    distances, indices = kdtree.query(pcd_xyz_centered, k=num_nearest_neighbors, workers=-1)
    
    # Calculate max of num_nearest_neighbors nearest distances, excluding self (index 0)
    max_distances = np.mean(distances[:, 1:num_nearest_neighbors], axis=1)
    
    pcd_xyz = pcd_xyz_centered
    
    if cfg.sce.relocate_origin:
        scanner_pos[2] += cfg.sce.z_offset
        pcd_xyz = pcd[:,:3] - scanner_pos
    
    if not cfg.sce.calculate_normals:
        return max_distances, pcd_xyz, None

    # Select the point clouds corresponding to the indices
    point_clouds = pcd_xyz_centered[indices]

    # Compute the normals of the selected point clouds
    normals = compute_normals_numba(indices, point_clouds)
    
    return max_distances, pcd_xyz, normals, scanner_pos


def align_normals_with_scanner_pos(cfg, pcd, normals):
    # Calculate the orientation of the points with respect to the scanner position
    # -normals_xyz point normals are facing the scanner
    normals_xyz = pcd / np.linalg.norm(pcd, axis=1, keepdims=True)
    
    if not cfg.sce.calculate_normals:
        return -normals_xyz, None

    # Calculate the dot product of each normal with the scanner direction
    dot_product = np.einsum('ij,ij->i', normals, -normals_xyz)

    # Flip the normals where the dot product is negative
    normals[dot_product < 0] *= -1

    return -normals_xyz, normals