import numpy as np 
from typing import Tuple
from numba import njit, prange, int64, float64
from scipy.spatial import cKDTree


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
    
    # Calculate mean of k nearest distances, excluding self (index 0)
    mean_distances = np.mean(distances[:, 1:k], axis=1)
    
    # Replace zero distances with a small number
    mean_distances = np.where(mean_distances == 0, 0.000001, mean_distances)
    
    return mean_distances, indices


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


def calculate_binned_distances(mean_distances: np.ndarray, 
                               binned_data: np.ndarray, 
                               bins: np.ndarray) -> np.ndarray:
    """
    Calculates the mean distance for each bin.

    Parameters:
    mean_distances (np.ndarray): The mean distances to calculate the binned distances from.
    binned_data (np.ndarray): The binned data.
    bins (np.ndarray): The bins.

    Returns:
    np.ndarray: The binned distances.
    """
    # Initialize array for binned distances
    binned_distances = np.zeros(len(bins))
    
    # For each bin, calculate mean distance
    for i in range(bins.shape[0]):
        bin_distance = mean_distances[binned_data == i]
        if len(bin_distance) > 0:
            binned_distances[i] = np.mean(bin_distance)
    
    # Replace zero distances with NaN
    binned_distances[binned_distances == 0] = np.nan
    
    return binned_distances


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
                                binned_distance_interp: np.ndarray) -> np.ndarray:
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

    # Add the expected value of distance to the point cloud data
    pcd = np.c_[pcd, expected_value_distance]
    
    return pcd