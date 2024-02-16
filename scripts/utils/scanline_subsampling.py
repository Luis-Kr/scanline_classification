import numpy as np
from typing import Tuple
from numba import njit, prange, float64, int64
from numba.typed import Dict
from typing import List


@njit()
def segment_subsampling(pcd: np.ndarray, 
                        segment_indices: int,
                        x_col: int,
                        y_col: int,
                        z_col: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsamples a segment of a point cloud.

    This function extracts the x, y, and z columns of the points in a segment of a point cloud,
    and returns the median, minimum, and maximum values of these columns.

    Parameters:
    pcd (np.ndarray): The point cloud array.
    segment_indices (int): The indices of the points in the segment.
    x_col (int): The index of the x column in the pcd. 
    y_col (int): The index of the y column in the pcd. 
    z_col (int): The index of the z column in the pcd.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three 1D arrays. The first array contains the median values of the x, y, and z columns. 
                                               The second array the minimum values and the third the maximum values. 
    """
    # Extract the relevant columns for the segment
    x_segment = pcd[segment_indices, x_col]
    y_segment = pcd[segment_indices, y_col]
    z_segment = pcd[segment_indices, z_col]
    xyz_segment = np.column_stack((x_segment, y_segment, z_segment))
    
    # Create empty arrays to store the nearest neighbor coordinates
    xyz_segment_median = np.zeros((1, 3))   
    xyz_segment_min = np.zeros((1, 3))
    xyz_segment_max = np.zeros((1, 3))
    distances_median = np.zeros(xyz_segment.shape[0])
    distances_min = np.zeros(xyz_segment.shape[0])
    distances_max = np.zeros(xyz_segment.shape[0])

    # Calculate the median, min, and max of the xyz coordinates
    for i in prange(xyz_segment.shape[1]):
        xyz_segment_median[0,i] = np.median(xyz_segment[:, i])
        xyz_segment_min[0,i] = np.nanpercentile(xyz_segment[:, i], 2)
        xyz_segment_max[0,i] = np.nanpercentile(xyz_segment[:, i], 98)    

    # KdTree didn't work with numba, so use Euclidean distance instead
    # Calculate the Euclidean distance from each point to the median, min, and max point
    for i in prange(xyz_segment.shape[0]):
        distances_median[i] = np.linalg.norm(xyz_segment[i,:] - xyz_segment_median)
        distances_min[i] = np.linalg.norm(xyz_segment[i,:] - xyz_segment_min)
        distances_max[i] = np.linalg.norm(xyz_segment[i,:] - xyz_segment_max)

    # Find the index of the point with the smallest distance
    nearest_neighbor_index_median = np.argmin(distances_median)
    nearest_neighbor_index_min = np.argmin(distances_min)
    nearest_neighbor_index_max = np.argmin(distances_max)

    # Get the coordinates of the nearest neighbor
    xyz_segment_median_nn = xyz_segment[nearest_neighbor_index_median, 0:3]
    xyz_segment_min_nn = xyz_segment[nearest_neighbor_index_min, 0:3]
    xyz_segment_max_nn = xyz_segment[nearest_neighbor_index_max, 0:3]

    return xyz_segment_median_nn, xyz_segment_min_nn, xyz_segment_max_nn


@njit()
def count_labels(labels_segment):
    unique_labels = np.unique(labels_segment)
    max_count = 0
    max_label = -1
    for i in prange(unique_labels.shape[0]):
        label = unique_labels[i]
        count = np.sum(labels_segment == label)
        if count > max_count:
            max_count = count
            max_label = label
    return max_label


@njit()
def calculate_skewness(data: np.ndarray) -> float:
    # Calculate mean and standard deviation
    mean = np.nanmean(data)
    std_dev = np.nanstd(data)

    # Calculate skewness
    # for small segments, the standard deviation can be zero (e.g., only one point in the segment)
    if std_dev == 0:
        skewness = 0
    else:
        skewness = np.nanmean(((data - mean) / std_dev) ** 3)
        
    return skewness


@njit()
def calculate_attributes(segment):
    attributes = np.zeros(7)
    attributes[0] = np.nanmean(segment)
    attributes[1] = np.nanvar(segment)
    attributes[2] = np.nanstd(segment)
    attributes[3] = np.nanmedian(segment)
    attributes[4] = np.nanpercentile(segment, 2)
    attributes[5] = np.nanpercentile(segment, 98)
    attributes[6] = calculate_skewness(segment)
    return attributes


@njit()
def calculate_segment_attributes(pcd: np.ndarray, 
                                     segment_indices: np.ndarray,
                                     label_col: int,
                                     columns) -> np.ndarray:
    # Create an empty array to store the attributes
    segment_attributes = np.zeros((1, (len(columns) * 7)+1))

    # Calculate the attributes for each column
    for i, col in enumerate(columns):
        segment = pcd[segment_indices, col]
        segment_attributes[0, i*7:i*7+7] = calculate_attributes(segment)

    # Count labels
    labels_segment = pcd[segment_indices, label_col]
    segment_attributes[0, -1] = count_labels(labels_segment)

    return segment_attributes



@njit()
def combine_segment_attributes(xyz_segment_median_nn: np.ndarray, 
                               xyz_segment_min_nn: np.ndarray, 
                               xyz_segment_max_nn: np.ndarray, 
                               segment_attributes: np.ndarray) -> np.ndarray:
    """
    Combine centroid, min and max xyz positions and segment attributes into a single numpy array.

    Parameters:
    xyz_segment_median_nn (np.ndarray): The nearest neighbor of the median point of the segment.
    xyz_segment_min_nn (np.ndarray): The nearest neighbor of the minimum point of the segment.
    xyz_segment_max_nn (np.ndarray): The nearest neighbor of the maximum point of the segment.
    segment_attributes (np.ndarray): The existing segment attributes.

    Returns:
    np.ndarray: A numpy array containing the combined segment attributes.
    """  
    return np.hstack((xyz_segment_median_nn, 
                      xyz_segment_min_nn, 
                      xyz_segment_max_nn, 
                      segment_attributes.ravel()))
    
    
@njit(parallel=True)
def process_segments(pcd: np.ndarray, 
                     segment_classes: np.ndarray, 
                     processed_segments: np.ndarray,
                     counts: np.ndarray,
                     x_col: int,
                     y_col: int,
                     z_col: int,
                     column_indices,
                     label_col: int,
                     segment_ids_col: int) -> np.ndarray:
    # Sort the point cloud by segment id
    sorted_indices = np.argsort(pcd[:,segment_ids_col])
    
    # Split the sorted indices into segments
    indices_per_class = np.split(sorted_indices, np.cumsum(counts[:-1]))
    
    # Process each segment
    for i in prange(segment_classes.shape[0]):
        # Get the indices for the current segment
        segment_indices = indices_per_class[i]
        
        # Calculate the nearest neighbor point for the median, min, and max location of the segment
        xyz_segment_median_nn, xyz_segment_min_nn, xyz_segment_max_nn = segment_subsampling(pcd=pcd, 
                                                                                            segment_indices=segment_indices,
                                                                                            x_col=x_col,
                                                                                            y_col=y_col,
                                                                                            z_col=z_col)
        
        segment_attributes = calculate_segment_attributes(pcd=pcd,
                                                          segment_indices=segment_indices,
                                                          label_col=label_col,
                                                          columns=column_indices)
        
        # Add the combined attributes to the array
        processed_segments[i] = combine_segment_attributes(xyz_segment_median_nn, 
                                                           xyz_segment_min_nn, 
                                                           xyz_segment_max_nn, 
                                                           segment_attributes)
    
    return processed_segments