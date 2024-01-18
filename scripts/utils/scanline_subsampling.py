import numpy as np
from typing import Tuple
from numba import njit, prange


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
        xyz_segment_min[0,i] = np.min(xyz_segment[:, i])
        xyz_segment_max[0,i] = np.max(xyz_segment[:, i])    

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
def calculate_segment_attributes(pcd: np.ndarray, 
                                 segment_indices: np.ndarray,
                                 height_col: int,
                                 intensity_col: int,
                                 red_col: int,
                                 green_col: int,
                                 blue_col: int,
                                 rho_col: int,
                                 slope_col: int,
                                 curvature_col: int,
                                 normals_xyz_col: np.ndarray,
                                 normals_col: np.ndarray) -> np.ndarray:
    """
    Calculate attributes for a specific segment in a point cloud data array.

    This function extracts various attributes (height, reflectance, color, rho, slope, curvature) 
    for the points in a given segment of a point cloud and calculates their mean, variance, 
    standard deviation, median, minimum and maximum values.

    Parameters:
    pcd (np.ndarray): The point cloud data array.
    segment_indices (np.ndarray): The indices of the points in the segment.
    height_col (int): The index of the height column in the pcd. 
    reflectance_col (int): The index of the reflectance column. 
    red_col (int): The index of the red color column. 
    green_col (int): The index of the green color column. 
    blue_col (int): The index of the blue color column. 
    rho_col (int): The index of the rho column. 
    slope_col (int): The index of the slope column. 
    curvature_col (int): The index of the curvature column. 
    normals_xyz_col (list): The indices of the xyz columns of the normals.
    normals_col (list): The indices of the normals.

    Returns:
    np.ndarray: A 1D array containing the calculated attributes for the segment.
    """
    # Create an empty dictionary to store the attributes
    segment_attributes = np.zeros((1, 90))
    
    # Extract the relevant columns for the segment
    height_segment = pcd[segment_indices, height_col]
    reflectance_segment = pcd[segment_indices, intensity_col]
    red_segment = pcd[segment_indices, red_col]
    green_segment = pcd[segment_indices, green_col]
    blue_segment = pcd[segment_indices, blue_col]
    rho_segment = pcd[segment_indices, rho_col]
    slope_segment = pcd[segment_indices, slope_col]
    curvature_segment = pcd[segment_indices, curvature_col]
    nx_xyz = pcd[segment_indices, normals_xyz_col[0]]
    ny_xyz = pcd[segment_indices, normals_xyz_col[1]]
    nz_xyz = pcd[segment_indices, normals_xyz_col[2]]
    nx = pcd[segment_indices, normals_col[0]]
    ny = pcd[segment_indices, normals_col[1]]
    nz = pcd[segment_indices, normals_col[2]]
    
    # Calculate the segment attributes
    segment_attributes[0, 0] = np.nanmean(height_segment)
    segment_attributes[0, 1] = np.nanvar(height_segment)
    segment_attributes[0, 2] = np.nanstd(height_segment)
    segment_attributes[0, 3] = np.nanmedian(height_segment)
    segment_attributes[0, 4] = np.nanmin(height_segment)
    segment_attributes[0, 5] = np.nanmax(height_segment)

    segment_attributes[0, 6] = np.nanmean(reflectance_segment)
    segment_attributes[0, 7] = np.nanvar(reflectance_segment)
    segment_attributes[0, 8] = np.nanstd(reflectance_segment)
    segment_attributes[0, 9] = np.nanmedian(reflectance_segment)
    segment_attributes[0, 10] = np.nanmin(reflectance_segment)
    segment_attributes[0, 11] = np.nanmax(reflectance_segment)

    segment_attributes[0, 12] = np.nanmean(red_segment)
    segment_attributes[0, 13] = np.nanvar(red_segment)
    segment_attributes[0, 14] = np.nanstd(red_segment)
    segment_attributes[0, 15] = np.nanmedian(red_segment)
    segment_attributes[0, 16] = np.nanmin(red_segment)
    segment_attributes[0, 17] = np.nanmax(red_segment)

    segment_attributes[0, 18] = np.nanmean(green_segment)
    segment_attributes[0, 19] = np.nanvar(green_segment)
    segment_attributes[0, 20] = np.nanstd(green_segment)
    segment_attributes[0, 21] = np.nanmedian(green_segment)
    segment_attributes[0, 22] = np.nanpercentile(green_segment, 2)
    segment_attributes[0, 23] = np.nanpercentile(green_segment, 98)

    segment_attributes[0, 24] = np.nanmean(blue_segment)
    segment_attributes[0, 25] = np.nanvar(blue_segment)
    segment_attributes[0, 26] = np.nanstd(blue_segment)
    segment_attributes[0, 27] = np.nanmedian(blue_segment)
    segment_attributes[0, 28] = np.nanmin(blue_segment)
    segment_attributes[0, 29] = np.nanmax(blue_segment)

    segment_attributes[0, 30] = np.nanmean(rho_segment)
    segment_attributes[0, 31] = np.nanvar(rho_segment)
    segment_attributes[0, 32] = np.nanstd(rho_segment)
    segment_attributes[0, 33] = np.nanmedian(rho_segment)
    segment_attributes[0, 34] = np.nanmin(rho_segment)
    segment_attributes[0, 35] = np.nanmax(rho_segment)

    segment_attributes[0, 36] = np.nanmean(slope_segment)
    segment_attributes[0, 37] = np.nanvar(slope_segment)
    segment_attributes[0, 38] = np.nanstd(slope_segment)
    segment_attributes[0, 39] = np.nanmedian(slope_segment)
    segment_attributes[0, 40] = np.nanmin(slope_segment)
    segment_attributes[0, 41] = np.nanmax(slope_segment)

    segment_attributes[0, 42] = np.nanmean(curvature_segment)
    segment_attributes[0, 43] = np.nanvar(curvature_segment)
    segment_attributes[0, 44] = np.nanstd(curvature_segment)
    segment_attributes[0, 45] = np.nanmedian(curvature_segment)
    segment_attributes[0, 46] = np.nanmin(curvature_segment)
    segment_attributes[0, 47] = np.nanmax(curvature_segment)
    
    segment_attributes[0, 48] = np.nanmean(nx_xyz)
    segment_attributes[0, 49] = np.nanvar(nx_xyz)
    segment_attributes[0, 50] = np.nanstd(nx_xyz)
    segment_attributes[0, 51] = np.nanmedian(nx_xyz)
    segment_attributes[0, 52] = np.nanmin(nx_xyz)
    segment_attributes[0, 53] = np.nanmax(nx_xyz)
    segment_attributes[0, 54] = np.ptp(nx_xyz)
    
    segment_attributes[0, 55] = np.nanmean(ny_xyz)
    segment_attributes[0, 56] = np.nanvar(ny_xyz)
    segment_attributes[0, 57] = np.nanstd(ny_xyz)
    segment_attributes[0, 58] = np.nanmedian(ny_xyz)
    segment_attributes[0, 59] = np.nanmin(ny_xyz)
    segment_attributes[0, 60] = np.nanmax(ny_xyz)
    segment_attributes[0, 61] = np.ptp(ny_xyz)
    
    segment_attributes[0, 62] = np.nanmean(nz_xyz)
    segment_attributes[0, 63] = np.nanvar(nz_xyz)
    segment_attributes[0, 64] = np.nanstd(nz_xyz)
    segment_attributes[0, 65] = np.nanmedian(nz_xyz)
    segment_attributes[0, 66] = np.nanmin(nz_xyz)
    segment_attributes[0, 67] = np.nanmax(nz_xyz)
    segment_attributes[0, 68] = np.ptp(nz_xyz)
    
    segment_attributes[0, 69] = np.nanmean(nx)
    segment_attributes[0, 70] = np.nanvar(nx)
    segment_attributes[0, 71] = np.nanstd(nx)
    segment_attributes[0, 72] = np.nanmedian(nx)
    segment_attributes[0, 73] = np.nanmin(nx)
    segment_attributes[0, 74] = np.nanmax(nx)
    segment_attributes[0, 75] = np.ptp(nx)
    
    segment_attributes[0, 76] = np.nanmean(ny)
    segment_attributes[0, 77] = np.nanvar(ny)
    segment_attributes[0, 78] = np.nanstd(ny)
    segment_attributes[0, 79] = np.nanmedian(ny)
    segment_attributes[0, 80] = np.nanmin(ny)
    segment_attributes[0, 81] = np.nanmax(ny)
    segment_attributes[0, 82] = np.ptp(ny)
    
    segment_attributes[0, 83] = np.nanmean(nz)
    segment_attributes[0, 84] = np.nanvar(nz)
    segment_attributes[0, 85] = np.nanstd(nz)
    segment_attributes[0, 86] = np.nanmedian(nz)
    segment_attributes[0, 87] = np.nanmin(nz)
    segment_attributes[0, 88] = np.nanmax(nz)
    segment_attributes[0, 89] = np.ptp(nz)

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
                     x_col: int=0,
                     y_col: int=1,
                     z_col: int=2,
                     height_col: int=2,
                     intensity_col: int=3,
                     red_col: int=4,
                     green_col: int=5,
                     blue_col: int=6,
                     rho_col: int=7,
                     slope_col: int=13,
                     curvature_col: int=14,
                     segment_ids_col: int=15,
                     normals_xyz_col: np.ndarray=np.array([16,17,18]),
                     normals_col: np.ndarray=np.array([19,20,21])) -> np.ndarray:
    """
    Subsample each segment in a point cloud array and calculate the segment attributes.

    Parameters:
    pcd (np.ndarray): The point cloud data array.
    segment_classes (np.ndarray): An array containing the class of each segment.
    processed_segments (np.ndarray): An array to store the processed segments.
    counts (np.ndarray): An array containing the count of points in each segment.
    x_col (int, optional): The index of the x column in the pcd. Defaults to 0.
    y_col (int, optional): The index of the y column. Defaults to 1.
    z_col (int, optional): The index of the z column. Defaults to 2.
    height_col (int, optional): The index of the height column. Defaults to 2.
    reflectance_col (int, optional): The index of the reflectance column. Defaults to 3.
    red_col (int, optional): The index of the red color column. Defaults to 4.
    green_col (int, optional): The index of the green color column. Defaults to 5.
    blue_col (int, optional): The index of the blue color column. Defaults to 6.
    rho_col (int, optional): The index of the rho column. Defaults to 7.
    slope_col (int, optional): The index of the slope column. Defaults to 13.
    curvature_col (int, optional): The index of the curvature column. Defaults to 14.
    segment_ids_col (int, optional): The index of the segment ids column. Defaults to 15.
    normals_xyz_col (list, optional): The indices of the xyz columns of the normals. Defaults to [16,17,18].
    normals_col (list, optional): The indices of the normals. Defaults to [19,20,21].


    Returns:
    np.ndarray: A numpy array containing the processed segment attributes.
    """
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

        # Calculate the attributes for the current segment
        segment_attributes = calculate_segment_attributes(pcd=pcd, 
                                                          segment_indices=segment_indices,
                                                          height_col=height_col,
                                                          intensity_col=intensity_col,
                                                          red_col=red_col,
                                                          green_col=green_col,
                                                          blue_col=blue_col,
                                                          rho_col=rho_col,
                                                          slope_col=slope_col,
                                                          curvature_col=curvature_col,
                                                          normals_xyz_col=normals_xyz_col,
                                                          normals_col=normals_col)
        
        # Add the combined attributes to the array
        processed_segments[i] = combine_segment_attributes(xyz_segment_median_nn, 
                                                           xyz_segment_min_nn, 
                                                           xyz_segment_max_nn, 
                                                           segment_attributes)
    
    return processed_segments