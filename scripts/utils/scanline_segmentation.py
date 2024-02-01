import numpy as np
from typing import Tuple
from numba import njit, prange


def sort_scanline(pcd: np.ndarray, 
                  scanline_id_col: int,
                  vert_angle_col: int) -> np.ndarray:
    """
    Sorts a point cloud (pcd) array based on a specified column.

    Parameters:
    pcd (np.ndarray): The point cloud array.
    col (int): The column index to sort by.

    Returns:
    np.ndarray: The sorted point cloud data array.
    """
    # Sort the pcd by the specified column and return the sorted pcd
    sort_idx = np.lexsort(np.rot90(pcd[:,(scanline_id_col, 
                                          vert_angle_col)]))
    
    return pcd[sort_idx], sort_idx


def get_scanline_intervals(pcd: np.ndarray,
                           scanline_id_col: int) -> np.ndarray:
    """
    Extracts the unique scanline ids. 

    Parameters:
    pcd (np.ndarray): The point cloud array.
    col (int): The column index to extract the ids from.

    Returns:
    np.ndarray: An array of the unique ids.
    """
    # Calculate the difference between consecutive elements
    diff = np.diff(pcd[:, scanline_id_col])

    # Find the indices where the difference is greater than 0
    scanline_intervals = np.where(diff > 0)[0] + 1

    # add the first index
    scanline_intervals = np.insert(scanline_intervals, 0, 0)

    # add the last index
    scanline_intervals = np.append(scanline_intervals, pcd.shape[0])
    
    return scanline_intervals


@njit()
def get_scanline(pcd: np.ndarray, 
                 lower_boundary: int, 
                 upper_boundary: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a scanline from a point cloud.

    Parameters:
    pcd (np.ndarray): The point cloud array.
    col (int): The column index to look for the id.
    id (int): The id to look for in the specified column.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the extracted scanline and its indices in the original pcd.
    """
    scanline = pcd[lower_boundary:upper_boundary]
    scanline_indices = np.arange(lower_boundary, upper_boundary)
    
    return scanline, scanline_indices


@njit()
def calculate_rho_diff(pcd: np.ndarray, 
                       col: int) -> np.ndarray:
    """
    Calculates the absolute difference between adjacent elements in a specified column of a point cloud array.

    Parameters:
    pcd (np.ndarray): The point cloud array.
    col (int): The column index to calculate the difference on.

    Returns:
    np.ndarray: An array of the absolute differences between adjacent elements in the specified column.
    """
    # Calculate the absolute difference between adjacent elements
    rho_diff = np.abs(np.diff(np.ascontiguousarray(pcd[:, col])))

    # Append the last element of rho_diff to itself to maintain the same length as the input pcd
    rho_diff = np.append(rho_diff, rho_diff[-1])
    
    return rho_diff


@njit()
def pad_reflect(arr: np.ndarray, 
                pad_width: int) -> np.ndarray:
    """
    Function to pad an array using the 'reflect' mode (numpy.pad replacement). 
    
    Parameters:
    arr (np.ndarray): The input array to be padded.
    pad_width (int): The number of elements by which to pad the array.
    
    Returns:
    np.ndarray: The padded array.
    """
    return np.concatenate((arr[pad_width:0:-1], arr, arr[-2:-pad_width-2:-1]))


@njit()
def get_slope_3D(points_left_side: np.ndarray,
                 points_right_side: np.ndarray) -> np.ndarray:
    
    # Get the z-differences between the points
    z_diff = points_right_side[:, 2] - points_left_side[:, 2]
    
    # Get the 3D distance based on the x,y,z coordinates
    dist_3d = np.sqrt(((points_right_side[:, 0] - points_left_side[:, 0]) ** 2) + 
                      ((points_right_side[:, 1] - points_left_side[:, 1]) ** 2) + 
                      (z_diff ** 2))
    
    # Calculate the slope in radians
    local_slope = np.arctan2(z_diff, dist_3d)
    
    # Calculate the slope in degrees
    local_slope_deg = np.rad2deg(local_slope)
    
    return np.abs(local_slope_deg)


@njit()
def calculate_slope(scanline: np.ndarray,
                    num_neighbors: np.ndarray,
                    x_col: int,
                    y_col: int,
                    z_col: int) -> np.ndarray:
    """
    Calculates the slope between each point and its two neighbors (central differences) in a scanline.

    Parameters:
    scanline (np.ndarray): The scanline, a 2D array where each row is a point.
    x_col (int): The column index of the x coordinates.
    y_col (int): The column index of the y coordinates.
    z_col (int): The column index of the z coordinates.

    Returns:
    np.ndarray: An array of the absolute slopes between each point and its neighbors.
    """
    
    # Extract the x, y, z coordinates from the scanline
    x = scanline[:, x_col]
    y = scanline[:, y_col]
    z = scanline[:, z_col]
    
    # Merge the x, y, z coordinates into a single array
    scanline_xyz = np.column_stack((x, y, z))
    
    # Calculate the maximum number of neighbors to consider
    num_max_neighbors = int(np.max(num_neighbors))  
    
    # Pad the array at the beginning and end
    padded_scanline = pad_reflect(scanline_xyz, num_max_neighbors)

    # Calculate the slope for each point in the scanline
    slope = np.zeros(scanline.shape[0])
    for idx in range(scanline.shape[0]):
        n = int(idx + num_max_neighbors)
        slope[idx] = np.mean(get_slope_3D(points_left_side=padded_scanline[n-num_neighbors[idx]:n], 
                                  points_right_side=padded_scanline[n+1:n+num_neighbors[idx]+1]))

    return slope


@njit()
def slope_lstsq_local_neighborhood(points_left_side: np.ndarray, 
                                   points_right_side: np.ndarray) -> np.ndarray: 
    
    """2D case with rho and z-coordinates"""
    # Merge the left and right side points into a single array
    neighborhood_points = np.concatenate((points_left_side, points_right_side))
    X = neighborhood_points[:, 0]
    
    # Calculate the least-squares solution
    A = np.column_stack((X, np.ones(neighborhood_points.shape[0])))
    B = neighborhood_points[:, 1]
    
    lstsq_solution, _, _, _ = np.linalg.lstsq(A, B)
    
    # Convert the angle to degrees
    slope_deg = np.rad2deg(np.abs(np.arctan(lstsq_solution[0])))
    
    return slope_deg


@njit()
def calculate_slope_least_squares(scanline: np.ndarray,
                                  rho_col: int,
                                  z_col: int,
                                  num_neighbors: np.ndarray) -> np.ndarray:
    # Extract the x, y, z coordinates from the scanline (numba does not support indexing with multiple columns)
    x = scanline[:, rho_col]
    y = scanline[:, z_col]
    
    # Merge the x, y, z coordinates into a single array
    scanline_xyz = np.column_stack((x, y))
    
    # Calculate the maximum number of neighbors to consider
    num_max_neighbors = int(np.max(num_neighbors))  
    
    # Pad the array at the beginning and end
    padded_scanline = pad_reflect(scanline_xyz, num_max_neighbors)
    
    # Calculate the slope
    slope = np.zeros(scanline.shape[0])
    for idx in range(scanline.shape[0]):
        n = int(idx + num_max_neighbors)
        slope[idx] = slope_lstsq_local_neighborhood(points_left_side=padded_scanline[n-num_neighbors[idx]:n], 
                                                    points_right_side=padded_scanline[n+1:n+num_neighbors[idx]+1])
    
    return slope



@njit()
def calculate_curvature(slope_arr: np.ndarray, 
                        num_neighbors: np.ndarray) -> np.ndarray:
    """
    Function to calculate the curvature between elements in an array and their neighbors.
    
    Parameters:
    arr (np.ndarray): The input array.
    num_neighbors (np.ndarray): The number of neighbors to consider on each side for each element in arr.
    
    Returns:
    np.ndarray: An array of the calculated differences.
    """
    # Calculate the maximum number of neighbors to consider
    num_max_neighbors = int(np.max(num_neighbors))
    
    # Pad the array at the beginning and end
    pad_arr = pad_reflect(slope_arr, num_max_neighbors)

    # Initialize an empty array to store the differences
    curvature = np.zeros(slope_arr.shape[0])
    
    # Loop over each point in the original array
    for idx in range(slope_arr.shape[0]):
        i = idx + num_max_neighbors
        # Calculate the differences
        curvature[idx] = np.mean(np.abs(pad_arr[i+1:i+num_neighbors[idx]+1] - pad_arr[i-num_neighbors[idx]:i][::-1]))

    return curvature


@njit()
def calculate_distances_point_lines(center_point: np.ndarray, 
                                    points_left_side: np.ndarray, 
                                    points_right_side: np.ndarray) -> np.ndarray:
    a = points_right_side - points_left_side
    vec_p1_p0 = center_point - points_left_side
    distance = np.zeros(vec_p1_p0.shape[0])
    
    for v in range(vec_p1_p0.shape[0]):
        dproduct_direction_vector = np.dot(a[v], a[v])
        if dproduct_direction_vector != 0:
            t = np.dot(vec_p1_p0[v], a[v]) / dproduct_direction_vector
            l1 = points_left_side[v] + t*a[v]
            distance[v] = np.linalg.norm(l1 - center_point)
        else:
            # ZeroDivisionError occurs because points are too close together (identical points) 
            #print('ZeroDivisionError')
            distance[v] = 0
            
    return distance


@njit()
def calculate_roughness(scanline: np.ndarray,
                        num_neighbors: np.ndarray,
                        x_col: int,
                        y_col: int,
                        z_col: int) -> np.ndarray:
    """
    Function to calculate the roughness of a scanline with variable number of neighbors.
    
    Parameters:
    scanline (np.ndarray): The input scanline.
    num_neighbors (np.ndarray): The number of neighbors to consider on each side for each point in the scanline.
    
    Returns:
    np.ndarray: An array of the calculated roughness.
    """ 
    # Extract the x, y, z coordinates from the scanline (numba does not support indexing with multiple columns)
    x = scanline[:, x_col]
    y = scanline[:, y_col]
    z = scanline[:, z_col]
    
    # Merge the x, y, z coordinates into a single array
    scanline_xyz = np.column_stack((x, y, z))
    
    # Calculate the maximum number of neighbors to consider
    num_max_neighbors = int(np.max(num_neighbors))  
    
    # Pad the array at the beginning and end
    padded_scanline = pad_reflect(scanline_xyz, num_max_neighbors)
    
    # Calculate the roughness
    roughness = np.zeros(scanline.shape[0])
    for idx in range(scanline.shape[0]):
        n = int(idx + num_max_neighbors)
        roughness[idx] = np.nanvar(calculate_distances_point_lines(center_point=padded_scanline[n], 
                                                                   points_left_side=padded_scanline[n-num_neighbors[idx]:n], 
                                                                   points_right_side=padded_scanline[n+1:n+num_neighbors[idx]+1]))
    
    return roughness


@njit(parallel=True)
def calculate_segmentation_metrics(pcd: np.ndarray,
                                   scanline_intervals: np.ndarray,
                                   x_col: int,
                                   y_col: int,
                                   z_col: int,
                                   expected_value_col: int,
                                   rho_col: int,
                                   least_squares_method: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the segmentation metrics (rho_diff, slope and curvature) for each scanline.

    Parameters:
    pcd (np.ndarray): The point cloud array.
    x_col (int): The column index containing the x coordinates.
    y_col (int): The column index containing the y coordinates.
    z_col (int): The column index containing the z coordinates.
    sort_col (int): The column index to sort the pcd by.
    scanline_id_col (int): The column index to extract the scanline ids from.
    rho_col (int): The column index containing the rho values.

    Returns:
    tuple: A tuple of three arrays, each containing one of the segmentation metrics (rho_diff, slope, curvature, sorted pcd) for the pcd.
    """
    # Create empty arrays to store the segmentation metrics
    rho_diff = np.zeros(pcd.shape[0])
    slope = np.zeros(pcd.shape[0])
    curvature = np.zeros(pcd.shape[0])
    roughness = np.zeros(pcd.shape[0])
    
    # Calculate the segmentation metrics for each scanline
    for i in prange(scanline_intervals.shape[0]-1):
        # Extract the current scanline and its indices in the pcd
        scanline, scanline_indices = get_scanline(pcd=pcd, 
                                                  lower_boundary=scanline_intervals[i], 
                                                  upper_boundary=scanline_intervals[i+1])
        
        density = 1 / scanline[:, expected_value_col]
        
        # Smoothing case
        k_neighbors = np.ceil(np.sqrt(density))
        
        # If k_neighbors is 1, set it to 2 to avoid too small neighborhoods
        k_neighbors[k_neighbors == 1] = 2
        
        # Smoothing with constant k
        #k_neighbors = np.ones(scanline.shape[0]) * 5
        
        # No smoothing case
        #k_neighbors = np.ones(scanline.shape[0])
        
        # Calculate the rho_diff, slope, curvature, and roughness for the current scanline
        rho_diff_i = calculate_rho_diff(scanline, 
                                        col=rho_col)
        
        if least_squares_method:
            slope_i = calculate_slope_least_squares(scanline=scanline,
                                                    num_neighbors=k_neighbors,
                                                    rho_col=rho_col,
                                                    z_col=z_col)
        else:
            slope_i = calculate_slope(scanline=scanline, 
                                      num_neighbors=k_neighbors,
                                      x_col=x_col, 
                                      y_col=y_col, 
                                      z_col=z_col)
        
        curvature_i = calculate_curvature(slope_arr=slope_i, 
                                          num_neighbors=k_neighbors)
        
        roughness_i = calculate_roughness(scanline=scanline, 
                                          num_neighbors=k_neighbors,
                                          x_col=x_col,
                                          y_col=y_col,
                                          z_col=z_col)
        
        rho_diff[scanline_indices] = rho_diff_i
        slope[scanline_indices] = slope_i
        curvature[scanline_indices] = curvature_i
        roughness[scanline_indices] = roughness_i
                
        # # Store the calculated metrics in the corresponding arrays
        # for j in prange(scanline.shape[0]):
        #     rho_diff[scanline_indices[j]] = rho_diff_i[j]
        #     slope[scanline_indices[j]] = slope_i[j]
        #     curvature[scanline_indices[j]] = curvature_i[j]
        #     roughness[scanline_indices[j]] = roughness_i[j]

    return rho_diff, slope, curvature, roughness


@njit(parallel=True)
def scanline_segmentation(pcd: np.ndarray, 
                          expected_value_col: int, 
                          rho_diff_col: int, 
                          slope_col: int, 
                          curvature_col: int,
                          expected_value_factor: int,
                          slope_threshold: float,
                          curvature_threshold: float) -> np.ndarray:
    """
    Segments a point cloud based on rho difference, slope and curvature conditions.

    Parameters:
    pcd (np.ndarray): The point cloud array.
    expected_value_col (int): The index of the column in the pcd that contains the expected rho values.
    rho_diff_col (int): The index of the column in the pcd that contains the rho_diff values.
    slope_col (int): The index of the column in the pcd that contains the slope values.
    curvature_col (int): The index of the column in the pcd that contains the curvature values.
    expected_value_factor (int): The factor to multiply the expected value with.
    slope_threshold (float): The threshold for the slope.
    curvature_threshold (float): The threshold for the curvature.

    Returns:
    np.ndarray: An array of the segment ids for each point in the pcd.
    """
    
    # Initialize an array of zeros to store the segment ids
    segment_ids = np.zeros(pcd.shape[0])
    
    # Identify the segments based on the conditions
    segments = np.where((pcd[:,rho_diff_col] > pcd[:,expected_value_col]*expected_value_factor) | 
                        (pcd[:,curvature_col] > curvature_threshold))[0] #(pcd[:,slope_col] < slope_threshold) |
    
    # Assign the segment ids to the points
    for i in prange(pcd.shape[0]):
        segment_ids[i] = np.searchsorted(segments, i, side='left')
    
    # Increment the segment ids by 1 to start from 1 instead of 0
    segment_ids += 1
    
    return segment_ids
