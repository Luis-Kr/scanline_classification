import numpy as np
from typing import Tuple
from numba import jit, njit, prange


@njit()
def sort_scanline(pcd: np.ndarray, 
                  col: int) -> np.ndarray:
    """
    Sorts a point cloud (pcd) array based on a specified column.

    Parameters:
    pcd (np.ndarray): The point cloud array.
    col (int): The column index to sort by.

    Returns:
    np.ndarray: The sorted point cloud data array.
    """
    # Sort the pcd by the specified column and return the sorted pcd
    sort_idx = np.argsort(pcd[:, col])
    return pcd[sort_idx], sort_idx


@njit()
def get_scanline_ids(pcd: np.ndarray,
                     col: int) -> np.ndarray:
    """
    Extracts the unique scanline ids. 

    Parameters:
    pcd (np.ndarray): The point cloud array.
    col (int): The column index to extract the ids from.

    Returns:
    np.ndarray: An array of the unique ids.
    """
    
    return np.unique(pcd[:, col])


@njit()
def get_scanline(pcd: np.ndarray, 
                 col: int, 
                 id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a scanline from a point cloud.

    Parameters:
    pcd (np.ndarray): The point cloud array.
    col (int): The column index to look for the id.
    id (int): The id to look for in the specified column.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the extracted scanline and its indices in the original pcd.
    """
    
    # Find the indices where the specified column equals the specified id
    scanline_indices = np.where(pcd[:, col] == id)[0]
    
    # Extract the scanline from the pcd
    scanline = pcd[scanline_indices]
    
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
def get_dist_3D(x1: float, 
                y1: float, 
                z1: float, 
                x2: float, 
                y2: float, 
                z2: float) -> float:
    """
    Calculates the Euclidean distance between two points in 3D space.

    Parameters:
    x1, y1, z1 (float): The x, y, z coordinates of the first point.
    x2, y2, z2 (float): The x, y, z coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    
    return np.sqrt(np.abs(((x2 - x1) ** 2) + ((y2 - y1) ** 2)) + ((z2 - z1) ** 2))


@njit()
def get_slope_3D(x1: float, 
                 y1: float, 
                 z1: float, 
                 x2: float, 
                 y2: float, 
                 z2: float, 
                 deg: bool=True) -> float:
    """
    Calculates the slope between two points in 3D space.

    Parameters:
    x1, y1, z1 (float): The x, y, z coordinates of the first point.
    x2, y2, z2 (float): The x, y, z coordinates of the second point.
    deg (bool): If True, the slope is returned in degrees. If False, the slope is returned in radians.

    Returns:
    float: The slope between the two points. If the distance between the points is zero, returns NaN.
    """
    
    # Calculate the Euclidean distance between the two points
    dist = get_dist_3D(x1, y1, z1, x2, y2, z2)
    
    # If the distance is zero, return NaN
    if dist == 0:
        return np.nan
    else:
        # If deg is True, return the slope in degrees
        if deg:
            return np.round(np.rad2deg((z1 - z2) / dist), 4)
        # If deg is False, return the slope in radians
        else:
            return np.round((z1 - z2) / dist, 4)

 
@njit()
def calculate_slope(scanline: np.ndarray,
                    x_col: int=0,
                    y_col: int=1,
                    z_col: int=2) -> np.ndarray:
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

    # Initialize an array of zeros to store the slopes
    slope = np.zeros(scanline.shape[0])

    # Calculate the slope for each point in the scanline
    for i in range(scanline.shape[0]):
        # For the first point, calculate the slope with the next point
        if i == 0:
            slope[i] = get_slope_3D(x[i], y[i], z[i], x[i+1], y[i+1], z[i+1])
        # For the middle points, calculate the slope with the previous and next points
        elif i <= scanline.shape[0]-2:
            slope[i] = get_slope_3D(x[i-1], y[i-1], z[i-1], x[i+1], y[i+1], z[i+1])
        # For the last point, calculate the slope with the previous point
        else:
            slope[i] = get_slope_3D(x[i-1], y[i-1], z[i-1], x[i], y[i], z[i])

    return np.abs(slope)


@njit()
def calculate_curvature(slope: np.ndarray) -> np.ndarray:
    """
    Calculates the curvature of a scanline based on its slope.

    Parameters:
    slope (np.ndarray): An array of the slopes of the scanline.

    Returns:
    np.ndarray: An array of the curvatures of the scanline. The first and last values are set to NaN.
    """
    
    # Calculate the difference between adjacent slopes
    slope_diff = np.roll(slope, -1) - np.roll(slope, 1)
    
    # Calculate the curvature as the absolute value of the slope difference
    curvature = np.abs(slope_diff)

    # Set the first and last values of the curvature to NaN, as they are not correct due to the roll operation
    curvature[0] = np.nan
    curvature[-1] = np.nan

    return curvature


@njit()
def orient_cross_product(vector_ba: np.ndarray, 
                         vector_bc: np.ndarray, 
                         viewpoint: np.ndarray) -> np.ndarray:
    """
    Orient the cross product (normal) of two vectors towards a viewpoint.

    Parameters:
    vector_ba (np.ndarray): The first vector.
    vector_bc (np.ndarray): The second vector.
    viewpoint (np.ndarray): The viewpoint.

    Returns:
    np.ndarray: The oriented cross product of the two vectors.
    """
    # Calculate the cross product of the two vectors
    cross_product = np.cross(vector_ba, vector_bc)
    
    # Calculate the vector from the cross product to the viewpoint
    vector_to_viewpoint = viewpoint - cross_product
    
    # If the cross product is not oriented towards the viewpoint, flip it
    if np.dot(cross_product, vector_to_viewpoint) < 0:
        cross_product = np.cross(vector_bc, vector_ba)
    
    return cross_product


@njit()
def compute_normals(pcd_scanline, scanner_pos, x_col, y_col, z_col):
    normals = np.zeros((pcd_scanline.shape[0], 3))
    
    pcd_x = pcd_scanline[:, x_col]
    pcd_y = pcd_scanline[:, y_col]
    pcd_z = pcd_scanline[:, z_col]
    pcd_scanline_xyz = np.column_stack((pcd_x, pcd_y, pcd_z))
    
    for i in prange(pcd_scanline_xyz.shape[0]):
        if i == 0:
            #normal = np.cross(pcd_scanline_xyz[i+1], pcd_scanline_xyz[i])
            normal = orient_cross_product(pcd_scanline_xyz[i+1], pcd_scanline_xyz[i], scanner_pos)
            normals[i] = ( normal / np.linalg.norm(normal) ) / 1
        elif i <= pcd_scanline_xyz.shape[0]-2:
            #normal = np.cross(pcd_scanline_xyz[i-1], pcd_scanline_xyz[i+1])
            normal = orient_cross_product(pcd_scanline_xyz[i-1], pcd_scanline_xyz[i+1], scanner_pos)
            normals[i] = (normal / np.linalg.norm(normal)) / 1
        else:
            #normal = np.cross(pcd_scanline_xyz[i-1], pcd_scanline_xyz[i])
            normal = orient_cross_product(pcd_scanline_xyz[i-1], pcd_scanline_xyz[i], scanner_pos)
            normals[i] = (normal / np.linalg.norm(normal)) / 1

    return normals



@njit(parallel=True)
def calculate_segmentation_metrics(pcd: np.ndarray,
                                   x_col: int,
                                   y_col: int,
                                   z_col: int,
                                   scanline_id_col: int,
                                   rho_col: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    # Extract the unique scanline ids from the pcd
    scanline_ids = get_scanline_ids(pcd, col=scanline_id_col)
    
    # Create empty arrays to store the segmentation metrics
    rho_diff = np.zeros(pcd.shape[0])
    slope = np.zeros(pcd.shape[0])
    curvature = np.zeros(pcd.shape[0])

    # Calculate the segmentation metrics for each scanline
    for i in prange(scanline_ids.shape[0]):
        # Extract the current scanline and its indices in the pcd
        scanline, scanline_indices = get_scanline(pcd, col=scanline_id_col, id=scanline_ids[i])
        
        # Calculate the rho_diff, slope, and curvature for the current scanline
        rho_diff_i = calculate_rho_diff(scanline, col=rho_col)
        slope_i = calculate_slope(scanline, x_col=x_col, y_col=y_col, z_col=z_col)
        curvature_i = calculate_curvature(slope_i)
        
        # Store the calculated metrics in the corresponding arrays
        for j in prange(scanline.shape[0]):
            rho_diff[scanline_indices[j]] = rho_diff_i[j]
            slope[scanline_indices[j]] = slope_i[j]
            curvature[scanline_indices[j]] = curvature_i[j]

    return rho_diff, slope, curvature


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
                        (pcd[:,slope_col] < slope_threshold) |
                        (pcd[:,curvature_col] > curvature_threshold))[0]
    
    # Assign the segment ids to the points
    for i in prange(pcd.shape[0]):
        segment_ids[i] = np.searchsorted(segments, i, side='left')
    
    # Increment the segment ids by 1 to start from 1 instead of 0
    segment_ids += 1
    
    return segment_ids
