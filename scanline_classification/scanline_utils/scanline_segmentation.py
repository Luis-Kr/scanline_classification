import numpy as np
from typing import Tuple
from numba import njit, prange

def sort_scanline(pcd: np.ndarray, 
                  scanline_id_col: int,
                  vert_angle_col: int) -> np.ndarray:
    # Sort point cloud based on scanline ID and vertical angle
    sort_idx = np.lexsort(np.rot90(pcd[:,(scanline_id_col, 
                                          vert_angle_col)]))
    
    return pcd[sort_idx], sort_idx


def get_scanline_intervals(pcd: np.ndarray,
                           scanline_id_col: int) -> np.ndarray:
    # Get differences between consecutive scanline IDs
    diff = np.diff(pcd[:, scanline_id_col])

    # Get indices where scanline ID changes
    scanline_intervals = np.where(diff > 0)[0] + 1

    # Add first and last indices
    scanline_intervals = np.insert(scanline_intervals, 0, 0)
    scanline_intervals = np.append(scanline_intervals, pcd.shape[0])
    
    return scanline_intervals


@njit()
def get_scanline(pcd: np.ndarray, 
                 lower_boundary: int, 
                 upper_boundary: int) -> Tuple[np.ndarray, np.ndarray]:
    # Extract a scanline from the point cloud given its lower and upper boundaries
    scanline = pcd[lower_boundary:upper_boundary]
    
    # Generate an array of indices corresponding to the scanline
    scanline_indices = np.arange(lower_boundary, upper_boundary)
    
    return scanline, scanline_indices


def recalculate_rho(cfg, pcd, pcd_xyz_scanpos_centered):
    # Calculate the rho values based on the x, y, z coordinates
    rho = np.sqrt(pcd_xyz_scanpos_centered[:,0]**2 + pcd_xyz_scanpos_centered[:,1]**2 + pcd_xyz_scanpos_centered[:,2]**2)
    
    # Replace the rho values in the pcd
    pcd[:, cfg.pcd_col.rho] = rho
    
    return pcd


@njit()
def calculate_rho_diff(pcd: np.ndarray, 
                       rho_col: int) -> np.ndarray:
    # Calculate the absolute difference between adjacent elements
    rho_diff = np.abs(np.diff(np.ascontiguousarray(pcd[:, rho_col])))
    
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
    
    return local_slope_deg


@njit()
def calculate_slope(scanline_xyz: np.ndarray,
                    padded_scanline: np.ndarray,
                    num_neighbors: np.ndarray,
                    max_num_neighbors: int) -> np.ndarray:
    # Initialize an array to store the slope values
    slope = np.zeros(scanline_xyz.shape[0])
    
    # Loop over each point in the scanline
    for idx in range(scanline_xyz.shape[0]):
        # Calculate the end index for the neighbors on the right
        n = int(idx + max_num_neighbors)
        
        # Calculate the slope at the current point by taking the mean slope
        # between the current point and its neighbors on the left and right
        slope[idx] = np.median(get_slope_3D(points_left_side=padded_scanline[n-num_neighbors[idx]:n], 
                                          points_right_side=padded_scanline[n+1:n+num_neighbors[idx]+1]))

    return slope


@njit()
def slope_lstsq_local_neighborhood(points_left_side: np.ndarray, 
                                   points_right_side: np.ndarray) -> np.ndarray: 
    neighborhood_points = np.concatenate((points_left_side, points_right_side))
    
    # Extract the rho-values from the neighborhood points
    X = neighborhood_points[:, 0]
    
    # Create a matrix A with rho-values and a column of ones
    A = np.column_stack((X, np.ones(neighborhood_points.shape[0])))
    
    # Extract the z-coordinates from the neighborhood points
    B = neighborhood_points[:, 1]
    
    # Solve the linear system Ax = B using least squares method
    lstsq_solution, _, _, _ = np.linalg.lstsq(A, B)
    
    # Calculate the slope in degrees
    slope_deg = np.rad2deg(np.arctan(lstsq_solution[0]))
    
    return slope_deg


@njit()
def calculate_slope_least_squares(scanline: np.ndarray,
                                  num_neighbors: np.ndarray,
                                  x_col: int,
                                  y_col: int,
                                  max_num_neighbors: int,
                                  X_col: int,
                                  Y_col: int) -> np.ndarray:
    # Extract the rho and z coordinates from the scanline
    X = scanline[:, X_col]
    Y = scanline[:, Y_col]
    # x = scanline[:, x_col]
    # y = scanline[:, y_col]
    # Y = np.sqrt(x**2 + y**2)
    
    # Merge the rho and z-coordinates into a single array
    scanline_XY = np.column_stack((X, Y))
    
    # Pad the array at the beginning and end
    pad_scanline_XY = pad_reflect(scanline_XY, max_num_neighbors)
    
    # Calculate the slope using the least-squares method for each point in the scanline
    slope = np.zeros(scanline.shape[0])
    for idx in range(scanline.shape[0]):
        n = int(idx + max_num_neighbors)
        slope[idx] = slope_lstsq_local_neighborhood(points_left_side=pad_scanline_XY[n-num_neighbors[idx]:n], 
                                                    points_right_side=pad_scanline_XY[n+1:n+num_neighbors[idx]+1])
    
    return slope


@njit()
def calculate_curvature(slope_arr: np.ndarray, 
                        num_neighbors: np.ndarray,
                        max_num_neighbors: int) -> np.ndarray:
    # Pad the array at the beginning and end
    pad_slope_arr = pad_reflect(slope_arr, max_num_neighbors)
    
    # Store the curvature values
    curvature = np.zeros(slope_arr.shape[0])
    
    for idx in range(slope_arr.shape[0]):
        i = idx + max_num_neighbors
        curvature[idx] = np.median(np.abs(pad_slope_arr[i+1:i+num_neighbors[idx]+1] - pad_slope_arr[i-num_neighbors[idx]:i][::-1]))

    return curvature


# np.gradient is not supported by numba (replacement)
@njit()
def numba_gradient(arr):
    gradient = np.empty_like(arr)
    gradient[0] = arr[1] - arr[0]
    gradient[-1] = arr[-1] - arr[-2]
    gradient[1:-1] = (arr[2:] - arr[:-2]) / 2
    return gradient


@njit()
def calculate_curvature_gradient(slope_arr: np.ndarray, 
                                 max_num_neighbors: int) -> np.ndarray:
    # Pad the array at the beginning and end
    pad_slope_arr = pad_reflect(slope_arr, max_num_neighbors)
    
    # Calculate the curvature 
    curvature = numba_gradient(pad_slope_arr)
    
    # Trim the gradient array to the original shape
    curvature = curvature[max_num_neighbors:-max_num_neighbors]
    
    return curvature


@njit()
def calculate_distances_point_lines(center_point: np.ndarray, 
                                    points_left_side: np.ndarray, 
                                    points_right_side: np.ndarray) -> np.ndarray:
    a = points_right_side - points_left_side
    vec_p1_p0 = center_point - points_left_side
    distance = np.zeros(vec_p1_p0.shape[0])
    
    for v in range(vec_p1_p0.shape[0]):
        # Calculate the dot product of the direction vector with itself
        dproduct_direction_vector = np.dot(a[v], a[v])
        
        # If the dot product is not zero (i.e., the points are not identical)
        if dproduct_direction_vector != 0:
            # Calculate the parameter t for the line equation
            t = np.dot(vec_p1_p0[v], a[v]) / dproduct_direction_vector
            
            # Calculate the point on the line that is closest to the center point
            l1 = points_left_side[v] + t*a[v]
            
            # Calculate the distance from this point to the center point
            distance[v] = np.linalg.norm(l1 - center_point)
        else:
            # If the dot product is zero (i.e., the points are identical), set the distance to zero
            distance[v] = 0
            
    return distance


@njit()
def calculate_roughness(scanline_xyz: np.ndarray,
                        padded_scanline: np.ndarray,
                        num_neighbors: np.ndarray,
                        max_num_neighbors: int) -> np.ndarray:
    roughness = np.zeros(scanline_xyz.shape[0])
    for idx in range(scanline_xyz.shape[0]):
        n = int(idx + max_num_neighbors)
        roughness[idx] = np.nanpercentile(calculate_distances_point_lines(center_point=padded_scanline[n], 
                                                                          points_left_side=padded_scanline[n-num_neighbors[idx]:n], 
                                                                          points_right_side=padded_scanline[n+1:n+num_neighbors[idx]+1]), 95)
    
    return roughness


@njit(parallel=True)
def calculate_segmentation_metrics(pcd: np.ndarray,
                                   scanline_intervals: np.ndarray,
                                   x_col: int,
                                   y_col: int,
                                   z_col: int,
                                   expected_value_col: int,
                                   rho_col: int,
                                   horiz_angle_col: int,
                                   neighborhood_multiplier: float,
                                   least_squares_method: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Create empty arrays to store the segmentation metrics
    rho_diff = np.zeros(pcd.shape[0])
    slope = np.zeros(pcd.shape[0])
    slope_lstsq = np.zeros(pcd.shape[0])
    curvature = np.zeros(pcd.shape[0])
    curvature_lstsq = np.zeros(pcd.shape[0])
    roughness = np.zeros(pcd.shape[0])
    
    # Calculate the segmentation metrics for each scanline
    for i in prange(scanline_intervals.shape[0]-1):
        # Extract the current scanline and its indices in the pcd
        scanline, scanline_indices = get_scanline(pcd=pcd, 
                                                  lower_boundary=scanline_intervals[i], 
                                                  upper_boundary=scanline_intervals[i+1])
        
        # Calculate the density and the number of neighbors for each point in the scanline
        density = 1 / scanline[:, expected_value_col]
        
        # Smoothing case
        k_neighbors = np.ceil(np.sqrt(density))
        
        # If k_neighbors is smaller than 3, set it to 3 to avoid too small neighborhoods
        k_neighbors[k_neighbors < 3] = 3
        k_neighbors *= neighborhood_multiplier
        
        # Round to the nearest integer
        k_neighbors = np.round(k_neighbors)
        
        # Smoothing with constant k
        #k_neighbors = np.ones(scanline.shape[0]) * 20
        
        # Extract the x, y, z coordinates from the scanline (numba does not support indexing with multiple columns)
        x = scanline[:, x_col]
        y = scanline[:, y_col]
        z = scanline[:, z_col]
        
        # Merge the x, y, z coordinates into a single array
        scanline_xyz = np.column_stack((x, y, z))
        
        # Calculate the maximum number of neighbors to consider
        max_num_neighbors = int(np.max(k_neighbors))  
        
        # Pad the array at the beginning and end
        padded_scanline = pad_reflect(scanline_xyz, max_num_neighbors)
        
        # Calculate the rho_diff, slope, curvature, and roughness for the current scanline
        rho_diff_i = calculate_rho_diff(scanline, 
                                        rho_col=rho_col)
        
        #if least_squares_method:
        slope_lstsq_i = calculate_slope_least_squares(scanline=scanline,
                                                num_neighbors=k_neighbors,
                                                x_col= x_col,
                                                y_col= y_col,
                                                max_num_neighbors=max_num_neighbors,
                                                X_col=horiz_angle_col,
                                                Y_col=rho_col)
        #else:
        slope_i = calculate_slope(scanline_xyz=scanline, 
                                    padded_scanline=padded_scanline,
                                    num_neighbors=k_neighbors,
                                    max_num_neighbors=max_num_neighbors)
        
        # curvature_i = calculate_curvature(slope_arr=slope_i, 
        #                                   num_neighbors=k_neighbors,
        #                                   max_num_neighbors=max_num_neighbors)
        
        curvature_i = calculate_curvature_gradient(slope_arr=slope_i, 
                                                   max_num_neighbors=max_num_neighbors)
        
        curvature_lstsq_i = calculate_curvature_gradient(slope_arr=slope_lstsq_i, 
                                                         max_num_neighbors=max_num_neighbors)
        
        roughness_i = calculate_roughness(scanline_xyz=scanline_xyz, 
                                          padded_scanline=padded_scanline,
                                          num_neighbors=k_neighbors,
                                          max_num_neighbors=max_num_neighbors)
        
        rho_diff[scanline_indices] = rho_diff_i
        slope[scanline_indices] = slope_i
        slope_lstsq[scanline_indices] = slope_lstsq_i
        curvature[scanline_indices] = curvature_i
        curvature_lstsq[scanline_indices] = curvature_lstsq_i
        roughness[scanline_indices] = roughness_i

    return rho_diff, slope, slope_lstsq, curvature, curvature_lstsq, roughness


@njit(parallel=True)
def scanline_segmentation(pcd: np.ndarray, 
                          expected_value_col: int, 
                          expected_value_std_col: int,
                          std_multiplier: int,
                          rho_diff_col: int, 
                          slope_col: int, 
                          curvature_col: int,
                          slope_threshold: float,
                          curvature_threshold: float) -> np.ndarray:
    # Initialize an array of zeros to store the segment ids
    segment_ids = np.zeros(pcd.shape[0])
    
    # Identify the segments based on the conditions
    segments = np.where((pcd[:,rho_diff_col] > (pcd[:,expected_value_col] + pcd[:,expected_value_std_col]*std_multiplier)) | 
                        (np.abs(pcd[:,curvature_col]) > curvature_threshold))[0] #(pcd[:,slope_col] < slope_threshold) |
    
    # Assign the segment ids to the points
    for i in prange(pcd.shape[0]):
        segment_ids[i] = np.searchsorted(segments, i, side='left')
    
    # Increment the segment ids by 1 to start from 1 instead of 0
    segment_ids += 1
    
    return segment_ids
