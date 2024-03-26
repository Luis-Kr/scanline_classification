import numpy as np
from typing import Tuple
from numba import njit, prange
from omegaconf import DictConfig
import sys
import numba


def sort_scanline(cfg: DictConfig, 
                  pcd: np.ndarray) -> np.ndarray:
    # Sort point cloud based on scanline ID and vertical angle
    #sort_idx = np.lexsort(np.rot90(pcd[:,(scanline_id_col, vert_angle_col)]))
    
    structured_pcd = np.zeros(pcd.shape[0], dtype=[('scanline_id', np.int64), ('vert_angle', np.float64)])
    structured_pcd['scanline_id'] = pcd[:, cfg.pcd_col.scanline_id]
    structured_pcd['vert_angle'] = pcd[:, cfg.pcd_col.vert_angle]
    
    sort_idx = np.argsort(structured_pcd, order=['scanline_id', 'vert_angle'])
    
    return pcd[sort_idx], sort_idx


def get_scanline_intervals(pcd: np.ndarray,
                           scanline_id_col: int) -> np.ndarray:
    # Get differences between consecutive scanline IDs
    diff = np.diff(pcd[:, scanline_id_col])

    # Get indices where scanline ID changes
    scanline_intervals = np.where(diff > 0)[0] + 1

    # Add first and last indices
    scanline_intervals = np.insert(scanline_intervals, 0, 0)
    scanline_intervals = np.append(scanline_intervals, pcd.shape[0]+1)
    
    scanline_intervals_dict = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.int64[:])
    
    scanline_id_arr = np.zeros((scanline_intervals.shape[0]-1), dtype=np.int64)

    for point_idx, v in enumerate(scanline_intervals[:-1]):
        scanline_intervals_dict[point_idx+1] = np.array([scanline_intervals[point_idx], scanline_intervals[point_idx+1]])
        scanline_id_arr[point_idx] = point_idx+1
    
    return scanline_intervals_dict, scanline_id_arr


@njit()
def get_scanline(pcd: np.ndarray, 
                 lower_boundary: int, 
                 upper_boundary: int) -> Tuple[np.ndarray, np.ndarray]:
    # Extract a scanline from the point cloud given its lower and upper boundaries
    scanline = pcd[lower_boundary:upper_boundary]
    
    # Generate an array of indices corresponding to the scanline
    scanline_indices = np.arange(lower_boundary, upper_boundary)
    
    return scanline, scanline_indices


# def recalculate_rho(cfg, pcd, pcd_xyz_scanpos_centered):
#     # Calculate the rho values based on the x, y, z coordinates
#     rho = np.sqrt(pcd_xyz_scanpos_centered[:,0]**2 + pcd_xyz_scanpos_centered[:,1]**2 + pcd_xyz_scanpos_centered[:,2]**2)
    
#     # Replace the rho values in the pcd
#     pcd[:, cfg.pcd_col.rho] = rho
    
#     return pcd


# @njit()
# def calculate_rho_diff(pcd: np.ndarray, 
#                        rho_col: int) -> np.ndarray:
#     # Calculate the absolute difference between adjacent elements
#     rho_diff_forward = np.abs(np.diff(np.ascontiguousarray(pcd[:, rho_col])))
#     rho_diff_backward = np.abs(np.diff(np.ascontiguousarray(pcd[:, rho_col])[::-1]))
    
#     # Append the last element of rho_diff to itself to maintain the same length as the input pcd
#     rho_diff_forward = np.append(rho_diff_forward, rho_diff_forward[-1])
#     rho_diff_backward = np.append(rho_diff_backward, rho_diff_backward[-1])
    
#     # Always take the maximum value along rho_diff_forward and rho_diff_backward
#     rho_diff = np.maximum(rho_diff_forward, rho_diff_backward[::-1])
    
#     return rho_diff

@njit()
def diff(arr):
    return arr[1:] - arr[:-1]

@njit()
def calculate_rho_diff(pcd: np.ndarray, rho_col: int) -> np.ndarray:
    # Calculate the absolute difference between adjacent elements
    rho_diff_forward = np.abs(diff(np.ascontiguousarray(pcd[:, rho_col])))
    rho_diff_backward = np.abs(diff(np.ascontiguousarray(pcd[:, rho_col])[::-1]))
    
    # Append the last element of rho_diff to itself to maintain the same length as the input pcd
    rho_diff_forward = np.append(rho_diff_forward, rho_diff_forward[-1])
    rho_diff_backward = np.append(rho_diff_backward, rho_diff_backward[-1])
    
    # Always take the maximum value along rho_diff_forward and rho_diff_backward
    rho_diff = np.maximum(rho_diff_forward, rho_diff_backward[::-1])
    
    return rho_diff


# @njit()
# def pad_reflect(arr: np.ndarray, 
#                 pad_width: int) -> np.ndarray:
#     """
#     Function to pad an array using the 'reflect' mode (numpy.pad replacement). 
    
#     Parameters:
#     arr (np.ndarray): The input array to be padded.
#     pad_width (int): The number of elements by which to pad the array.
    
#     Returns:
#     np.ndarray: The padded array.
#     """
#     return np.concatenate((arr[pad_width:0:-1], arr, arr[-2:-pad_width-2:-1]))


# @njit()
# def get_slope_3D(points_left_side: np.ndarray,
#                  points_right_side: np.ndarray) -> np.ndarray:
#     # Get the z-differences between the points
#     z_diff = points_right_side[:, 2] - points_left_side[:, 2]
    
#     # Get the 3D distance based on the x,y,z coordinates
#     dist_3d = np.sqrt(((points_right_side[:, 0] - points_left_side[:, 0]) ** 2) + 
#                       ((points_right_side[:, 1] - points_left_side[:, 1]) ** 2) + 
#                       (z_diff ** 2))
    
#     # Calculate the slope in radians
#     local_slope = np.arctan2(z_diff, dist_3d)
    
#     # Calculate the slope in degrees
#     local_slope_deg = np.rad2deg(local_slope)
    
#     return local_slope_deg


# @njit()
# def calculate_slope_rise_over_run(scanline_xyz: np.ndarray,
#                                   padded_scanline: np.ndarray,
#                                   num_neighbors: np.ndarray,
#                                   max_num_neighbors: int) -> np.ndarray:
#     # Initialize an array to store the slope values
#     slope = np.zeros(scanline_xyz.shape[0])
    
#     # Loop over each point in the scanline
#     for idx in range(scanline_xyz.shape[0]):
#         # Calculate the end index for the neighbors on the right
#         n = int(idx + max_num_neighbors)
        
#         # Calculate the slope at the current point by taking the mean slope
#         # between the current point and its neighbors on the left and right
#         slope[idx] = np.median(get_slope_3D(points_left_side=padded_scanline[n-num_neighbors[idx]:n], 
#                                             points_right_side=padded_scanline[n+1:n+num_neighbors[idx]+1]))

#     return slope


@njit()
def calculate_slope(point_neighborhood: np.ndarray,
                    X_col: int,
                    Y_col: int) -> np.ndarray: 
    # Extract the desired X and Y coordinates from the neighborhood points
    X = point_neighborhood[:, X_col]
    
    # Create a matrix A with X values and a column of ones
    A = np.column_stack((X, np.ones(point_neighborhood.shape[0])))
    
    # Extract the Y-coordinates from the neighborhood points
    B = point_neighborhood[:, Y_col]
    
    # Solve the linear system Ax = B using least squares method
    lstsq_solution, _, _, _ = np.linalg.lstsq(A, B)
    
    # Calculate the slope in degrees
    slope_deg = np.rad2deg(np.arctan(lstsq_solution[0]))
    
    return np.abs(slope_deg)


@njit()
def calculate_slope_D_Z(point_neighborhood: np.ndarray,
                        center_point: np.ndarray,
                        X_col: int,
                        Y_col: int,
                        Z_col: int) -> np.ndarray: 
    X = np.sqrt(point_neighborhood[:, X_col]**2 + point_neighborhood[:, Y_col]**2)
    
    # Create a matrix A with X values and a column of ones
    A = np.column_stack((X, np.ones(point_neighborhood.shape[0])))
    
    # Extract the Y-coordinates from the neighborhood points
    B = point_neighborhood[:, Z_col]    
    
    # Solve the linear system Ax = B using least squares method
    lstsq_solution, _, _, _ = np.linalg.lstsq(A, B)
    
    # Calculate the slope in degrees
    slope_deg = np.rad2deg(np.arctan(lstsq_solution[0]))
    
    # Calculate the distance from the point to the line derived from the least squares solution
    X_center_point = np.sqrt(center_point[X_col]**2 + center_point[Y_col]**2)
    Y_center_point = center_point[Z_col]
    center_point = np.array([X_center_point, Y_center_point])
    distance = np.abs(lstsq_solution[0] * center_point[0] - center_point[1] + lstsq_solution[1]) / np.sqrt(lstsq_solution[0]**2 + 1)
    
    return np.abs(slope_deg), np.abs(distance)


# np.gradient is not supported by numba (replacement)
@njit()
def calculate_curvature_gradient(arr):
    gradient = np.empty_like(arr)
    gradient[0] = arr[1] - arr[0]
    gradient[-1] = arr[-1] - arr[-2]
    gradient[1:-1] = (arr[2:] - arr[:-2]) / 2
    return gradient


@njit()
def distance_neighborhood_points(center_point, point_neighborhood_coarse):
    
    difference = point_neighborhood_coarse[:,:3] - center_point
    distances = np.sqrt(np.sum(difference**2, axis=1))
    
    mean_dist = np.mean(distances[np.argsort(distances)][:5])
    std_dist = np.std(distances[np.argsort(distances)][:5])
    density = np.ceil(np.sqrt(1/mean_dist))
    
    return mean_dist, std_dist, density


@njit()
def scanline_neighborhood_points(pcd, i, theta_range_reference, knickpoints_dict, scanline_id_arr, center_point, neighborhood_search):
    scanline_neighborhood_minus = np.empty((0, 3), dtype=pcd.dtype)
    scanline_neighborhood_plus = np.empty((0, 3), dtype=pcd.dtype)

    for j in range(1, 11):
        scanline_minus = pcd[knickpoints_dict[scanline_id_arr[i - j]][0]:knickpoints_dict[scanline_id_arr[i - j]][1], :3]
        scanline_neighborhood_minus = np.concatenate((scanline_neighborhood_minus, scanline_minus[np.searchsorted(scanline_minus[:, 9], theta_range_reference), :3]))
        scanline_plus = pcd[knickpoints_dict[scanline_id_arr[i + j]][0]:knickpoints_dict[scanline_id_arr[i + j]][1], :3]
        scanline_neighborhood_plus = np.concatenate((scanline_neighborhood_plus, scanline_plus[np.searchsorted(scanline_plus[:, 9], theta_range_reference), :3]))
    
    # Merge the scanlines
    scanline_neighborhoods = np.vstack((
    scanline_neighborhood_minus,
    scanline_neighborhood_plus,
    ))
    
    difference = scanline_neighborhoods - center_point
    distances = np.sqrt(np.sum(difference**2, axis=1))
    
    if scanline_neighborhoods.shape[0] >= neighborhood_search:
        nearest_neighborhood_points = scanline_neighborhoods[np.argsort(distances)][:neighborhood_search].copy()
    else:
        nearest_neighborhood_points = scanline_neighborhoods.copy()
    
    return nearest_neighborhood_points


def compute_scanner_LOS(pcd: np.ndarray):
    scanner_pos = np.mean(pcd[:,:3], axis=0).copy()
    pcd_xyz_centered = pcd[:,:3].copy() - scanner_pos
    scanner_LOS = pcd_xyz_centered[:, :3] / np.linalg.norm(pcd_xyz_centered[:,:3], axis=1, keepdims=True)
    return -scanner_LOS


# @njit()
# def compute_roughness(pcd: np.ndarray, normal: np.ndarray, point: np.ndarray) -> np.ndarray:
#     # Compute the mean of the neighborhood points along axis 0 
#     # Approximates the best fit plane to the neighborhood points
#     mean_pt_nbh = np.sum(pcd[:,:3], axis=0) / pcd.shape[0]
    
#     # Compute the distance from the points to the plane
#     return np.abs(np.dot(point - mean_pt_nbh, normal))


# @njit()
# def calc_R(c, x, y):
#     """ calculate the distance of each 2D points from the center c=(xc, yc) """
#     return np.sqrt((x-c[0])**2 + (y-c[1])**2)


# @njit()
# def calculate_curvature(nearest_neighborhood_points):
#     X = nearest_neighborhood_points[:, 0].copy()
#     Y = nearest_neighborhood_points[:, 1].copy()
#     A = np.column_stack((X, Y, np.ones(X.shape[0])))
#     B = X**2 + Y**2
#     center = np.linalg.lstsq(A, B)[0] 
#     xc, yc = center[0]/2, center[1]/2
#     Ri = calc_R([xc, yc], X, Y)
#     R = Ri.mean()
#     curvature = 1 / R
#     curvature = curvature if curvature < 500 else 0
#     return curvature


@njit(parallel=True)
def calculate_segmentation_metrics(pcd: np.ndarray,
                                   scanline_intervals,
                                   scanline_id_arr: np.ndarray,
                                   x_col: int,
                                   y_col: int,
                                   z_col: int,
                                   rho_col: int,
                                   horiz_angle_col: int,
                                   scanner_LOS: np.ndarray,
                                   scanline_3D_attributes: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Create empty arrays to store the segmentation metrics
    rho_diff = np.zeros(pcd.shape[0])
    slope_phi_rho = np.zeros(pcd.shape[0])
    slope_D_Z = np.zeros(pcd.shape[0])
    curvature_phi_rho = np.zeros(pcd.shape[0])
    curvature_D_Z = np.zeros(pcd.shape[0])
    roughness = np.zeros(pcd.shape[0])
    mean_dist = np.zeros(pcd.shape[0])
    std_dist = np.zeros(pcd.shape[0])
    density = np.zeros(pcd.shape[0])
    
    if scanline_3D_attributes:
        normals_scanline_3D = np.zeros((pcd.shape[0], 3))
        curvature_scanline_3D = np.zeros(pcd.shape[0])
        zenith_angle_scanline_3D = np.zeros(pcd.shape[0])
    
    # Calculate the segmentation metrics for each scanline
    for i in prange(scanline_id_arr.shape[0]):
        # Extract the current scanline and its indices in the pcd
        scanline, scanline_indices = get_scanline(pcd=pcd, 
                                                  lower_boundary=scanline_intervals[scanline_id_arr[i]][0], 
                                                  upper_boundary=scanline_intervals[scanline_id_arr[i]][1])
        
        rho_diff[scanline_indices] = calculate_rho_diff(scanline, rho_col=rho_col)
        
        for point_idx in range(scanline.shape[0]):
            start_index = max(0, point_idx - 20)
            end_index = min(scanline.shape[0], point_idx + 21)
            point_neighborhood_coarse = scanline[start_index:end_index, :].copy()
            theta_range_reference = point_neighborhood_coarse[:, 9]
            
            # Calculate the mean distance, standard deviation, and density of the neighborhood points
            mean_dist_i, std_dist_i, density_i = distance_neighborhood_points(scanline[point_idx, :3], point_neighborhood_coarse)
            density_phirho_i = np.round(density_i*2)
            density_dz_i = np.round(density_i)
            
            # Adjust the density values range
            if density_phirho_i < 11:
                density_phirho_i = 11
            if density_phirho_i > 17:
                density_phirho_i = 17
                
            if density_dz_i < 4:
                density_dz_i = 4
            if density_dz_i > 20:
                density_dz_i = 19
            
            # Extract the neighborhood points for the current point based on the density values
            difference = point_neighborhood_coarse[:, :3] - scanline[point_idx, :3]
            distances = np.sqrt(np.sum(difference**2, axis=1))
            point_neighborhood_phi_rho = point_neighborhood_coarse[np.argsort(distances)][:int(density_phirho_i), :].copy()
            point_neighborhood_d_z = point_neighborhood_coarse[np.argsort(distances)][:int(density_dz_i), :]
            
            # Center the neighborhood points around the current point
            center_pos = np.array([point_neighborhood_d_z[:, x_col].mean(), point_neighborhood_d_z[:, y_col].mean(), point_neighborhood_d_z[:, z_col].mean()])
            center_point_xyz = scanline[point_idx, :3] - center_pos
            point_neighborhood_d_z[:,x_col] -= center_pos[0]
            point_neighborhood_d_z[:,y_col] -= center_pos[1]
            point_neighborhood_d_z[:,z_col] -= center_pos[2]
            
            # Calculate the 2D attributes for the current point
            slope_phi_rho_i = calculate_slope(point_neighborhood=point_neighborhood_phi_rho,
                                              X_col=horiz_angle_col,
                                              Y_col=rho_col)
            
            slope_D_Z_i, distance_i = calculate_slope_D_Z(point_neighborhood=point_neighborhood_d_z,
                                                          center_point=center_point_xyz, 
                                                          X_col=x_col,
                                                          Y_col=y_col,
                                                          Z_col=z_col)
            
            slope_phi_rho[scanline_intervals[scanline_id_arr[i]][0] + point_idx] = slope_phi_rho_i
            slope_D_Z[scanline_intervals[scanline_id_arr[i]][0] + point_idx] = slope_D_Z_i
            roughness[scanline_intervals[scanline_id_arr[i]][0] + point_idx] = distance_i
            mean_dist[scanline_intervals[scanline_id_arr[i]][0] + point_idx] = mean_dist_i
            std_dist[scanline_intervals[scanline_id_arr[i]][0] + point_idx] = std_dist_i
            density[scanline_intervals[scanline_id_arr[i]][0] + point_idx] = density_i
            
            if scanline_3D_attributes:
                density_3D = np.round(density_i)*3
            
                # Adjust the density values range
                if density_3D < 10:
                    density_3D = 10
                if density_3D > 100:
                    density_3D = 100
                    
                nearest_neighborhood_points = scanline_neighborhood_points(pcd, 
                                                                           i, 
                                                                           theta_range_reference, 
                                                                           scanline_intervals, 
                                                                           scanline_id_arr, 
                                                                           pcd[scanline_intervals[scanline_id_arr[i]][0] + point_idx, :3],
                                                                           int(density_3D))

                if nearest_neighborhood_points.shape[0] > 3:
                    nearest_neighborhood_points -= np.array([nearest_neighborhood_points[:, 0].mean(), nearest_neighborhood_points[:, 1].mean(), nearest_neighborhood_points[:, 2].mean()])
                    
                    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(nearest_neighborhood_points.T))
                    normal = eigenvectors[:, 0]
                    if np.dot(normal, scanner_LOS[scanline_intervals[scanline_id_arr[i]][0] + point_idx, :3]) < 0:
                        normal *= -1

                    normals_scanline_3D[scanline_intervals[scanline_id_arr[i]][0] + point_idx, :] = normal
                    curvature_scanline_3D[scanline_intervals[scanline_id_arr[i]][0] + point_idx] = np.min(eigenvalues) / np.sum(eigenvalues)
                    zenith_angle_scanline_3D[scanline_intervals[scanline_id_arr[i]][0] + point_idx] = np.degrees(np.arccos(normal[2]))
                    
        curvature_phi_rho[scanline_indices] = np.abs(calculate_curvature_gradient(arr=slope_phi_rho[scanline_indices]))
        curvature_D_Z[scanline_indices] = np.abs(calculate_curvature_gradient(arr=slope_D_Z[scanline_indices]))

    return rho_diff, curvature_phi_rho, slope_D_Z, curvature_D_Z, roughness, mean_dist, std_dist, curvature_scanline_3D, zenith_angle_scanline_3D


@njit(parallel=True)
def scanline_segmentation(rho_diff: np.ndarray, 
                          curvature: np.ndarray,
                          mean_dist: np.ndarray, 
                          std_dist: np.ndarray,
                          std_multiplier: int,
                          curvature_threshold: float) -> np.ndarray:
    # Initialize an array of zeros to store the segment ids
    segment_ids = np.zeros(rho_diff.shape[0])
    
    # Identify the segments based on the conditions
    segments = np.where((rho_diff > (mean_dist + std_dist*std_multiplier)) | 
                        (curvature > curvature_threshold))[0]
    
    # Assign the segment ids to the points
    for i in prange(rho_diff.shape[0]):
        segment_ids[i] = np.searchsorted(segments, i, side='left')
    
    # Increment the segment ids by 1 to start from 1 instead of 0
    segment_ids += 1
    
    return segment_ids
