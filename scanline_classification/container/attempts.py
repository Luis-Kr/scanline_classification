import numpy as np
from typing import Tuple
from numba import njit, prange


## scanline segmentation

# @njit()
# def orient_cross_product(vector_ba: np.ndarray, 
#                          vector_bc: np.ndarray, 
#                          viewpoint: np.ndarray) -> np.ndarray:
#     """
#     Orient the cross product (normal) of two vectors towards a viewpoint.

#     Parameters:
#     vector_ba (np.ndarray): The first vector.
#     vector_bc (np.ndarray): The second vector.
#     viewpoint (np.ndarray): The viewpoint.

#     Returns:
#     np.ndarray: The oriented cross product of the two vectors.
#     """
#     # Calculate the cross product of the two vectors
#     cross_product = np.cross(vector_ba, vector_bc)
    
#     # Calculate the vector from the cross product to the viewpoint
#     vector_to_viewpoint = viewpoint - cross_product
    
#     # If the cross product is not oriented towards the viewpoint, flip it
#     if np.dot(cross_product, vector_to_viewpoint) < 0:
#         cross_product = np.cross(vector_bc, vector_ba)
    
#     return cross_product


# @njit()
# def compute_normals(pcd_scanline, scanner_pos, x_col, y_col, z_col):
#     normals = np.zeros((pcd_scanline.shape[0], 3))
    
#     pcd_x = pcd_scanline[:, x_col]
#     pcd_y = pcd_scanline[:, y_col]
#     pcd_z = pcd_scanline[:, z_col]
#     pcd_scanline_xyz = np.column_stack((pcd_x, pcd_y, pcd_z))
    
#     for i in prange(pcd_scanline_xyz.shape[0]):
#         if i == 0:
#             #normal = np.cross(pcd_scanline_xyz[i+1], pcd_scanline_xyz[i])
#             normal = orient_cross_product(pcd_scanline_xyz[i+1], pcd_scanline_xyz[i], scanner_pos)
#             normals[i] = ( normal / np.linalg.norm(normal) ) / 1
#         elif i <= pcd_scanline_xyz.shape[0]-2:
#             #normal = np.cross(pcd_scanline_xyz[i-1], pcd_scanline_xyz[i+1])
#             normal = orient_cross_product(pcd_scanline_xyz[i-1], pcd_scanline_xyz[i+1], scanner_pos)
#             normals[i] = (normal / np.linalg.norm(normal)) / 1
#         else:
#             #normal = np.cross(pcd_scanline_xyz[i-1], pcd_scanline_xyz[i])
#             normal = orient_cross_product(pcd_scanline_xyz[i-1], pcd_scanline_xyz[i], scanner_pos)
#             normals[i] = (normal / np.linalg.norm(normal)) / 1

#     return normals



# @njit()
# def slope_lstsq_local_neighborhood_old(points_left_side: np.ndarray, 
#                                    points_right_side: np.ndarray) -> np.ndarray: 
#     # Merge the left and right side points into a single array
#     neighborhood_points = np.concatenate((points_left_side, points_right_side))
#     X = neighborhood_points[:, 0]
#     Y = neighborhood_points[:, 1]
    
#     # Calculate the least-squares solution
#     A = np.column_stack((X, Y, np.ones(neighborhood_points.shape[0])))
#     B = neighborhood_points[:, 2]
    
#     lstsq_solution, _, _, _ = np.linalg.lstsq(A, B)
    
#     # Select two points on the line (could be any two points)
#     t1 = 10
#     t2 = 0
    
#     # Calculate the z-coordinates of the two points
#     x1, y1 = t1, t1
#     x2, y2 = t2, t2
#     z1 = lstsq_solution[0]*t1 + lstsq_solution[1]*t1 + lstsq_solution[2]
#     z2 = lstsq_solution[0]*t2 + lstsq_solution[1]*t2 + lstsq_solution[2]
    
#     # Calculate the change in z
#     z_change = z2 - z1
    
#     # Calculate the distance in 3D (x, y, z)
#     distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
#     slope = np.abs(np.arctan2(z_change, distance))
    
#     # Calculate the slope in radians (z_change/distance; equivalent to rise/run)
#     slope_rad = np.arctan2(z_change, distance)
    
#     # Convert the angle to degrees
#     slope_deg = np.rad2deg(np.abs(slope_rad))
    
#     return slope_deg