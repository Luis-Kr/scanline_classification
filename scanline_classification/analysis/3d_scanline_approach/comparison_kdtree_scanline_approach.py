import numpy as np
import open3d as o3d
from typing import List, Tuple
from scipy.spatial import cKDTree
from pathlib import Path  
import numba
from numba import njit, prange, jit
import time
import pandas as pd

root_dir = Path(__file__).resolve().parents[3]
print(root_dir)


def import_pcd(pcd_path):
    pcd = np.loadtxt(pcd_path, delimiter=" ")
    
    return pcd
    

def preprocessing(pcd):
    diff = np.diff(pcd[:, 14])
    knickpoints = np.where(diff != 0)[0] + 1
    knickpoints = np.insert(knickpoints, 0, 0)
    knickpoints = np.append(knickpoints, pcd[:, 14].shape[0]+1)

    knickpoints_dict = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.int64[:]
    )
    scanline_id_arr = np.zeros((knickpoints.shape[0]-1), dtype=np.int64)

    for point_idx, v in enumerate(knickpoints[:-1]):
        knickpoints_dict[point_idx+1] = np.array([knickpoints[point_idx], knickpoints[point_idx+1]])
        scanline_id_arr[point_idx] = point_idx+1
        
    return knickpoints_dict, scanline_id_arr


#@jit(nopython=True)
def sort_theta(pcd, knickpoints_dict, scanline_id_arr):
    high_theta_idx = np.where(pcd[:, 14] > 0)[0]
    
    structured_pcd = np.zeros(pcd[high_theta_idx,:].shape[0], dtype=[('scanline_id', np.int64), ('value', np.float64)])
    structured_pcd['scanline_id'] = pcd[high_theta_idx, 14]
    structured_pcd['value'] = pcd[high_theta_idx, 9]

    # Sort the structured array
    sort_idx = np.argsort(structured_pcd, order=['scanline_id', 'value'])

    # Apply the sorted indices to the point cloud data
    pcd[high_theta_idx] = pcd[high_theta_idx][sort_idx]
    
    return pcd


def compute_scanner_LOS(pcd: np.ndarray):
    scanner_pos = np.mean(pcd[:,:3], axis=0)
    pcd_xyz_centered = pcd[:,:3] - scanner_pos
    scanner_LOS = pcd_xyz_centered[:, :3] / np.linalg.norm(pcd_xyz_centered[:,:3], axis=1, keepdims=True)
    return -scanner_LOS 


@njit()
def compute_roughness(pcd: np.ndarray, normal: np.ndarray, point: np.ndarray) -> np.ndarray:
    # Compute the mean of the neighborhood points along axis 0 manually
    # Approximates the best fit plane to the neighborhood points
    mean_pt_nbh = np.sum(pcd[:,:3], axis=0) / pcd.shape[0]
    
    # Compute the distance from the points to the plane
    return np.abs(np.dot(point - mean_pt_nbh, normal))


@njit()
def scanline_neighborhood_points(pcd, i, theta_range_reference, knickpoints_dict, scanline_id_arr, center_point):
    scanline_minus1 = pcd[knickpoints_dict[scanline_id_arr[i - 1]][0]:knickpoints_dict[scanline_id_arr[i - 1]][1], :3]
    scanline_minus2 = pcd[knickpoints_dict[scanline_id_arr[i - 2]][0]:knickpoints_dict[scanline_id_arr[i - 2]][1], :3]
    scanline_minus3 = pcd[knickpoints_dict[scanline_id_arr[i - 3]][0]:knickpoints_dict[scanline_id_arr[i - 3]][1], :3]
    scanline_minus4 = pcd[knickpoints_dict[scanline_id_arr[i - 4]][0]:knickpoints_dict[scanline_id_arr[i - 4]][1], :3]
    scanline_minus5 = pcd[knickpoints_dict[scanline_id_arr[i - 5]][0]:knickpoints_dict[scanline_id_arr[i - 5]][1], :3]
    
    scanline_plus1 = pcd[knickpoints_dict[scanline_id_arr[i + 1]][0]:knickpoints_dict[scanline_id_arr[i + 1]][1], :3]
    scanline_plus2 = pcd[knickpoints_dict[scanline_id_arr[i + 2]][0]:knickpoints_dict[scanline_id_arr[i + 2]][1], :3]
    scanline_plus3 = pcd[knickpoints_dict[scanline_id_arr[i + 3]][0]:knickpoints_dict[scanline_id_arr[i + 3]][1], :3]
    scanline_plus4 = pcd[knickpoints_dict[scanline_id_arr[i + 4]][0]:knickpoints_dict[scanline_id_arr[i + 4]][1], :3]
    scanline_plus5 = pcd[knickpoints_dict[scanline_id_arr[i + 5]][0]:knickpoints_dict[scanline_id_arr[i + 5]][1], :3]

    scanline_neighborhood_minus1 = scanline_minus1[np.searchsorted(scanline_minus1[:, 9], theta_range_reference), :]
    scanline_neighborhood_minus2 = scanline_minus2[np.searchsorted(scanline_minus2[:, 9], theta_range_reference), :]
    scanline_neighborhood_minus3 = scanline_minus3[np.searchsorted(scanline_minus3[:, 9], theta_range_reference), :]
    scanline_neighborhood_minus4 = scanline_minus4[np.searchsorted(scanline_minus4[:, 9], theta_range_reference), :]
    scanline_neighborhood_minus5 = scanline_minus5[np.searchsorted(scanline_minus5[:, 9], theta_range_reference), :]
    
    scanline_neighborhood_plus1 = scanline_plus1[np.searchsorted(scanline_plus1[:, 9], theta_range_reference), :]
    scanline_neighborhood_plus2 = scanline_plus2[np.searchsorted(scanline_plus2[:, 9], theta_range_reference), :]
    scanline_neighborhood_plus3 = scanline_plus3[np.searchsorted(scanline_plus3[:, 9], theta_range_reference), :]
    scanline_neighborhood_plus4 = scanline_plus4[np.searchsorted(scanline_plus4[:, 9], theta_range_reference), :]
    scanline_neighborhood_plus5 = scanline_plus5[np.searchsorted(scanline_plus5[:, 9], theta_range_reference), :]
    
    # Merge the scanlines
    scanline_neighborhoods = np.vstack((
    scanline_neighborhood_minus1,
    scanline_neighborhood_minus2,
    scanline_neighborhood_minus3,
    scanline_neighborhood_minus4,
    scanline_neighborhood_minus5,
    scanline_neighborhood_plus1,
    scanline_neighborhood_plus2,
    scanline_neighborhood_plus3,
    scanline_neighborhood_plus4,
    scanline_neighborhood_plus5,
    ))
    
    difference = scanline_neighborhoods - center_point
    distances = np.sqrt(np.sum(difference**2, axis=1))
    
    if scanline_neighborhoods.shape[0] >= 30:
        nearest_neighborhood_points = scanline_neighborhoods[np.argsort(distances)][:30]
    else:
        nearest_neighborhood_points = scanline_neighborhoods
    
    return nearest_neighborhood_points


# @njit()
# def scanline_neighborhood_points(pcd, i, theta_range_reference, knickpoints_dict, scanline_id_arr, center_point):
#     num_points = 15
#     num_neighborhoods = 11
#     num_nearest_points = 15
    
#     nearest_neighborhood_points = np.zeros((5000, 3))
    
#     for k in prange(num_neighborhoods):
#         idx = knickpoints_dict[scanline_id_arr[i + k - 5]][0]
#         end_idx = knickpoints_dict[scanline_id_arr[i + k - 5]][1]
#         scanline = pcd[idx:end_idx, :3]
#         neighborhood = scanline[np.searchsorted(scanline[:, 9], theta_range_reference), :]
#         nearest_neighborhood_points[k*neighborhood.shape[0]:(k+1)*neighborhood.shape[0]] = neighborhood

#     difference = nearest_neighborhood_points - center_point
#     distances = np.sqrt(np.sum(difference**2, axis=1))
    
#     if nearest_neighborhood_points.shape[0] >= num_nearest_points:
#         nearest_neighborhood_points = nearest_neighborhood_points[np.argsort(distances)][:num_nearest_points]
    
#     return nearest_neighborhood_points


# @njit()
# def scanline_neighborhood_points(pcd, i, theta_range_reference, knickpoints_dict, scanline_id_arr, center_point):
#     offsets = np.arange(-5, 6)
#     offsets = offsets[offsets != 0]  # Exclude 0

#     # Get the start and end indices for each scanline
#     indices = np.array([knickpoints_dict[scanline_id_arr[i + offset]] for offset in offsets])

#     # Get all the scanlines
#     scanlines = [pcd[start:end, :3] for start, end in zip(indices[:, 0], indices[:, 1])]

#     # Get the neighborhood for each scanline
#     scanline_neighborhoods = [scanline[np.searchsorted(scanline[:, 9], theta_range_reference), :] for scanline in scanlines]

#     # Calculate distances
#     distances = [np.sqrt(np.sum((neighborhood - center_point)**2)) for neighborhood in scanline_neighborhoods]

#     # Get the 30 nearest neighborhood points or all points if less than 30
#     if len(scanline_neighborhoods) >= 30:
#         nearest_neighborhood_points = [scanline_neighborhoods[i] for i in np.argsort(distances)[:30]]
#     else:
#         nearest_neighborhood_points = scanline_neighborhoods

#     return nearest_neighborhood_points


@njit(parallel=True)
def calculate_normals_curvature(pcd, knickpoints_dict, scanline_id_arr, scanner_LOS):
    curvature_pseudo_3d = np.zeros((pcd.shape[0]))
    normals_pseudo_3d = np.zeros((pcd.shape[0], 3))
    roughness_pseudo_3d = np.zeros((pcd.shape[0]))
    mean_dist = np.zeros((pcd.shape[0]))
    std_dist = np.zeros((pcd.shape[0]))
    quality = np.zeros((pcd.shape[0]))
    zenith_angle = np.zeros((pcd.shape[0]))

    for i in prange(scanline_id_arr.shape[0]):
        start_idx = knickpoints_dict[scanline_id_arr[i]][0]
        end_idx = knickpoints_dict[scanline_id_arr[i]][1]
        scanline_reference = pcd[start_idx:end_idx, :].copy()

        for point_idx in range(scanline_reference.shape[0]):
            start_index = max(0, point_idx - 10)
            end_index = min(scanline_reference.shape[0], point_idx + 11)

            point_nghb_scanline_reference = scanline_reference[start_index:end_index, :]
            theta_range_reference = point_nghb_scanline_reference[:, 9]
            
            nearest_neighborhood_points = scanline_neighborhood_points(pcd, i, theta_range_reference, knickpoints_dict, scanline_id_arr, pcd[start_idx + point_idx, :3])

            if nearest_neighborhood_points.shape[0] > point_nghb_scanline_reference.shape[0]:
                center_pos = np.array([nearest_neighborhood_points[:, 0].mean(), nearest_neighborhood_points[:, 1].mean(), nearest_neighborhood_points[:, 2].mean()])
                nearest_neighborhood_points -= center_pos
                eigenvalues, eigenvectors = np.linalg.eigh(np.cov(nearest_neighborhood_points.T))
                normal = eigenvectors[:, 0]

                if np.dot(normal, scanner_LOS[start_idx + point_idx, :3]) < 0:
                    normal *= -1
                    
                curvature = np.min(eigenvalues) / np.sum(eigenvalues)

                normals_pseudo_3d[start_idx + point_idx, :] = normal 
                curvature_pseudo_3d[start_idx + point_idx] = curvature
                roughness_pseudo_3d[start_idx + point_idx] = compute_roughness(nearest_neighborhood_points, normal, pcd[start_idx + point_idx, :3])
                zenith_angle[start_idx + point_idx] = np.degrees(np.arccos(normal[2]))

    return normals_pseudo_3d, curvature_pseudo_3d, roughness_pseudo_3d, zenith_angle


def compute_kdtree(pcd: np.ndarray, number_neighbors: float) -> Tuple[np.ndarray, np.ndarray]:
    scanner_pos = np.mean(pcd[:,:3], axis=0)
    pcd_xyz_centered = pcd[:,:3] - scanner_pos
    
    # Build a k-d tree from point_clouds for efficient nearest neighbor search
    kdtree = cKDTree(pcd_xyz_centered)
    
    _, indices = kdtree.query(pcd_xyz_centered, k=number_neighbors, workers=-1)
    
    point_clouds = pcd_xyz_centered[indices]
    
    return indices, point_clouds


@njit(parallel=True)
def compute_covariance_attributes(indices: np.ndarray, 
                                  point_clouds: np.ndarray,
                                  pcd: np.ndarray, 
                                  scanner_LOS: np.ndarray) -> numba.typed.Dict:
    # Initialize the arrays
    normals = np.zeros((indices.shape[0],3))
    roughness = np.zeros(indices.shape[0])
    curvature = np.zeros(indices.shape[0])
    zenith_angle = np.zeros(indices.shape[0])

    # Loop over all indices in parallel
    for i in prange(indices.shape[0]):
        point_cloud = point_clouds[i]

        if point_cloud.shape[0] > 2:
            center_pos = np.array([point_cloud[:, 0].mean(), point_cloud[:, 1].mean(), point_cloud[:, 2].mean()])
            point_cloud[:,:3] -= center_pos
            
            # Compute the covariance matrix of point_cloud and find its eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(point_cloud[:,:3].T))

            # The first eigenvector (corresponding to the smallest eigenvalue) is the normal of the point cloud
            normal = eigenvectors[:, 0]
            
            # Align the normal with the scanner position
            if np.dot(normal, scanner_LOS[i, :3]) < 0:
                normal *= -1
            
            normals[i, :] = normal
            curvature[i] = np.min(eigenvalues) / np.sum(eigenvalues)
            roughness[i] = compute_roughness(point_cloud, normal, pcd[i, :3])
            zenith_angle[i] = np.degrees(np.arccos(normal[2]))
        else:
            continue     

    return normals, roughness, curvature, zenith_angle 


def main():
    pcd_path = root_dir / "data/04_scanline_extraction/SiteA_RHV_01_Labeled_scnln_thetaoriginal.txt"
    print('Import the point cloud')
    pcd = import_pcd(pcd_path)
    print('Preprocessing the point cloud')
    knickpoints_dict, scanline_id_arr = preprocessing(pcd)
    print('Sorting the point cloud')
    pcd = sort_theta(pcd, knickpoints_dict, scanline_id_arr)
    print('Computing the scanner line of sight')
    scanner_LOS = compute_scanner_LOS(pcd)
    
    # 3D scanline approach 
    print('Running the 3d scanline approach')
    start_time = time.time()
    _, curvature_pseudo_3d, roughness_pseudo_3d, zenith_angle_pseudo_3d = calculate_normals_curvature(pcd, knickpoints_dict, scanline_id_arr, scanner_LOS)
    end_time = time.time()
    execution_time_3D_scanline = end_time - start_time
    print("Execution time of 3D scanline approach is: ", execution_time_3D_scanline, "seconds")
    
    # # 3D KdTree approach
    # print('Running the KdTree approach')
    # start_time = time.time()
    # indices, point_clouds = compute_kdtree(pcd, 15)
    # print(indices.shape)
    # print(point_clouds.shape)
    # _, roughness, curvature, zenith_angle = compute_covariance_attributes(indices, point_clouds, pcd, scanner_LOS)
    # end_time = time.time()
    # execution_time_kdtree = end_time - start_time
    # print("Execution time of kdtree approach is: ", execution_time_kdtree, "seconds")
    
    # time_df = pd.DataFrame(data={"execution_time": [execution_time_3D_scanline, execution_time_kdtree],
    #                              "nearest_neighbor_search": [50, 50],
    #                              "method": ["3D_scanline", "KdTree"]})
    
    # time_df_out_path = root_dir / "data/08_3D_scanline_approach/comparison_kdtree"
    # time_df_out_path.mkdir(parents=False, exist_ok=True)
    # csv_file = time_df_out_path / "execution_time2.csv"
    
    # if csv_file.exists():
    #     time_df.to_csv(csv_file, mode='a', header=False, index=False)
    # else:
    #     time_df.to_csv(csv_file, mode='w', header=True, index=False)
    
    
    # Save the point cloud
    print('Saving the point cloud')
    pcd_out = np.c_[pcd[:,:3], curvature_pseudo_3d, roughness_pseudo_3d, zenith_angle_pseudo_3d, pcd[:,8], pcd[:,9], pcd[:, 14]] #roughness, curvature, zenith_angle
    out_path = root_dir / "data/08_3D_scanline_approach/comparison_kdtree/scanline3d_kdtree_pcd_out2.txt"
    np.savetxt(out_path, pcd_out, delimiter=" ", fmt="%.6f")
    


if __name__ == "__main__":
    main()

