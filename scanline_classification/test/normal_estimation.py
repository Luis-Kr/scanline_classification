import numpy as np
import open3d as o3d
from pathlib import Path
from numba import njit, prange

root_dir = Path(__file__).parent.parent.parent.absolute()

@njit()
def get_scanline(pcd, scanline_id):
    return pcd[np.where(pcd[:, 11] == scanline_id)[0]][:,:3]


@njit()
def compute_normals(pcd_scanline): 
    normals = np.zeros((pcd_scanline.shape[0], 3))

    for i in prange(pcd_scanline.shape[0]):
        if i == 0:
            normal = np.cross(pcd_scanline[i+1], pcd_scanline[i])
            # normalize
            normals[i] = ( normal / np.linalg.norm(normal) ) / 5
        elif i <= pcd_scanline.shape[0]-2:
            normal = np.cross(pcd_scanline[i-1], pcd_scanline[i+1])
            normals[i] = (normal / np.linalg.norm(normal)) / 5
        else:
            normal = np.cross(pcd_scanline[i-1], pcd_scanline[i])
            normals[i] = (normal / np.linalg.norm(normal)) / 5

    return normals

@njit(parallel=True)
def pcd_compute_normals(pcd, segment_classes, counts, sorted_indices):
    indices_per_class = np.split(sorted_indices, np.cumsum(counts[:-1]))
    normals = np.zeros((pcd.shape[0], 3))
    
    for i in prange(segment_classes.shape[0]):
        segment_indices = indices_per_class[i]
        normals_i = compute_normals(pcd[segment_indices, :3])
        
        for j in prange(segment_indices.shape[0]):
            normals[segment_indices[j]] = normals_i[j]
            
    return normals, pcd[sorted_indices, :]



if __name__ == "__main__":
    # Read the pcd file
    print("Reading the pcd file...")
    pcd = np.loadtxt(Path(root_dir) / "data/03_raw_plus_scanline_plus_segmentation/SiteA_Scans_Global_I_RGB_RHV/Scan01_ScanlineID_Segmentation.asc", delimiter=' ')

    PART_A = False
    PART_B = True
    
    # Part A
    if PART_A:
        print("Calculating the normals...")
        scanline_pcd1 = get_scanline(pcd, 1) # 1 normals upwards facing, 2000 normals downwards facing, 750 normals lateral facing
        scanline_pcd2000 = get_scanline(pcd, 2000)
        scanline_pcd750 = get_scanline(pcd, 750)
        normals1 = compute_normals(scanline_pcd1)
        normals2000 = compute_normals(scanline_pcd2000)
        normals750 = compute_normals(scanline_pcd750)
        
        pcd_o3d1 = o3d.geometry.PointCloud()
        pcd_o3d1.points = o3d.utility.Vector3dVector(np.vstack((scanline_pcd1, scanline_pcd2000, scanline_pcd750)) - np.min(pcd[:,:3], axis=0))
        pcd_o3d1.normals = o3d.utility.Vector3dVector(np.vstack((normals1, normals2000, normals750)) / 2)

        o3d.visualization.draw_geometries([pcd_o3d1])
    
    
    # Part B
    if PART_B:
        print("Calculating the normals...")
        sorted_indices = np.lexsort(np.rot90(pcd[:,(11,9)]))
        pcd = pcd[sorted_indices, :]
        segment_classes = np.array(list(set(pcd[:,11])))
        _, counts = np.unique(pcd[:,11], return_counts=True)
        normals, pcd_sorted = pcd_compute_normals(pcd, segment_classes, counts, sorted_indices)
        
        pcd_xyz = pcd_sorted[::40, :3] - np.mean(pcd_sorted[:,:3], axis=0)
        colors = pcd_sorted[::40, [4,5,6]]
        colors /= 255.
        normals = normals[::40]

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_xyz)
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
        pcd_o3d.normals = o3d.utility.Vector3dVector(normals / 2)

        o3d.visualization.draw_geometries([pcd_o3d])