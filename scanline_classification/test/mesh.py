import numpy as np
import open3d as o3d
from pathlib import Path
from numba import njit, prange

root_dir = Path(__file__).parent.parent.parent.absolute()

pcd1 = np.loadtxt(Path(root_dir) / "data/05_segmentation/SiteA_Scans_Global_I_RGB_RHV/SiteA_RHV_01_Labeled_Segmentation.txt", delimiter=' ')
pcd2 = np.loadtxt(Path(root_dir) / "data/05_segmentation/SiteA_Scans_Global_I_RGB_RHV/SiteA_RHV_02_Labeled_Segmentation.txt", delimiter=' ')

# stack the two point clouds
pcd = np.vstack((pcd1, pcd2))


pcd_xyz = pcd[:,:3]
pcd_colors = pcd[:,4:7]
normals = pcd[:,-3:]
print(normals.shape)

pcd_o3d = o3d.geometry.PointCloud()
pcd_o3d.points = o3d.utility.Vector3dVector(pcd_xyz - np.mean(pcd_xyz, axis=0))
pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_colors / 255.)
pcd_o3d.normals = o3d.utility.Vector3dVector(normals)

# downsample
pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=0.1)
distances = pcd_o3d.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
print(f'Average distance: {avg_dist}')
radius = 1 * avg_dist

bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_o3d, o3d.utility.DoubleVector([radius, radius * 2]))

bpa_mesh.remove_degenerate_triangles()
bpa_mesh.remove_duplicated_triangles()
bpa_mesh.remove_duplicated_vertices()
bpa_mesh.remove_non_manifold_edges()


o3d.visualization.draw_geometries([bpa_mesh])