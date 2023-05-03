import numpy as np
import pymeshlab
from sklearn.neighbors import NearestNeighbors

import trimesh
import numpy as np

def nearest_neighbor(src, dst):
    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def find_near_points(src, dst, d):
    distances, indices = nearest_neighbor(src, dst)
    near_indices = np.where(distances <= d)
    src_indices = near_indices[0]
    dst_indices = indices[near_indices]

    return src_indices, dst_indices

def find_near_points_multiple_point_clouds(points, d):
    """
    Find indices of points in each point cloud that have at least one neighbor within distance d.
    Input:
        points: List of Nxm arrays of point clouds
        d: threshold distance
    Output:
        result: List of tuples containing indices of points in each point cloud with at least one neighbor within distance d
    """
    num_point_clouds = len(points)
    indices = []

    for i in range(num_point_clouds):
        for j in range(num_point_clouds):
            if i == j:
                continue
            src_indices, dst_indices = find_near_points(points[i], points[j], d)
            if len(indices) == i:
                indices.append(src_indices)
            elif len(src_indices) > 0:
                indices[i] = np.append(indices[i], src_indices)

    return indices

def mesh_intersection(mesh1, mesh2, file_loc):
    # Save intersection of two meshes
    ms = pymeshlab.MeshSet()

    # Add the input meshes to the MeshSet
    ms.load_new_mesh(mesh1)
    ms.load_new_mesh(mesh2)

    # Apply the boolean intersection filter
    ms.apply_filter(
        'generate_boolean_intersection',
        first_mesh=0,
        second_mesh=1,
        transfer_face_color=False,
        transfer_face_quality=False,
        transfer_vert_color=False,
        transfer_vert_quality=False)
    
    ms.save_current_mesh(file_loc)


def calculate_mesh_volume(mesh_file):
    # Load the mesh
    mesh = trimesh.load_mesh(mesh_file)
    return mesh.volume