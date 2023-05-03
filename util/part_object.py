import torch
import pytorch3d
import random
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
import os
from sklearn.neighbors import NearestNeighbors

from knn_cuda import KNN

import trimesh
import numpy as np

from simpleicp import PointCloud, SimpleICP
from pytorch3d.ops.points_alignment import iterative_closest_point
from pytorch3d.transforms import euler_angles_to_matrix

import open3d as o3d
# from knn_cuda import KNN

from .visualize import show_point_clouds
from .open3d_util import open3d_preprocess_pcd, open3d_ransac, open3d_icp, open3d_fast_global_registration
from .algebra import get_rotation_matrix

class PartObjSet:
    def __init__(self, folder_path, num_total_point=50000, broken_threshold=0.01):
        # get all obj files
        self.objs:list[PartObj] = self._parse_obj_files(folder_path, num_total_point, broken_threshold)


    def __getitem__(self, index):
        return self.objs[index]


    def _parse_obj_files(self, folder_path, num_total_point, broken_threshold):

        print("loading objects...")
        obj_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.obj')]
        meshes = [trimesh.load_mesh(file_name) for file_name in obj_files]

        print("sampling point clouds...")

        surface_areas = torch.Tensor([mesh.area for mesh in meshes])
        n_pt_per_mesh = (num_total_point * surface_areas / surface_areas.sum()).int().tolist()

        pcds = []
        normals = []
        for mesh, n_pt in zip(meshes, n_pt_per_mesh):
            samples, face_index = trimesh.sample.sample_surface(mesh, n_pt)
            normal = mesh.face_normals[face_index]
            normals.append(torch.Tensor(normal))
            pcds.append(torch.Tensor(samples)) # N * 10000 * 3

        print("finding broken points clouds...")
        broken_indices = self._find_broken_point(pcds, broken_threshold) # N * B_i * 3
        
        objs = []
        for i in range(len(obj_files)):
            broken_idx = broken_indices[i]
            pcd = pcds[i]
            normal = normals[i]
            objs.append(PartObj(mesh=meshes[i], 
                                broken_pcd=pcd[broken_idx], 
                                ordinary_pcd=pcd[torch.logical_not(broken_idx)], 
                                broken_normal=normal[broken_idx], 
                                ordinary_normal=normal[torch.logical_not(broken_idx)]
                            ))
        print(f"Loaded {len(obj_files)} objects from {folder_path}")
        return objs


    def _knn(self, src, dst, k=1, is_naive=False):
        """return k nearest neighbors using GPU"""
        if len(src) * len(dst) > 10e8:
            # TODO: optimize memory through recursion
            pass

        assert(len(src.shape) == 2)
        assert(len(dst.shape) == 2)
        assert(src.shape[-1] == dst.shape[-1])
        src = src.cuda()
        dst = dst.cuda()
        
        if is_naive:

            src = src.reshape(-1, 1, src.shape[-1])
            distance = torch.norm(src - dst, dim=-1)

            knn = distance.topk(k, largest=False)
            distance = knn.values.ravel().cpu()
            indices = knn.indices.ravel().cpu()
        else:
            knn = KNN(k=1, transpose_mode=True)
            distance, indices = knn(src, dst)

        return distance, indices


    def _find_broken_point(self, points, d):
        indices = []

        for i in range(len(points)):
            idx_i = torch.zeros(len(points[i]))
            idx_i = idx_i.to(torch.bool)

            for j in range(len(points)):
                if i == j:
                    continue
                distances, _ = self._knn(points[i], points[j])
                idx_i = torch.logical_or(idx_i, distances < d)
            indices.append(idx_i)
                
        return indices


    def show(self, indice, total_num=3000, normal=False):
        if isinstance(indice, int):
            indice = [indice]
            
        broken_pcds = self.objs[indice[0]].broken_pcd
        ordinary_pds = self.objs[indice[0]].ordinary_pcd
        for i, idx in enumerate(indice):
            if i == 0:
                continue
            broken_pcds = torch.cat([broken_pcds, self.objs[idx].broken_pcd], axis=0)
            ordinary_pds = torch.cat([ordinary_pds, self.objs[idx].ordinary_pcd], axis=0)

        # generate randome index in numpy
        rand_idx0 = np.random.choice(broken_pcds.shape[0], total_num)
        rand_idx1 = np.random.choice(ordinary_pds.shape[0], total_num)

        broken_pcds = broken_pcds[rand_idx0]
        ordinary_pds = ordinary_pds[rand_idx1]

        if not normal:
            show_point_clouds([broken_pcds, ordinary_pds], ['red', 'blue'])

        else:
            oridnary_normals = self.objs[indice[0]].ordinary_normal
            broken_normals = self.objs[indice[0]].broken_normal
            
            for i, idx in enumerate(indice):
                if i == 0:
                    continue
                broken_normals = np.append([broken_normals, self.objs[idx].broken_pcd], axis=0)
                oridnary_normals = np.append([oridnary_normals, self.objs[idx].ordinary_pcd], axis=0)
            broken_normals = broken_normals[rand_idx0]
            oridnary_normals = oridnary_normals[rand_idx1]

            show_point_clouds([broken_pcds, ordinary_pds], ['red', 'blue'],
                              normals=[broken_normals, oridnary_normals])
        

    def show_all(self, total_num=3000, normal=False):
        self.show(list(range(len(self.objs))), total_num, normal)


    def _get_light_to_dark_blue_colors(self, length):
        r = np.array([0, 173])
        g = np.array([0, 216])
        b = np.array([139, 230])

        rs = np.linspace(0, 173, num=5)

        # perform linear interpolation
        gs = np.interp(rs, r, g)
        bs = np.interp(rs, r, b)

        return np.stack([rs, gs, bs]).T


    def get_volume(self, idx):
        return self.objs[idx].mesh.volume


    def get_overlap(self, idx1, idx2):
        mesh1 = self.objs[idx1].mesh
        mesh2 = self.objs[idx2].mesh
        return mesh1.intersection(mesh2, engine='scad')


    def union(self, idx1, idx2, delete_threshold=0.02):
        # # union two meshes
        mesh1 = self.objs[idx1].mesh
        # mesh2 = self.objs[idx2].mesh
        # mesh = mesh1.union(mesh2, engine='scad')

        # union two ordinary point clouds
        ordinary_pcd1 = self.objs[idx1].ordinary_pcd
        ordinary_pcd2 = self.objs[idx2].ordinary_pcd
        ordinary_pcd = torch.cat([ordinary_pcd1, ordinary_pcd2], axis=0)


        # union two ordinary normal
        ordinary_normal1 = self.objs[idx1].ordinary_normal
        ordinary_normal2 = self.objs[idx2].ordinary_normal
        ordinary_normal = torch.cat([ordinary_normal1, ordinary_normal2], axis=0)

        # eXclusive OR (XOR) two broken point clouds
        broken_pcd1 = self.objs[idx1].broken_pcd
        broken_pcd2 = self.objs[idx2].broken_pcd

        distances1, _ = self._knn(broken_pcd1, broken_pcd2)
        distances2, _ = self._knn(broken_pcd2, broken_pcd1)
        
        indice1 = distances1 > delete_threshold
        indice2 = distances2 > delete_threshold
        
        # eXclusive OR (XOR) two broken normals
        broken_normal1 = self.objs[idx1].broken_normal
        broken_normal2 = self.objs[idx2].broken_normal

        broken_pcd = torch.cat([broken_pcd1[indice1], 
                                broken_pcd2[indice2]], axis=0)
        broken_normal = torch.cat([broken_normal1[indice1], 
                                  broken_normal2[indice2]], axis=0)
        
        
        # append new object
        new_obj = PartObj(mesh1, broken_pcd, ordinary_pcd, broken_normal, ordinary_normal)
        self.objs.append(new_obj)

        # delete two previous objects
        self.objs[idx1] = empty_obj
        self.objs[idx2] = empty_obj



    def volume_overlap(self, idx1, idx2):
        mesh1 = self.objs[idx1].mesh
        mesh2 = self.objs[idx2].mesh
        return mesh1.intersection(mesh2)


    def align_location(self, src_idx, dst_idx):
        src_mean = self.objs[src_idx].broken_pcd.mean(axis=0)
        dst_mean = self.objs[dst_idx].broken_pcd.mean(axis=0)

        vector = dst_mean - src_mean
        self.objs[src_idx].translate(vector[0], vector[1], vector[2])


    def align_normal(self, src_idx, dst_idx):
        """the mean normal of a broken point is oposite of the mean normal another broken point if they are adjacent"""
        normal1 = self.objs[src_idx].broken_normal
        normal2 = self.objs[dst_idx].broken_normal
        normal1 = torch.mean(normal1, axis=0)
        normal2 = torch.mean(normal2, axis=0)
        normal1 = normal1 / normal1.norm()
        normal2 = normal2 / normal2.norm()
        
        rotation_matrix = get_rotation_matrix(normal1, -normal2)
        
        self.objs[src_idx].rotate_via_matrix(rotation_matrix)
        normal1 = self.objs[src_idx].broken_normal
        normal2 = self.objs[dst_idx].broken_normal
        normal1 = torch.mean(normal1, axis=0)
        normal2 = torch.mean(normal2, axis=0)
        normal1 = normal1 / normal1.norm()
        normal2 = normal2 / normal2.norm()


    def random_transform(self):
        for obj in self.objs:
            obj.random_transform()

    
    def ransac(self, src_idx, dst_idx, voxel_size=0.003, normal_angle=0.05):
        src_pcd = self.objs[src_idx].broken_pcd
        dst_pcd = self.objs[dst_idx].broken_pcd
        src_normal = self.objs[src_idx].broken_normal
        dst_normal = self.objs[dst_idx].broken_normal
        
        src, src_down, source_fpfh = open3d_preprocess_pcd(src_pcd, src_normal, voxel_size)
        dst, dst_down, target_fpfh = open3d_preprocess_pcd(dst_pcd, dst_normal, voxel_size)

        transformation = open3d_ransac(src, dst, source_fpfh, target_fpfh, voxel_size, normal_angle)

        self.objs[src_idx].transform(transformation)


    def fast_global_registration(self, src_idx, dst_idx, voxel_size=0.01):
        src_pcd = self.objs[src_idx].broken_pcd
        dst_pcd = self.objs[dst_idx].broken_pcd
        src_normal = self.objs[src_idx].broken_normal
        dst_normal = self.objs[dst_idx].broken_normal
        
        src, src_down, source_fpfh = open3d_preprocess_pcd(src_pcd, src_normal, voxel_size)
        dst, dst_down, target_fpfh = open3d_preprocess_pcd(dst_pcd, dst_normal, voxel_size)

        transformation = open3d_fast_global_registration(src, dst, source_fpfh, target_fpfh, voxel_size)

        self.objs[src_idx].transform(transformation)


    def icp(self, src_idx, dst_idx, library='open3d'):
        library = library.lower()
        if library in ['torch', 'pytorch', 'torch3d', 'pytorch3d']:

            src = self.objs[src_idx].broken_pcd
            dst = self.objs[dst_idx].broken_pcd

            result = iterative_closest_point(src.reshape((1, ) + src.shape), dst.reshape((1, ) + dst.shape), verbose=True)

            if result.converged:
                self.objs[src_idx].rotate_via_matrix(result.RTs.R.reshape(3, 3))
                self.objs[src_idx].translate(result.RTs.T[0][0], result.RTs.T[0][1], result.RTs.T[0][2])
            
            return result.converged, result.rmse, result.RTs

        elif library == 'simpleicp':
            src = self.objs[src_idx].broken_pcd
            dst = self.objs[dst_idx].broken_pcd

            pc_fix = PointCloud(dst, columns=["x", "y", "z"])
            pc_mov = PointCloud(src, columns=["x", "y", "z"])

            icp = SimpleICP()
            icp.add_point_clouds(pc_fix, pc_mov)
            H, X_mov_transformed, rigid_body_transformation_params = icp.run(max_overlap_distance=1)

            R = torch.tensor(H[:3, :3], dtype=torch.float32)
            T = torch.tensor(H[3, :3], dtype=torch.float32)
            self.objs[src_idx].rotate_via_matrix(R)
            self.objs[src_idx].translate(T[0], T[1], T[2])

            return True, None, None
        
        elif library == 'open3d':

            src_pcd = self.objs[src_idx].broken_pcd
            dst_pcd = self.objs[dst_idx].broken_pcd
            src_normal = self.objs[src_idx].broken_normal
            dst_normal = self.objs[dst_idx].broken_normal

            src, src_down, source_fpfh = open3d_preprocess_pcd(src_pcd, src_normal)
            dst, dst_down, target_fpfh = open3d_preprocess_pcd(dst_pcd, dst_normal)

            transformation = open3d_icp(src, dst, voxel_size=0.01)
            self.objs[src_idx].transform(transformation)
    


def skip_if_empty(func):
    def wrapper(self, *args, **kwargs):
        if len(self.ordinary_pcd) == 0:
            return
        return func(self, *args, **kwargs)
    return wrapper



class PartObj:
    def __init__(self, mesh, broken_pcd, ordinary_pcd, broken_normal=None, ordinary_normal=None):
        self.mesh = mesh
        self.broken_pcd = torch.Tensor(broken_pcd)
        self.ordinary_pcd = torch.Tensor(ordinary_pcd)

        if broken_normal is None or ordinary_normal is None:
            # estimate normals
            pcd = torch.cat([self.broken_pcd, self.ordinary_pcd], axis=0)
            pcd = PointCloud(pcd, columns=["x", "y", "z"])
            pcd.estimate_normals(neighbors=10)

            normal = pcd[["nx", "ny", "nz"]].to_numpy()
            normal = torch.Tensor(normal)

            self.broken_normal = normal[:self.broken_pcd.shape[0]]
            self.ordinary_normal = normal[self.broken_pcd.shape[0]:]
        else:
            self.broken_normal = torch.Tensor(broken_normal)
            self.ordinary_normal = torch.Tensor(ordinary_normal)

    @skip_if_empty
    def transform(self, H):
        if isinstance(H, (np.ndarray, np.generic)):
            H = torch.Tensor(H)

        self.rotate_via_matrix(H[:3, :3])
        self.translate(H[0, 3], H[1, 3], H[2, 3])


    @skip_if_empty
    def random_transform(self):
        
        self.rotate(random.uniform(0, 2 * torch.pi), 
                    random.uniform(0, 2 * torch.pi), 
                    random.uniform(0, 2 * torch.pi))
        self.translate(random.uniform(0, 0.1), 
                    random.uniform(0, 0.1), 
                    random.uniform(0, 0.1))
        

    @skip_if_empty
    def translate(self, dx=0, dy=0, dz=0):
        
        self.mesh.vertices += np.array([[dx, dy, dz]])
        self.broken_pcd += torch.Tensor([[dx, dy, dz]])
        self.ordinary_pcd += torch.Tensor([[dx, dy, dz]])


    @skip_if_empty
    def rotate(self, theta_x=0, theta_y=0, theta_z=0):
        
        angle = torch.Tensor([theta_x, theta_y, theta_z])
        rotation = euler_angles_to_matrix(angle, convention='XYZ')

        self.mesh.vertices = self.mesh.vertices @ rotation.T.cpu().numpy()

        self.broken_pcd = self.broken_pcd @ rotation.T
        self.ordinary_pcd = self.ordinary_pcd @ rotation.T

        self.broken_normal = self.broken_normal @ rotation.T
        self.ordinary_normal = self.ordinary_normal @ rotation.T
    

    @skip_if_empty
    def rotate_via_matrix(self, rotation):
        
        if isinstance(rotation, (np.ndarray, np.generic)):
            rotation = torch.Tensor(rotation)
            
        self.mesh.vertices = self.mesh.vertices @ rotation.T.cpu().numpy()

        self.broken_pcd = self.broken_pcd @ rotation.T
        self.ordinary_pcd = self.ordinary_pcd @ rotation.T

        self.broken_normal = self.broken_normal @ rotation.T
        self.ordinary_normal = self.ordinary_normal @ rotation.T


empty_obj = PartObj(mesh=None, broken_pcd=[], ordinary_pcd=[], broken_normal=[], ordinary_normal=[])
