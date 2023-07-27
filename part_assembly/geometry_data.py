import os
import random

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

from torch.utils.data import DataLoader
import torch

# from knn_cuda import KNN
from functools import lru_cache
import jhutil

from copy import copy
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Dataset
from time import time
import shutil


class GeometryPartDataset(Dataset):
    """Geometry part assembly dataset.

    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
        self,
        data_dir,
        data_fn,
        category='',
        sample_weight=150000,
        num_points=1000,
        min_num_part=2,
        max_num_part=1000,
        shuffle_parts=False,
        rot_range=-1,
        overfit=-1,
        scale=1,
        max_sample=20000,
    ):
        import jhutil; jhutil.jhprint(1111, )
        # for training stage1
        self.max_sample = max_sample
        self.data_fn = data_fn
        self.num_obj_path = f"{self.data_fn.split('.txt')[0]}.num_obj_dir"
        self.category = category if category.lower() != 'all' else ''
        self.num_points = num_points
        self.sample_weight = sample_weight
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part  # ignore shapes with more parts
        self.shuffle_parts = shuffle_parts  # shuffle part orders
        self.scale = scale
        self.rot_range = rot_range  # rotation range in degree
        self.overfit = overfit

        import jhutil; jhutil.jhprint(2222, )
        super().__init__(root=data_dir, transform=None, pre_transform=None)
        import jhutil; jhutil.jhprint(3333, )
        self.num_objs = self.load_num_objs()

    def get_length_list(self):
        data_lengs = []
        for data_folder in self.raw_file_names:
            data_folder = os.path.join(self.raw_dir, data_folder)
            file_names = os.listdir(data_folder)
            data_lengs.append(len(file_names))
        return data_lengs

    def load_num_objs(self):
        num_objs_path = os.path.join(self.processed_dir, f"{self.data_fn.split('.txt')[0]}.num_obj_dir.pt")
        if os.path.exists(num_objs_path):
            num_objs = torch.load(num_objs_path)
            return num_objs[:self.overfit]

        """Filter out invalid number of parts."""
        with open(os.path.join(self.raw_dir, self.data_fn), 'r') as f:
            mesh_list = [line.strip() for line in f.readlines()]
            if self.category:
                mesh_list = [
                    line for line in mesh_list
                    if self.category in line.split('/')
                ]
        num_objs = []
        for mesh in mesh_list:
            mesh_dir = os.path.join(self.raw_dir, mesh)
            if not os.path.isdir(mesh_dir):
                print(f'{mesh} does not exist')
                continue
            for frac in os.listdir(mesh_dir):
                # we take both fractures and modes for training
                if 'fractured' not in frac and 'mode' not in frac:
                    continue
                frac = os.path.join(mesh, frac)
                file_names = os.listdir(os.path.join(self.raw_dir, frac))
                file_names = [fn for fn in file_names if fn.endswith('.obj')]
                num_parts = len(file_names)
                if self.min_num_part <= num_parts <= self.max_num_part:
                    num_objs.append(num_parts)

        torch.save(num_objs, num_objs_path)

        return num_objs[:self.overfit]

    @property
    @lru_cache(maxsize=1)
    def raw_file_names(self):
        """Filter out invalid number of parts."""
        with open(os.path.join(self.raw_dir, self.data_fn), 'r') as f:
            mesh_list = [line.strip() for line in f.readlines()]
            if self.category:
                mesh_list = [
                    line for line in mesh_list
                    if self.category in line.split('/')
                ]
        data_list = []
        for mesh in mesh_list:
            mesh_dir = os.path.join(self.raw_dir, mesh)
            if not os.path.isdir(mesh_dir):
                print(f'{mesh} does not exist')
                continue
            for frac in os.listdir(mesh_dir):
                # we take both fractures and modes for training
                if 'fractured' not in frac and 'mode' not in frac:
                    continue
                frac = os.path.join(mesh, frac)
                file_names = os.listdir(os.path.join(self.raw_dir, frac))
                file_names = [fn for fn in file_names if fn.endswith('.obj')]
                num_parts = len(file_names)
                if self.min_num_part <= num_parts <= self.max_num_part:
                    data_list.append(frac)

        if 0 < self.overfit < len(data_list):
            data_list = data_list[:self.overfit]
        data_list = data_list + ['tmp.tmp']
        return data_list

    @property
    def processed_file_names(self):
        return [name + ".pt" for name in self.raw_file_names]

    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = pc.mean(axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    def _get_rotation_matrix(self):
        if self.rot_range > 0.:
            rot_euler = (np.random.rand(3) - 0.5) * 2. * self.rot_range
            rot_mat = R.from_euler(
                'xyz', rot_euler, degrees=True).as_matrix()
        else:
            rot_mat = R.random().as_matrix()
        return rot_mat

    def _rotate_pc(self, pc, rot_mat):
        """pc: [N, 3]"""
        rot_mat = torch.Tensor(rot_mat)
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt

    @staticmethod
    def _shuffle_pc(pc):
        """pc: [N, 3]"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        pc = pc[order]
        return pc

    def _get_broken_pcs_idxs(self, points, threshold=0.0001):
        broken_pcs_idxs = []

        for i in range(len(points)):
            idx_i = torch.zeros(len(points[i]))
            idx_i = idx_i.to(torch.bool)

            for j in range(len(points)):
                if i == j:
                    continue
                if not self._box_overlap(points[i], points[j]):
                    continue
                distances, _ = jhutil.knn(points[i], points[j])
                idx_i = torch.logical_or(idx_i, distances < threshold)
            broken_pcs_idxs.append(idx_i)

        return broken_pcs_idxs

    def _box_overlap(self, src, ref):
        # src : (N, 3)
        # ref : (M, 3)
        src_min = src.min(axis=0)[0]  # (3,)
        src_max = src.max(axis=0)[0]  # (3,)
        ref_min = ref.min(axis=0)[0]  # (3,)
        ref_max = ref.max(axis=0)[0]  # (3,)

        # Check x-axis overlap
        if src_max[0] < ref_min[0] or src_min[0] > ref_max[0]:
            return False

        # Check y-axis overlap
        if src_max[1] < ref_min[1] or src_min[1] > ref_max[1]:
            return False

        # Check z-axis overlap
        if src_max[2] < ref_min[2] or src_min[2] > ref_max[2]:
            return False

        return True

    def _parse_data(self, data_folder):
        """Read mesh and sample point cloud from a folder."""
        # `data_folder`: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0
        data_folder = os.path.join(self.raw_dir, data_folder)
        file_names = os.listdir(data_folder)
        file_names = [fn for fn in file_names if fn.endswith('.obj')]
        if not self.min_num_part <= len(file_names) <= self.max_num_part:
            raise ValueError

        # shuffle part orders
        if self.shuffle_parts:
            random.shuffle(file_names)

        # read mesh and sample points
        meshes = [
            trimesh.load(os.path.join(data_folder, mesh_file))
            for mesh_file in file_names
        ]

        # parsing into list
        vertices_all = []  # (N, v_i, 3)
        faces_all = []  # (N, f_i, 3)
        area_faces_all = []  # (N, f_i)
        for mesh in meshes:
            faces = torch.Tensor(mesh.faces)  # (f_i, 3)
            vertices = torch.Tensor(mesh.vertices)  # (v_i, 3)
            area_faces = torch.Tensor(mesh.area_faces)  # (f_i)

            faces_all.append(faces)
            vertices_all.append(vertices)
            area_faces_all.append(area_faces)

        is_pts_broken_all = self._get_broken_pcs_idxs(vertices_all, 0.0001)  # (N, v_i)
        is_broken_face_all = []  # (N, f_i)
        for faces, is_pts_broken, vertices in zip(faces_all, is_pts_broken_all, vertices_all):

            is_face_broken = []  # (f_i, )
            for vertex_idx in faces:
                vertex_idx = vertex_idx.long()
                is_vertex_broken = is_pts_broken[vertex_idx]  # (3, )
                is_face_broken.append(torch.all(is_vertex_broken))
            is_broken_face_all.append(torch.tensor(is_face_broken))

        # TODO: error나면 copy 때문인지 확인하기
        data = {
            "meshes": meshes,
            'is_broken_vertices': is_pts_broken_all,  # (N, v_i)
            'is_broken_face': is_broken_face_all,  # (N, f_i)
            'file_names': file_names,  # (N, )
            'dir_name': data_folder,
        }
        return data

    def sample(self, mesh, is_broken_vertices):

        faces = torch.Tensor(mesh.faces)  # (f_i, 3)
        area_faces = torch.Tensor(mesh.area_faces)  # (f_i)

        is_face_broken = torch.zeros(len(faces), dtype=torch.bool)
        for i, vertex_indice in enumerate(faces):
            vertex_indice = vertex_indice.long()
            is_vertex_broken = is_broken_vertices[vertex_indice]  # (3, )
            is_face_broken[i] = torch.all(is_vertex_broken).item()

        total_area_broken = torch.sum(area_faces[is_face_broken])
        
        n_broken_sample = int(self.sample_weight * total_area_broken)
        n_broken_sample = max(1, n_broken_sample)

        total_area_skin = torch.sum(area_faces[(is_face_broken.logical_not())])
        n_skin_sample = int(self.sample_weight * total_area_skin)

        if n_broken_sample + n_skin_sample > self.max_sample:
            # make the sum of two equal to max_sample
            n_broken_sample = int(n_broken_sample * (self.max_sample / (n_broken_sample + n_skin_sample)))
            n_skin_sample = int(n_skin_sample * (self.max_sample / (n_broken_sample + n_skin_sample)))

        broken_face_weight = area_faces * is_face_broken
        broken_face_weight = broken_face_weight / torch.sum(broken_face_weight)
        broken_sample, face_indices = trimesh.sample.sample_surface(
            mesh, n_broken_sample, broken_face_weight.numpy())
        broken_sample = torch.tensor(broken_sample)
        broken_normal = torch.tensor(mesh.face_normals[face_indices])

        skin_face_weight = area_faces * is_face_broken.logical_not()
        skin_face_weight = skin_face_weight / torch.sum(skin_face_weight)
        skin_sample, face_indices = trimesh.sample.sample_surface(
            mesh, n_skin_sample, skin_face_weight.numpy())
        skin_sample = torch.tensor(skin_sample)
        skin_normal = torch.tensor(mesh.face_normals[face_indices])

        sample = torch.cat([broken_sample, skin_sample], dim=0)
        sample *= self.scale
        broken_label = torch.cat([torch.ones(n_broken_sample), torch.zeros(n_skin_sample)], dim=0).bool()
        normal = torch.cat([broken_normal, skin_normal], dim=0)

        # shuffle
        perm = torch.randperm(len(sample))
        sample = sample[perm]
        broken_label = broken_label[perm]
        normal = normal[perm]

        assert len(skin_sample) == n_skin_sample
        assert len(broken_sample) == n_broken_sample
        data = {
            'sample': sample,  # (N, 3)
            'normal': normal,
            'broken_label': broken_label,  # (N, )
        }
        return data

    def len(self):
        return len(self.raw_file_names)

    def process(self):

        for data_folder, processed_path in tqdm(list(zip(self.raw_paths, self.processed_paths))):
            if os.path.exists(processed_path):
                new_dir = processed_path.split(".pt")[0]
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                new_path = os.path.join(new_dir, "mesh_info.pt")
                shutil.move(processed_path, new_path)
                
                continue

            data = self._parse_data(data_folder)

            directory = os.path.dirname(processed_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(data, processed_path)
            print("saved:", processed_path)

    @lru_cache(maxsize=100)
    def load_data(self, path):
        return torch.load(path)

    @lru_cache(maxsize=100)
    def get(self, idx):
        parsed_data = self.load_data(self.processed_paths[idx])
        meshes = parsed_data['meshes']
        is_pts_broken_all = parsed_data['is_broken_vertices']
        num_parts = len(meshes)

        pcs = []
        broken_labels = []
        normals = []
        for mesh, is_broken_vertices in zip(meshes, is_pts_broken_all):
            sample_data = self.sample(mesh, is_broken_vertices)
            pcs.append(sample_data['sample'])
            broken_labels.append(sample_data['broken_label'])
            normals.append(sample_data['normal'])

        quat, trans = [], []
        for i in range(num_parts):

            rot_mat = self._get_rotation_matrix()

            pc_origin = pcs[i]
            pc, gt_trans = self._recenter_pc(pc_origin)
            pc, gt_quat = self._rotate_pc(pc.float(), rot_mat)
            quat.append(gt_quat)
            trans.append(gt_trans)

            # check the rotation and translation are correct
            pc_recovered = jhutil.quat_trans_transform(gt_quat, gt_trans, pc.double())
            diff = torch.abs(pc_origin - pc_recovered)

            assert torch.all(diff < 1e-5), f"all pcs must be recovered within 1e-5: {diff}"
            assert diff.mean().item() < 1e-6, f"mean of diff must be less than 1e-6: {diff.mean().item()}"

            pcs[i] = pc

        return {
            'pcs': pcs,  # (N, p_i, 3)
            'quat': quat,
            'trans': trans,
            'broken_labels': broken_labels,
            'normals': normals,
            'dir_name': parsed_data['dir_name'],
            'file_names': parsed_data['file_names'],  # (N, )
        }

    def __len__(self):
        return len(self.raw_file_names)


def collate_fn(data_dicts):
    assert len(data_dicts) == 1
    data_dict = data_dicts[0]
    if "pcs" in data_dict:
        del data_dict["pcs"]

    return data_dict


def build_geometry_dataloader(cfg):
    train_set, val_set = build_geometry_dataset(cfg)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=1,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def build_geometry_dataset(cfg):

    data_dict = dict(
        data_dir=cfg.data.data_dir,
        data_fn=cfg.data.data_fn.format('train'),
        category=cfg.data.category,
        min_num_part=cfg.data.min_num_part,
        max_num_part=cfg.data.max_num_part,
        sample_weight=cfg.data.sample_weight,
        shuffle_parts=cfg.data.shuffle_parts,
        rot_range=cfg.data.rot_range,
        overfit=cfg.data.overfit,
        scale=cfg.data.scale,
    )
    train_set = GeometryPartDataset(**data_dict)

    data_dict['data_fn'] = cfg.data.data_fn.format('val')
    data_dict['shuffle_parts'] = False
    val_set = GeometryPartDataset(**data_dict)

    return train_set, val_set


def build_stage1_dataset(cfg, n_threshold=10000, overfit=-1):
    if overfit < 0:
        overfit == 1e10

    train_file_name = os.path.join(cfg.data.data_dir, "processed",
                                   f"{cfg.data.data_fn.format('train').split('.txt')[0]}.stage1_dataset_{overfit}.pt")
    val_file_name = os.path.join(cfg.data.data_dir, "processed",
                                 f"{cfg.data.data_fn.format('val').split('.txt')[0]}.stage1_dataset_{overfit}.pt")
    if os.path.exists(train_file_name) and os.path.exists(val_file_name):
        cached_train_set = torch.load(train_file_name)
        cached_val_set = torch.load(val_file_name)
    else:
        data_dict = dict(
            data_dir=cfg.data.data_dir,
            data_fn=cfg.data.data_fn.format('train'),
            category=cfg.data.category,
            min_num_part=cfg.data.min_num_part,
            max_num_part=cfg.data.max_num_part,
            sample_weight=cfg.data.train_sample_weight,
            shuffle_parts=cfg.data.shuffle_parts,
            rot_range=cfg.data.rot_range,
            overfit=cfg.data.overfit,
            scale=cfg.data.scale,
        )
        train_set = GeometryPartDataset(**data_dict)

        cached_train_set = []
        print("caching train data...")
        for i, data in tqdm(enumerate(train_set)):
            pcs = data["pcs"]
            normals = data["normals"]
            broken_labels = data["broken_labels"]
            for pcd, normal, broken_label in zip(pcs, normals, broken_labels):
                
                if len(pcd) < n_threshold:
                    continue
                cached_train_set.append({"pcd": pcd,
                                         "normal": normal,
                                         "broken_label": broken_label})
            if i >= overfit:
                break
        torch.save(cached_train_set, train_file_name)

        data_dict['data_fn'] = cfg.data.data_fn.format('val')
        data_dict['sample_weight'] = cfg.data.train_sample_weight
        data_dict['shuffle_parts'] = False
        val_set = GeometryPartDataset(**data_dict)

        cached_val_set = []
        print("caching val data...")
        for i, data in tqdm(enumerate(val_set)):
            pcs = data["pcs"]
            normals = data["normals"]
            broken_labels = data["broken_labels"]
            for pcd, normal, broken_label in zip(pcs, normals, broken_labels):
                if len(pcd) < n_threshold:
                    continue
                cached_val_set.append({"pcd": pcd,
                                      "normal": normal,
                                       "broken_label": broken_label})
            if i >= overfit:
                break
        torch.save(cached_val_set, val_file_name)

    return cached_train_set, cached_val_set
