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
from part_assembly.data_util import create_mesh_info, sample_from_mesh_info


class Sample20k(Dataset):
    """
    Point cloud dataset for training segmentation model.
    Everey number of point cloud is fixed to 20,000.

    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
        self,
        data_dir,
        data_fn,
        category='',
        sample_weight=750000,
        num_points=1000,
        min_num_part=2,
        max_num_part=1000,
        shuffle_parts=False,
        rot_range=-1,
        overfit=-1,
        scale=1,
        max_sample=20000,
        sample_whole=True,
    ):
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

        super().__init__(root=data_dir, transform=None, pre_transform=None)
        self.dataset = torch.load(self.pcd_20k_all_path)

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
        data_list = data_list
        return data_list

    @property
    def processed_file_names(self):

        mesh_infos = [os.path.join(fn, "mesh_info.pt") for fn in self.raw_file_names]
        pcd_20ks = [os.path.join(fn, "pcd_20k.pt") for fn in self.raw_file_names]
        if self.overfit > 0:
            return mesh_infos + pcd_20ks
        else:
            pcd_20k_all = [os.path.join(self.processed_dir, "pcd_20k_all.pt")]
            return mesh_infos + pcd_20ks + pcd_20k_all

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
        pc = pc @ rot_mat.T
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

    def len(self):
        return len(self.dataset)

    @property
    def mesh_info_paths(self):
        return self.processed_paths[:len(self.raw_file_names)]
    
    @property
    def pcd_20k_paths(self):
        return self.processed_paths[len(self.raw_file_names):-1]
    
    @property
    def pcd_20k_all_path(self):
        return self.processed_paths[-1]

    def process(self):
        assert len(self.mesh_info_paths) == len(self.pcd_20k_paths)
        
        print('1. prcessing mesh_info.pt...')
        create_mesh_info(self.raw_paths, self.mesh_info_paths)

        print('2. prcessing pcd_20k.pt...')
        for mesh_info_path, pcd_20k_path in tqdm(list(zip(self.mesh_info_paths, self.pcd_20k_paths))):
            assert mesh_info_path.endswith("mesh_info.pt")
            assert pcd_20k_path.endswith("pcd_20k.pt")
            
            if os.path.exists(pcd_20k_path):
                continue

            mesh_info = torch.load(mesh_info_path)
            pcd_20k = sample_from_mesh_info(mesh_info,
                                            sample_weight=self.sample_weight,
                                            max_n_sample=20000,
                                            min_n_sample=20000,
                                            omit_large_n=False,
                                            omit_small_n=True)
            directory = os.path.dirname(pcd_20k_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            torch.save(pcd_20k, pcd_20k_path)
        
        if self.overfit < 0:
            print('3. prcessing pcd_20k_all.pt...')
            pcd_20k_all = {"sample": [], "normal": [], "broken_label": []}
            for pcd_20k_path in tqdm(self.pcd_20k_paths):
                pcd_20k = torch.load(pcd_20k_path)
                pcd_20k_all["sample"] += pcd_20k["sample"]
                pcd_20k_all["normal"] += pcd_20k["normal"]
                pcd_20k_all["broken_label"] += pcd_20k["broken_label"]
            
            pcd_20k_all["sample"] = torch.stack(pcd_20k["sample"])
            pcd_20k_all["normal"] = torch.stack(pcd_20k["normal"])
            pcd_20k_all["broken_label"] = torch.stack(pcd_20k["broken_label"])
            torch.save(pcd_20k_all, self.pcd_20k_all_path)


    @lru_cache(maxsize=100)
    def get(self, idx):
        idx = idx % len(self.dataset)
        try:
            pc = self.dataset["sample"][idx]
            normal = self.dataset["normal"][idx]
            broken_label = self.dataset["broken_label"][idx]
        except:
            data = torch.load(self.pcd_20k_paths[idx][0])
            pc = data["sample"][0]
            normal = data["normal"][0]
            broken_label = data["broken_label"][0]
        
        
        rot_mat = self._get_rotation_matrix()
        pc, gt_trans = self._recenter_pc(pc.float())
        pc, gt_quat = self._rotate_pc(pc.float(), rot_mat)
        normal, _ = self._rotate_pc(normal.float(), rot_mat)

        return {
            'pcs': pc,  # (N, p_i, 3)
            'quat': gt_quat,
            'trans': gt_trans,
            'normals': normal,
            'broken_labels': broken_label,
        }

    def __len__(self):
        return len(self.raw_file_names)


def build_sample_20k_dataloader(cfg):
    train_set, val_set = build_sample_20k_dataset(cfg)

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


def build_sample_20k_dataset(cfg):

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
    train_set = Sample20k(**data_dict)

    data_dict['data_fn'] = cfg.data.data_fn.format('val')
    data_dict['shuffle_parts'] = False
    val_set = Sample20k(**data_dict)

    return train_set, val_set