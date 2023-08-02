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
from part_assembly.data_util import create_mesh_info, sample_from_mesh_info, recenter_pc, rotate_pc

import argparse


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
        sample_whole=True,  # for stage1+stage2
        dataset_name='artifact',
        mode='train',  # train, val
        is_fracture_single=True,
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
        self.dataset_name = dataset_name
        self.mode = mode
        self.is_fracture_single = is_fracture_single

        super().__init__(root=data_dir, transform=None, pre_transform=None)
        self.single_files = os.listdir(self.pcd_20ks_dir_path)
        self.single_files = [os.path.join(self.pcd_20ks_dir_path, fn) for fn in self.single_files]

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
        return data_list

    @property
    def processed_file_names(self):
        mesh_infos = [os.path.join(fn, "mesh_info.pt") for fn in self.raw_file_names]
        pcd_20ks = [os.path.join(fn, "pcd_20k.pt") for fn in self.raw_file_names]
        return mesh_infos + pcd_20ks

    def len(self):
        if self.is_fracture_single:
            return len(self.single_files)
        else:
            return len(self.raw_file_names)

    @property
    def mesh_info_paths(self):
        return self.processed_paths[:len(self.raw_file_names)]

    @property
    def pcd_20k_paths(self):
        return self.processed_paths[len(self.raw_file_names): 2 * len(self.raw_file_names)]

    @property
    def pcd_20ks_dir_path(self):
        return os.path.join(self.processed_dir, self.dataset_name, f"pcd_20ks_{self.mode}")

    def process(self):
        assert len(self.mesh_info_paths) == len(self.pcd_20k_paths)

        print('1. prcessing mesh_info.pt...')
        create_mesh_info(self.raw_paths, self.mesh_info_paths)

        print('2. prcessing pcd_20k.pt...')

        if not os.path.exists(self.pcd_20ks_dir_path):
            os.mkdir(self.pcd_20ks_dir_path)

        i = 0
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

            # save pcd_20k.pt as pcd_20ks/{i}.pt
            samples = pcd_20k["sample"]
            broken_labels = pcd_20k["broken_label"]
            for sample, broken_label in zip(samples, broken_labels):
                single_data = {"sample": sample, "broken_label": broken_label}
                single_file = os.path.join(self.pcd_20ks_dir_path, f"{i}.pt")
                torch.save(single_data, single_file)
                i += 1

    @lru_cache(maxsize=100)
    def load_fracture_single(self, idx):
        single_data = torch.load(self.single_files[idx])
        sample = single_data["sample"]
        broken_label = single_data["broken_label"]

        rot_mat = R.random().as_matrix()
        sample, gt_trans = recenter_pc(sample.float())
        sample, gt_quat = rotate_pc(sample.float(), rot_mat)

        return {
            'pcd': sample.numpy(),  # (N, p_i, 3)
            'quat': gt_quat,
            'trans': gt_trans,
            'broken_label': broken_label.numpy(),
        }
    
    def load_fracture_set(self, idx):
        data = torch.load(self.pcd_20k_paths[idx])
        
        n = len(data["sample"])
        
        pcs, quats, trans, broken_labels = [], [], [], []
        for i in range(n):
            rot_mat = R.random().as_matrix()
            pcd = data["sample"][i]
            pcd, gt_trans = recenter_pc(pcd.float())
            pcd, gt_quat = rotate_pc(pcd.float(), rot_mat)
            
            pcs.append(pcd.numpy())
            quats.append(gt_quat)
            trans.append(gt_trans)
            broken_labels.append(data["broken_label"][i].numpy())
        
        return {
            'pcs': pcs,  # (N, p_i, 3)
            'quats': gt_quat,
            'trans': gt_trans,
            'broken_labels': broken_labels,
        }
    
    def get(self, idx):
        if self.is_fracture_single:
            return self.load_fracture_single(idx)
        else:
            return self.load_fracture_set(idx)


def build_sample_20k_dataloader(cfg):
    train_set = build_sample_20k_train_dataset(cfg)
    val_set = build_sample_20k_val_dataset(cfg)

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


def build_sample_20k_test_loder(cfg):
    test_set = build_sample_20k_val_dataset(cfg)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    return test_loader


def build_sample_20k_train_dataset(cfg):
    print("building sample_20k train dataset...")
    print(f"overfit: {cfg.overfit}")

    data_dict = dict(
        data_dir=cfg.data_dir,
        data_fn=cfg.data_fn.format('train'),
        category=cfg.category,
        min_num_part=cfg.min_num_part,
        max_num_part=cfg.max_num_part,
        sample_weight=cfg.sample_weight,
        shuffle_parts=cfg.shuffle_parts,
        rot_range=cfg.rot_range,
        overfit=cfg.overfit,
        scale=cfg.scale,
        dataset_name=cfg.data_fn.split('.')[0],
        mode='train',
    )
    return Sample20k(**data_dict)


def build_sample_20k_val_dataset(cfg):

    print("building sample_20k val dataset...")
    print(f"overfit: {cfg.overfit}")

    data_dict = dict(
        data_dir=cfg.data_dir,
        data_fn=cfg.data_fn.format('val'),
        category=cfg.category,
        min_num_part=cfg.min_num_part,
        max_num_part=cfg.max_num_part,
        sample_weight=cfg.sample_weight,
        shuffle_parts=False,
        rot_range=cfg.rot_range,
        overfit=cfg.overfit,
        scale=cfg.scale,
        dataset_name=cfg.data_fn.split('.')[0],
        mode='val',
    )
    return Sample20k(**data_dict)
