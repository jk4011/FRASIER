

import os
from scipy.spatial.transform import Rotation as R

from torch.utils.data import DataLoader
import torch

from functools import lru_cache
import jhutil

from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Dataset
import sys
sys.path.append("../")
from part_assembly.data_util import create_mesh_info, sample_from_mesh_info, recenter_pc, rotate_pc

import argparse


class SampleDense(Dataset):
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
        sample_weight=150000,
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

        super().__init__(root=data_dir, transform=None, pre_transform=None)

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
        pcd_denses = [os.path.join(fn, "pcd_dense.pt") for fn in self.raw_file_names]
        return mesh_infos + pcd_denses

    def len(self):
        return len(self.raw_file_names)

    @property
    def mesh_info_paths(self):
        return self.processed_paths[:len(self.raw_file_names)]

    @property
    def pcd_dense_paths(self):
        return self.processed_paths[len(self.raw_file_names):]

    def process(self):
        assert len(self.mesh_info_paths) == len(self.pcd_dense_paths)

        print('1. prcessing mesh_info.pt...')
        create_mesh_info(self.raw_paths, self.mesh_info_paths)

        print('2. prcessing pcd_dense.pt...')
        for mesh_info_path, pcd_dense_path in tqdm(list(zip(self.mesh_info_paths, self.pcd_dense_paths))):
            assert mesh_info_path.endswith("mesh_info.pt")
            assert pcd_dense_path.endswith("pcd_dense.pt")

            if os.path.exists(pcd_dense_path):
                continue

            mesh_info = torch.load(mesh_info_path)
            pcd_dense = sample_from_mesh_info(mesh_info,
                                              sample_weight=self.sample_weight,
                                              max_n_sample=None,
                                              min_n_sample=None,
                                              omit_large_n=False,
                                              omit_small_n=False)

            trans, quats = [], []
            for pc in pcd_dense["sample"]:

                rot_mat = R.random().as_matrix()
                pc, gt_trans = recenter_pc(pc.float())
                pc, gt_quat = rotate_pc(pc.float(), rot_mat)
                trans.append(gt_trans)
                quats.append(gt_quat)

            pcd_dense["trans"] = trans
            pcd_dense["quats"] = quats

            directory = os.path.dirname(pcd_dense_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(pcd_dense, pcd_dense_path)

    def get(self, idx):
        idx = idx % len(self)
        data = torch.load(self.pcd_dense_paths[idx])
        return data


def build_sample_dense_dataloader(cfg):
    test_set = build_sample_dense_dataset(cfg)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    return test_loader


def build_sample_dense_dataset(cfg):

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
    )

    test_set = SampleDense(**data_dict)

    return test_set
