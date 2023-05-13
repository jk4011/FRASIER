
from multi_part_assembly.datasets.geometry_data import build_geometry_dataset, build_geometry_dataloader, save_geometry_dataset
from multi_part_assembly.datasets.geometry_data import GeometryPartDataset
import jhutil
import torch
from torch.utils.data import Dataset, DataLoader

from pytorch3d.transforms import matrix_to_quaternion, matrix_to_axis_angle, \
    quaternion_to_matrix, quaternion_to_axis_angle, \
    axis_angle_to_quaternion, axis_angle_to_matrix

import argparse


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overfit', type=int, default=None)
    parser.add_argument('--min_numpart', type=int, default=None)

    # parse
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    cfg = jhutil.load_yaml("yamls/data_example.yaml")
    if args.overfit is not None:
        cfg.data.overfit = args.overfit
    if args.min_numpart is not None:
        cfg.data.min_numpart = args.min_numpart

    save_geometry_dataset(cfg)
