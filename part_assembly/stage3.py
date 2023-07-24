
from multi_part_assembly.datasets.geometry_data import build_geometry_dataset, build_geometry_dataloader
from multi_part_assembly.datasets.geometry_data import GeometryPartDataset
import jhutil
import torch
from torch.utils.data import DataLoader

from pytorch3d.transforms import matrix_to_quaternion, matrix_to_axis_angle, \
    quaternion_to_matrix, quaternion_to_axis_angle, \
    axis_angle_to_quaternion, axis_angle_to_matrix

import argparse
import numpy as np
import sys
from functools import partial
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src/geotransformer/experiments/lomatch/'))
from config import make_cfg
from model import create_model
from geotransformer.utils.data import registration_collate_fn_stack_mode
from jhutil import to_cuda


model = None
cfg_ = make_cfg()


def geo_transformer(src, ref):
    global model, cfg_
    
    assert isinstance(src, torch.Tensor) and isinstance(ref, torch.Tensor)
    src, ref = src.cpu(), ref.cpu()
    
    if model is None:
        model = create_model(cfg_)
        state_dict = torch.load(
            "/data/wlsgur4011/BreakingBad/GeoTransformer/weights/geotransformer-3dmatch.pth.tar", map_location=torch.device('cpu'))
        model_dict = state_dict['model']
        model.load_state_dict(model_dict, strict=False)
        model = model.cuda()
        model.eval()
        
    data = stage3_dataloader_format(src, ref)
    data = to_cuda(data)

    ret = model(data)
    return ret["estimated_transform"].cpu()


def collate_fn(
    data,
    cfg=None,
    neighbor_limits=np.array([38, 35, 35, 38]),
    precompute_data=True,
):
    if cfg is None:
        global cfg_
        cfg = cfg_
    num_stages = cfg.backbone.num_stages
    voxel_size = cfg.backbone.init_voxel_size
    search_radius = cfg.backbone.init_radius
    neighbor_limits,
    collate_fn = partial(
        registration_collate_fn_stack_mode,
        num_stages=num_stages,
        voxel_size=voxel_size,
        search_radius=search_radius,
        neighbor_limits=neighbor_limits,
        precompute_data=precompute_data,
    )
    return collate_fn([data])


def stage3_dataloader_format(src, ref):
    data = {
        "ref_points": src.contiguous(),
        "src_points": ref.contiguous(),
        "ref_feats": torch.ones(len(src), 1),
        "src_feats": torch.ones(len(ref), 1),
        "transform": torch.eye(4),
    }
    return collate_fn(data)


def original_to_stage3(data, src_idx, ref_idx):
    src_points = data['broken_pcs'][src_idx]
    ref_points = data['broken_pcs'][ref_idx]

    src_quat = data['quat'][src_idx]
    ref_quat = data['quat'][ref_idx]
    src_trans = data['trans'][src_idx]
    ref_trans = data['trans'][ref_idx]
    file_names = data["file_names"]

    overlap_ratios = data["overlap_ratios"]
    overlap_score = overlap_ratios[src_idx, ref_idx] * overlap_ratios[ref_idx, src_idx]

    transform = relative_transform_matrix(
        src_quat, ref_quat, src_trans, ref_trans)

    out = {
        "scene_name": data["dir_name"].split("data_split/")[-1],
        "dir_name": data["dir_name"],
        "src_file_name": file_names[src_idx],
        "ref_file_name": file_names[ref_idx],
        "src_frame": src_idx,
        "ref_frame": ref_idx,
        "ref_points": ref_points.contiguous(),
        "src_points": src_points.contiguous(),
        "ref_feats": torch.ones(len(src_points), 1),
        "src_feats": torch.ones(len(ref_points), 1),
        "transform": transform,
        "overlap_score": overlap_score,
    }
    return out


def transform_matrix_from_quaternion_translation(quaternion, translation):
    rotation_matrix = quaternion_to_matrix(torch.tensor(quaternion))
    translation = translation.reshape(3, 1)

    last_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    transform_matrix = torch.cat((rotation_matrix, translation), dim=1)
    transform_matrix = torch.cat((transform_matrix, last_row), dim=0)

    return transform_matrix


def relative_transform_matrix(src_quat, ref_quat, src_trans, ref_trans):
    src_transform_matrix = transform_matrix_from_quaternion_translation(
        src_quat, src_trans)
    ref_transform_matrix = transform_matrix_from_quaternion_translation(
        ref_quat, ref_trans)

    # Calculate the inverse of the ref_transform_matrix
    ref_transform_matrix_inv = torch.inverse(ref_transform_matrix)

    # Calculate the relative transform matrix
    relative_matrix = torch.matmul(
        ref_transform_matrix_inv, src_transform_matrix)
    relative_matrix = relative_matrix.to(torch.float32).contiguous()

    return relative_matrix
