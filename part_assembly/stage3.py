
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '../src/geotransformer/experiments/lomatch/'))
from config import make_cfg
from model import create_model
from geotransformer.utils.data import registration_collate_fn_stack_mode
from jhutil import to_cuda
from scipy.spatial.transform import Rotation as R


model = None
cfg_ = make_cfg()


def geo_transformer(src, ref):
    """
    Args:
        src (N, 3)
        ref (M, 3)

    Returns:
        estimated_transform (torch.Tensor): (4, 4)
    """
    global model, cfg_

    assert isinstance(src, torch.Tensor) and isinstance(ref, torch.Tensor)
    src, ref = src.cpu(), ref.cpu()
    if len(src) <= 3 or len(ref) <= 3:
        return torch.eye(4)
    
    if model is None:
        model = create_model(cfg_)
        # TODO : breaking bad data에 train 된 모델 사용하기... 뭐야 이게 ㅋㅋㅋㅋㅋ 이러니 안 되지 ㅋㅋㅋㅋㅋ 아마 inference도 다 다시 해야 할 듯
        state_dict = torch.load(
            "/data/wlsgur4011/BreakingBad/GeoTransformer/weights/geotransformer-3dmatch.pth.tar", map_location=torch.device('cpu'))
        model_dict = state_dict['model']
        model.load_state_dict(model_dict, strict=False)
        model = model.cuda()
        model.eval()

    data = stage3_dataloader_format(src, ref)
    
    voxel_size = 0.025
    while data["lengths"][-1][1] >= 800:
        voxel_size += 0.005
        data = stage3_dataloader_format(src, ref, voxel_size=voxel_size)
        
    data = to_cuda(data)
    
    if data["lengths"][-1][0] <= 3 or data["lengths"][-1][1] <= 3:
        return torch.eye(4)
    
    with torch.no_grad():
        ret = model(data)
    return ret["estimated_transform"].cpu()


def collate_fn(
    data,
    cfg=None,
    neighbor_limits=np.array([38, 35, 35, 38]),
    precompute_data=True,
    voxel_size=None
):
    if cfg is None:
        global cfg_
        cfg = cfg_
    num_stages = cfg.backbone.num_stages
    if voxel_size is None:
        voxel_size = cfg.backbone.init_voxel_size
    search_radius = cfg.backbone.init_radius
    neighbor_limits,
    collate = partial(
        registration_collate_fn_stack_mode,
        num_stages=num_stages,
        voxel_size=voxel_size,
        search_radius=search_radius,
        neighbor_limits=neighbor_limits,
        precompute_data=precompute_data,
    )
    return collate([data])


def stage3_dataloader_format(src, ref, voxel_size=None):
    data = {
        "ref_points": src.contiguous(),
        "src_points": ref.contiguous(),
        "ref_feats": torch.ones(len(src), 1),
        "src_feats": torch.ones(len(ref), 1),
        "transform": torch.eye(4),
    }
    return collate_fn(data, voxel_size=voxel_size)



