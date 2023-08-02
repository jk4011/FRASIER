
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port

from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from jhutil import to_cuda
# from multi_part_assembly.datasets.geometry_data import build_geometry_dataloader
from jhutil import load_yaml
import torch
import numpy as np


def subsample(data, voxel_size=0.04, presample=True, split='test', voxel_max=7500, variable=False, shuffle=False):

    sample = data['pos'].numpy()
    normal = data['x'].numpy()
    broken_label = data['y'].numpy().reshape(-1, 1)

    cdata = np.hstack((sample, normal, broken_label))  # (N, 7)
    cdata[:, :3] -= np.min(cdata[:, :3], 0)
    if voxel_size:
        coord, feat, label = cdata[:, 0:3], cdata[:, 3:6], cdata[:, 6:7]
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        cdata = np.hstack((coord, feat, label))

    if presample:
        coord, feat, label = np.split(cdata, [3, 6], axis=1)
    else:
        cdata[:, :3] -= np.min(cdata[:, :3], 0)
        coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
        coord, feat, label = crop_pc(
            coord, feat, label, split, voxel_size, voxel_max,
            downsample=not presample, variable=variable, shuffle=shuffle)

    label = label.squeeze(-1).astype(np.long)
    cls = np.zeros(1).astype(np.int64)
    data = {'pos': torch.Tensor(coord)[None, :],
            'x': torch.Tensor(feat)[None, :],
            'y': torch.Tensor(label)[None, :],
            'cls': torch.Tensor(cls)[None, :]}
    return data


def closest_point(src, dst):
    dst = dst.reshape(1, dst.shape[0], 3)  # (1, M, 3)
    
    chunk_size = 10000
    src_chunks = torch.chunk(src, src.shape[0] // chunk_size + 1, 0)

    min_dist_indices = []

    for src_chunk in src_chunks:
        src_chunk = src_chunk.reshape(src_chunk.shape[0], 1, 3)  # (N, 1, 3)
        dist = src_chunk - dst  # (N, M, 3)
        min_dist_idx_chunk = torch.argmin(torch.sum(dist ** 2, dim=2), dim=1)
        min_dist_indices.append(min_dist_idx_chunk)

    min_dist_idx = torch.cat(min_dist_indices, 0)

    return min_dist_idx



def broken_surface_segmentation(model, data, is_subsample=False, feature_keys='x', all_broken_threshold=5000):

    model.eval()
    broken_pcd_list = []
    for i in range(len(data['sample'])):
        if len(data['sample'][i][0]) < all_broken_threshold:
            broken_pcd_list.append(data['sample'][i].squeeze().cpu())
            continue

        data2 = {
            "pos": data['sample'][i].float(),
            "x": data['sample'][i].float(),
        }

        if is_subsample:
            data2 = subsample(data2)

        data2 = to_cuda(data2)

        data2['x'] = get_features_by_keys(data2, feature_keys)
        with torch.no_grad():
            res = model(data2).argmax(dim=1).squeeze()

        pcd = data2['pos'][0]
        broken_pcd_list.append(pcd[res == 1].cpu())

    return broken_pcd_list


def load_pointnext(cfg_path="/data/wlsgur4011/part_assembly/src/pointnext/cfgs/part_assembly/pointnext-l.yaml",
                   model_path="/data/wlsgur4011/part_assembly/src/pointnext/log/part_assembly/part_assembly-train-pointnext-l-ngpus4-seed9811-20230731-192146-4vMpY777YhAjPWULPKuSTZ/checkpoint/part_assembly-train-pointnext-l-ngpus4-seed9811-20230731-192146-4vMpY777YhAjPWULPKuSTZ_ckpt_best.pth"):
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)

    pointnext = build_model_from_cfg(cfg.model).cuda()
    best_epoch, best_val = load_checkpoint(pointnext, pretrained_path=model_path)
    pointnext.eval()

    return pointnext


def stage1_preprocess(overfit=5,
                      train_data_path="/data/wlsgur4011/DataCollection/BreakingBad/data_split/preprocessed_artifact.val.pth",
                      val_data_path="/data/wlsgur4011/DataCollection/BreakingBad/data_split/preprocessed_artifact.train.pth"):
    pointnext = load_pointnext().cuda()
    data_cfg = load_yaml("/data/wlsgur4011/part_assembly/yamls/data_example.yaml")
    train_loader, val_loader = build_geometry_dataloader(data_cfg, use_saved=True)

    broken_pcd_list = []
    for data in train_loader:
        if len(broken_pcd_list) >= overfit:
            break
        broken_pcd = broken_surface_segmentation(pointnext, data)
        broken_pcd_list.append(broken_pcd)
    torch.save(broken_pcd_list, train_data_path)
    del broken_pcd_list

    broken_pcd_list = []
    for data in val_loader:
        if len(broken_pcd_list) >= overfit:
            break
        broken_pcd = broken_surface_segmentation(pointnext, data)
        broken_pcd_list.append(broken_pcd)
    torch.save(broken_pcd_list, val_data_path)
    del broken_pcd_list


if __name__ == "__main__":
    stage1_preprocess()
