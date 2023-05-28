
from multi_part_assembly.datasets.geometry_data import build_geometry_dataset, build_geometry_dataloader, save_geometry_dataset
from multi_part_assembly.datasets.geometry_data import GeometryPartDataset
import jhutil
import torch
from torch.utils.data import Dataset, DataLoader

from pytorch3d.transforms import matrix_to_quaternion, matrix_to_axis_angle, \
    quaternion_to_matrix, quaternion_to_axis_angle, \
    axis_angle_to_quaternion, axis_angle_to_matrix

import argparse


class Stage3PairDataset(Dataset):
    def __init__(self,
                 datapath,
                 exclusive_pair=False,
                 min_num_part=2,
                 max_num_part=100,
                 overlap_threshold=0.01):
        raw_dataset = torch.load(datapath)

        self.dataset = []
        for data in raw_dataset:
            if min_num_part <= len(data['broken_pcs']) <= max_num_part:
                self.dataset.append(data)

        self.adjacent_all = []
        if exclusive_pair:
            for i, data in enumerate(self.dataset):
                leng = len(data['broken_pcs'])

                pairs = []
                for j in range(leng):
                    for k in range(j):
                        pairs.append([j, k])

                # overwrite into exclusive pair
                self.dataset[i]["adjacent_pair"] = pairs
                self.adjacent_all.append(pairs)
        else:
            for i, data in enumerate(self.dataset):
                n = len(data['broken_pcs'])
                overlap_ratios = data["overlap_ratios"]
                assert overlap_ratios.shape == (n, n)

                is_appended = [False] * n
                adjacent_pair = []
                for i in range(n):
                    for j in range(n):
                        if i >= j:
                            continue
                        overlap_score = overlap_ratios[i, j] * overlap_ratios[j, i]
                        if overlap_score > overlap_threshold:
                            is_appended[i] = True
                            is_appended[j] = True
                            adjacent_pair.append([i, j])
                for i in range(n):
                    if not is_appended[i]:
                        pair = [i, torch.argmax(overlap_ratios[i])]
                        sorted(pair)
                        adjacent_pair.append(pair)
                self.adjacent_all.append(adjacent_pair)

    def __len__(self):
        if not hasattr(self, "leng"):
            self.leng = 0
            for adjacent_pair in self.adjacent_all:
                self.leng += len(adjacent_pair)

        return self.leng

    def _get_index(self, index):
        data_idx = 0
        for adjacent_list in self.adjacent_all:
            if index >= len(adjacent_list):
                index -= len(adjacent_list)
                data_idx += 1
            else:
                src_idx, ref_idx = adjacent_list[index]
                break

        return data_idx, src_idx, ref_idx

    def __getitem__(self, index):

        data_idx, src_idx, ref_idx = self._get_index(index)
        data = self.dataset[data_idx]

        return original_to_stage3(data, src_idx, ref_idx)


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
