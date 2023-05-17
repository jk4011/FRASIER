
from multi_part_assembly.datasets.geometry_data import build_geometry_dataset, build_geometry_dataloader, save_geometry_dataset
from multi_part_assembly.datasets.geometry_data import GeometryPartDataset
import jhutil
import torch
from torch.utils.data import Dataset, DataLoader

from pytorch3d.transforms import matrix_to_quaternion, matrix_to_axis_angle, \
    quaternion_to_matrix, quaternion_to_axis_angle, \
    axis_angle_to_quaternion, axis_angle_to_matrix

import argparse


class PairBreakingBadDataset(Dataset):
    def __init__(self, datapath, exclusive_pair=False, min_num_part=2, max_num_part=100):
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
                self.adjacent_all.append(data["adjacent_pair"])
            

    def __len__(self):
        if not hasattr(self, "leng"):
            self.leng = 0
            for adjacent_pair in self.adjacent_all:
                self.leng += len(adjacent_pair)

        return self.leng

    def transform_matrix_from_quaternion_translation(self, quaternion, translation):
        rotation_matrix = quaternion_to_matrix(torch.tensor(quaternion))
        translation = translation.reshape(3, 1)

        last_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        transform_matrix = torch.cat((rotation_matrix, translation), dim=1)
        transform_matrix = torch.cat((transform_matrix, last_row), dim=0)

        return transform_matrix

    def relative_transform_matrix(self, src_quat, ref_quat, src_trans, ref_trans):
        src_transform_matrix = self.transform_matrix_from_quaternion_translation(
            src_quat, src_trans)
        ref_transform_matrix = self.transform_matrix_from_quaternion_translation(
            ref_quat, ref_trans)

        # Calculate the inverse of the ref_transform_matrix
        ref_transform_matrix_inv = torch.inverse(ref_transform_matrix)

        # Calculate the relative transform matrix
        relative_matrix = torch.matmul(
            ref_transform_matrix_inv, src_transform_matrix)
        relative_matrix = relative_matrix.to(torch.float32).contiguous()

        return relative_matrix

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

        src_points = data['broken_pcs'][src_idx]
        ref_points = data['broken_pcs'][ref_idx]
        
        # if len(src_points) > 1e4 and len(ref_points) > 1e4:
        #     src_points = src_points[::2]
        #     ref_points = ref_points[::2]
        #     import jhutil; jhutil.jhprint(1111, )

        src_quat = data['quat'][src_idx]
        ref_quat = data['quat'][ref_idx]
        src_trans = data['trans'][src_idx]
        ref_trans = data['trans'][ref_idx]
        file_names = data["file_names"]

        transform = self.relative_transform_matrix(
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
            # "overlap": -1,
            "ref_feats": torch.ones(len(src_points), 1),
            "src_feats": torch.ones(len(ref_points), 1),
            "transform": transform,
        }
        return out


if __name__ == "__main__":

    # datafolder = "/data/wlsgur4011/DataCollection/BreakingBad/data_split/"
    # artifact_train = f"{datafolder}artifact.train.pth"
    # artifact_val = f"{datafolder}artifact.val.pth"
    # everyday_train = f"{datafolder}everyday.train.pth"
    # everyday_val = f"{datafolder}everyday.val.pth"

    # dataset = PairBreakingBadDataset(artifact_train)

    # import jhutil;jhutil.jhprint(1111, len(dataset))

    if True:

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

        save_geometry_dataset(cfg)
