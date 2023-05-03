
from multi_part_assembly.datasets.geometry_data import build_geometry_dataset, build_geometry_dataloader
from multi_part_assembly.datasets.geometry_data import GeometryPartDataset as BreakingBadDataset
import jhutil
import torch
from torch.utils.data import Dataset, DataLoader
import torch


from pytorch3d.transforms import matrix_to_quaternion, matrix_to_axis_angle, \
    quaternion_to_matrix, quaternion_to_axis_angle, \
    axis_angle_to_quaternion, axis_angle_to_matrix

class PairBreakingBadDataset(Dataset):
    def __init__(self, dataset: BreakingBadDataset):
        leng_list = []
        for data in dataset:
            leng_list.append(len(data['broken_pcs']))
        
        num_pair_list = [leng * (leng - 1) / 2 for leng in leng_list]
        
        self.dataset = dataset
        self.leng_list = leng_list
        self.num_pair_list = num_pair_list
    
    
    def __len__(self):
        return int(sum(self.num_pair_list))
    
    def _get_part_idx(self, object_idx, idx):
        leng = self.leng_list[object_idx]
        count = 0
        for i in range(leng):
            for j in range(i + 1, leng):
                if count == idx:
                    return i, j
                count += 1
            
    
    def _get_part_indices(n, index):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if count == index:
                    return i, j
                count += 1
    
    def transform_matrix_from_quaternion_translation(self, quaternion, translation):
        rotation_matrix = quaternion_to_matrix(torch.tensor(quaternion))
        translation = translation.reshape(3, 1)

        last_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        transform_matrix = torch.cat((rotation_matrix, translation), dim=1)
        transform_matrix = torch.cat((transform_matrix, last_row), dim=0)

        return transform_matrix

    def relative_transform_matrix(self, src_quat, ref_quat, src_trans, ref_trans):
        src_transform_matrix = self.transform_matrix_from_quaternion_translation(src_quat, src_trans)
        ref_transform_matrix = self.transform_matrix_from_quaternion_translation(ref_quat, ref_trans)

        # Calculate the inverse of the src_transform_matrix
        src_transform_matrix_inv = torch.inverse(src_transform_matrix)

        # Calculate the relative transform matrix
        relative_matrix = torch.matmul(src_transform_matrix_inv, ref_transform_matrix)

        return relative_matrix

    
    def __getitem__(self, index):
        # index = int(index) % int(len(self))
        
        object_idx = 0
        for n in self.num_pair_list:
            if index < n:
                break
            else:
                object_idx += 1
                index -= n
        object = self.dataset[object_idx]
        src_idx, ref_idx = self._get_part_idx(object_idx, index)
        src_points = object['broken_pcs'][src_idx]
        ref_points = object['broken_pcs'][ref_idx]
        src_quat = object['quat'][src_idx]
        ref_quat = object['quat'][ref_idx]
        src_trans = object['trans'][src_idx]
        ref_trans = object['trans'][ref_idx]
        
        transform = self.relative_transform_matrix(src_quat, ref_quat, src_trans, ref_trans)
        
        out = {
            "scene_name" : object["data_path"],
            "ref_frame" : ref_idx,
            "src_frame" : src_idx,
            "ref_points" : ref_points,
            "src_points" : src_points,
            # "overlap": -1,
            "ref_feats": torch.ones(len(src_points), 1),# "array[18977, 1] f32 74Kb x∈[1.000, 1.000] μ=1.000 σ=0.",
            "src_feats": torch.ones(len(ref_points), 1),# "array[19082, 1] f32 75Kb x∈[1.000, 1.000] μ=1.000 σ=0.",
            "transform" : transform,
        }
        return out
        
        


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    cfg = jhutil.load_yaml("yamls/data_example.yaml")
    train_data, val_data = build_geometry_dataset(cfg)
    train_data = PairBreakingBadDataset(train_data)
    for data in train_data:
        import jhutil;jhutil.jhprint(4444, data)
        break
    
