from multi_part_assembly.datasets.geometry_data import build_geometry_dataset, build_geometry_dataloader
from multi_part_assembly.datasets.geometry_data import GeometryPartDataset as BreakingBadDataset
import jhutil
import torch
from torch.utils.data import Dataset, DataLoader
import torch

class PairBreakingBadDataset(Dataset):
    def __init__(self, dataset: BreakingBadDataset):
        leng_list = []
        for data in dataset:
            leng_list.append(len(data['breaking_pcs']))
        
        num_pair_list = [leng * (leng - 1) / 2 for leng in leng_list]
        
        self.dataset = dataset
        self.leng_list = leng_list
        self.num_pair_list = num_pair_list
    
    
    def __len__(self):
        return sum(self.num_pair_list)
    
    def _get_part_idx(self, object_idx, idx):
        leng = self.leng_list[object_idx]
        ref_idx = idx
        src_idx = 0
        for i in range(leng, 0, -1):
            if ref_idx < i:
                break
            else:
                src_idx += 1
                ref_idx -= i
            
        return src_idx, ref_idx
    
    def __getitem__(self, index):
        index /= len(self)
        
        object_idx = 0
        for n in self.num_pair_list:
            if index < n:
                break
            else:
                object_idx += 1
                index -= n
        object = self.dataset[object_idx]
        src_idx, ref_idx = self._get_part_idx(object_idx, index)
        src = object['breaking_pcs'][src_idx]
        ref =  object['breaking_pcs'][ref_idx]
        out = {
            "src_points" : src,
            "ref_points" : ref,
            "object_idx" : object_idx,
            "src_idx" : src_idx,
            "ref_idx" : ref_idx,
        }
        return out
        
        


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    cfg = jhutil.load_yaml("yamls/data_example.yaml")
    train_data, val_data = build_geometry_dataset(cfg)
    train_data = PairBreakingBadDataset(train_data)
    import jhutil;jhutil.jhprint(1111, )
