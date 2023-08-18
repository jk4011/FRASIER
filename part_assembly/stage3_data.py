from part_assembly.stage3 import original_to_stage3
import torch
from torch.utils.data import Dataset
from functools import lru_cache
import os
from part_assembly.data_util import create_mesh_info, sample_from_mesh_info, recenter_pc, rotate_pc
from tqdm import tqdm



class Stage3PairDataset(Dataset):
    def __init__(self,
                 datapath,
                 exclusive_pair=False,
                 min_num_part=2,
                 max_num_part=100,
                 overlap_threshold=0.01):
        raw_dataset = torch.load(datapath)


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
        pcd_broken_noisy = [os.path.join(fn, "pcd_broken_noisy.pt") for fn in self.raw_file_names]
        return mesh_infos + pcd_broken_noisy

    @property
    def mesh_info_paths(self):
        return self.processed_paths[:len(self.raw_file_names)]

    @property
    def pcd_broken_noisy_paths(self):
        return self.processed_paths[len(self.raw_file_names): 2 * len(self.raw_file_names)]

    @property
    def pcd_pair_dir_path(self):
        return os.path.join(self.processed_dir, self.dataset_name, f"pcd_20ks_{self.mode}")
    
    def process(self):
        assert len(self.mesh_info_paths) == len(self.pcd_broken_noisy_paths)
        
        print('1. prcessing mesh_info.pt...')
        create_mesh_info(self.raw_paths, self.mesh_info_paths)
   
        print('2. prcessing pcd_broken_noisy.pt...')
        if not os.path.exists(self.pcd_pair_dir_path):
            os.mkdir(self.pcd_pair_dir_path)  
        
        i = 0
        for mesh_info_path, pcd_broken_noisy_path in tqdm(list(zip(self.mesh_info_paths, self.pcd_broken_noisy_paths))):
            assert mesh_info_path.endswith("mesh_info.pt")
            assert pcd_broken_noisy_path.endswith("pcd_broken_noisy.pt")
            assert os.path.dirname(mesh_info_path) == os.path.dirname(pcd_broken_noisy_path)
            
            if os.path.exists(pcd_broken_noisy_path):
                continue
            
            mesh_info = torch.load(mesh_info_path)
            pcd_20k = sample_broken_noisy(mesh_info)
            
            
            
    
    
    def __getitem__(self, index):

        data_idx, src_idx, ref_idx = self._get_index(index)
        data = self.dataset[data_idx]

        return original_to_stage3(data, src_idx, ref_idx)
