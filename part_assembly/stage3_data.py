from part_assembly.stage3 import original_to_stage3
import torch
from torch.utils.data import Dataset


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
