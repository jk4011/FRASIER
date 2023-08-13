import torch
from jhutil import knn
from typing import Union
import random
from queue import PriorityQueue
import heapq
from jhutil import matrix_transform
from jhutil import load_yaml
from jhutil import to_cuda
from copy import deepcopy
from .stage3 import geo_transformer
from jhutil.log import create_logger
from jhutil import open3d_icp
import numpy as np
import torch.autograd.profiler as profiler

from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port

from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
import time

from part_assembly.data_util import pcd_subsample

logger = create_logger()


class Fracture:
    def __init__(self, pcd, merge_state:frozenset, n_removed:int, n_last_removed:int, T_dic=None):
        assert isinstance(merge_state, frozenset)
        self.pcd = pcd
        self.merge_state = merge_state
        self.n_removed = n_removed
        self.n_last_removed = n_last_removed

        if T_dic is None:
            T_dic = {merge_state: torch.eye(4)}
        self.T_dic = T_dic
        # TODO: bundle adjustment를 위해 original pcd도 저장해 두기

    @property
    def num_pcd(self):
        return len(self.merge_state)

    def merge(self, other, T):
        """merge two fracs

        Args:
            other (Fracture): other frac
            T (torch.Tensor): homographic transformation matrix

        Returns:
            new_frac: merged frac
        """
        
        pcd_transformed = matrix_transform(T, other.pcd)
        new_pcd, n_last_removed = pointcloud_xor(self.pcd, pcd_transformed)
        n_removed = self.n_removed + other.n_removed + n_last_removed
        
        merge_state = self.merge_state.union(other.merge_state)

        other_T_dic = {merge_state: T @ prev_T for merge_state, prev_T in other.T_dic.items()}
        T_dict = {**self.T_dic, **other_T_dic}

        return Fracture(new_pcd, merge_state, n_removed, n_last_removed, T_dict)

    def __eq__(self, other):
        return self.merge_state == other.merge_state

    def __gt__(self, other):
        return str(self.merge_state) > str(other.merge_state)

    def __lt__(self, other):
        return str(self.merge_state) < str(other.merge_state)

    def __hash__(self) -> int:
        return str(self.merge_state).__hash__()

    def __str__(self):
        return str(self.merge_state)


class FractureSet:
    def __init__(self, pcd_list, use_icp=True):
        self.n_origin_pcd = sum([pcd.shape[0] for pcd in pcd_list])
        self.fracs : list[Fracture] = [Fracture(pcd, merge_state=frozenset([i]), n_removed=0, n_last_removed=0) for i, pcd in enumerate(pcd_list)]
        self.k = 3
        self.use_icp = use_icp

    @property
    def n_removed(self):
        return sum([frac.n_removed for frac in self.fracs])

    @property
    def num_pcd(self):
        return sum(frac.num_pcd for frac in self.fracs)

    @property
    def depth(self):
        return self.num_pcd - len(self.fracs)

    @property
    def state(self):
        return set([str(set(frac.merge_state)) for frac in self.fracs])

    def merge(self, i, j):
        start = time.time()
        src = self.fracs[i].pcd
        ref = self.fracs[j].pcd
        
        src_coarse = src.clone()
        ref_coarse = ref.clone()
        while len(src_coarse) * len(ref_coarse) > 1e9:
            src_coarse = pcd_subsample(src_coarse)
            ref_coarse = pcd_subsample(ref_coarse)

        T = geo_transformer(src_coarse, ref_coarse)
        geo_transformer_time = time.time() - start
        
        if self.use_icp:
            T = open3d_icp(src, ref, trans_init=T)
            
        icp_time = time.time() - start - geo_transformer_time
        
        new_frac = self.fracs[i].merge(self.fracs[j], T)
        pcd_xor_time = time.time() - start - icp_time
        logger.info(f"geo_transformer_time : {geo_transformer_time:.2f}   icp_time : {icp_time:.2f}   pcd_xor_time : {pcd_xor_time:.2f}")
        del self.fracs[max(i, j)]
        del self.fracs[min(i, j)]
        self.fracs.append(new_frac)

    def search_one_step(self):
        k = 4
        if len(self.fracs) < k:
            pairs = [(i, j) for i in range(len(self.fracs)) for j in range(i+1, len(self.fracs))]
        else:
            topk_idx = np.argsort([frac.pcd.shape[0] for frac in self.fracs])[-k:]
            pairs = [(topk_idx[i], topk_idx[j]) for i in range(k) for j in range(i+1, k)]

        new_frac_set_lst = []
        for i, j in pairs:
            new_frac_set = deepcopy(self)
            new_frac_set.merge(i, j)
            new_frac_set_lst.append(new_frac_set)
            assert self.depth + 1 == new_frac_set.depth

        return new_frac_set_lst

    def search(self):
        assert self.depth == 0
        
        pq = []
        item = [self.depth, -self.n_removed, self]
        heapq.heappush(pq, item)

        # for top k search
        search_count = [0] * self.num_pcd

        while len(pq) > 0:
            
            depth, n_removed, frac_set = heapq.heappop(pq)
            n_removed = -n_removed

            if search_count[depth] >= self.k:
                continue

            search_count[depth] += 1
            logger.info(f"state={frac_set.state}   depth={depth}   n_removed={n_removed}   removed_ratio={(n_removed / self.n_origin_pcd):.2f}")

            if depth == self.num_pcd - 1:
                if search_count != [1] + [self.k] * (self.num_pcd - 2) + [1]:
                    raise Warning(f"search_count is incorrect: {search_count}")
                return frac_set

            new_frac_set_lst = frac_set.search_one_step()
            for new_frac_set in new_frac_set_lst:
                pq = push_frac_set(pq, frac_set=new_frac_set)

    def __str__(self) -> str:
        return str([str(frac) for frac in self.fracs])

    def __gt__(self, other):
        return str(self) > str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def state_eq(self, other):
        return self.depth == other.depth and set(self.fracs) == set(other.fracs)


def push_frac_set(pq, frac_set):
    for i, (_, _, frac_set_compare) in enumerate(pq):
        
        if frac_set.state_eq(frac_set_compare):
            if frac_set.n_removed > frac_set_compare.n_removed:
                pq[i] = (frac_set.depth, -frac_set.n_removed, frac_set)
                heapq.heapify(pq)
                return pq
            else:
                return pq
    heapq.heappush(pq, (frac_set.depth, -frac_set.n_removed, frac_set))
    return pq


def pointcloud_xor(src: torch.Tensor, ref: torch.Tensor, threshold=0.01):
    n_origin = src.shape[0] + ref.shape[0]

    src_coarse = src.clone()
    ref_coarse = ref.clone()
    src_threshold = ref_threshold = threshold
    
    while len(src_coarse) > 3e4:
        src_coarse = pcd_subsample(src_coarse)
        ref_threshold *= 1.414
        
    while len(ref_coarse) > 3e4:
        ref_coarse = pcd_subsample(ref_coarse)
        src_threshold *= 1.414
    
    dist_src, _ = knn(src, ref)
    dist_ref, _ = knn(ref, src)
    
    src = src[dist_src > src_threshold]
    ref = ref[dist_ref > ref_threshold]

    n_after_xor = src.shape[0] + ref.shape[0]
    n_removed = n_origin - n_after_xor

    return torch.cat((src, ref), dim=0), n_removed


def top_k_indices(matrix, k, only_triu=True):
    assert matrix.shape[0] == matrix.shape[1]
    n = matrix.size(0)
    k = min(k, n * (n - 1) / 2)
    k = int(k)

    # for upper triangle matrix replace into 0
    if only_triu:
        matrix = torch.triu(matrix, diagonal=1)

    # Flatten the matrix and get the values and indices of top-k elements
    values, indices = torch.topk(matrix.view(-1), k)

    # Convert to 2D indices
    row_indices = torch.div(indices, n, rounding_mode='floor')
    column_indices = indices % n
    indices = torch.stack((row_indices, column_indices), dim=1).tolist()

    return indices
