import torch
from jhutil import knn
from typing import Union
import random
from queue import PriorityQueue
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
    def __init__(self, pcd, merge_state, n_removed, n_last_removed, T_dic=None):
        self.pcd = pcd
        self.merge_state = merge_state
        self.n_removed = n_removed
        self.n_last_removed = n_last_removed

        if T_dic is None:
            assert isinstance(merge_state, int)
            T_dic = {merge_state: torch.eye(4)}
        self.T_dic = T_dic
        # TODO: bundle adjustment를 위해 original pcd도 저장해 두기

    @property
    def num_pcd(self):
        def count(merge_state):
            if isinstance(merge_state, int):
                return 1
            else:
                return count(merge_state[0]) + count(merge_state[1])

        return count(self.merge_state)

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

        # since merging is commutitive, fix order.
        if self < other:
            merge_state = [self.merge_state, other.merge_state]
        else:
            merge_state = [other.merge_state, self.merge_state]

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
        self.fracs : list[Fracture] = [Fracture(pcd, merge_state=i, n_removed=0, n_last_removed=0) for i, pcd in enumerate(pcd_list)]
        self.k = 3
        self.use_icp = use_icp
        self.use_similarity = False

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
        return set([str(frac.merge_state) for frac in self.fracs])

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
            T = open3d_icp(src_coarse, ref_coarse, trans_init=T)
            
        icp_time = time.time() - start - geo_transformer_time
        
        new_frac = self.fracs[i].merge(self.fracs[j], T)
        pcd_xor_time = time.time() - start - icp_time
        logger.info(f"geo_transformer_time : {geo_transformer_time:.2f}   icp_time : {icp_time:.2f}   pcd_xor_time : {pcd_xor_time:.2f}")
        del self.fracs[max(i, j)]
        del self.fracs[min(i, j)]
        self.fracs.append(new_frac)
        if self.use_similarity:
            self.update_similarity()

    def search_one_step(self):
        if self.use_similarity:
            pairs = top_k_indices(self.similarity, self.k)
        elif len(self.fracs) < 4:
            pairs = [(i, j) for i in range(len(self.fracs)) for j in range(i+1, len(self.fracs))]
        else:
            top4_idx = np.argsort([frac.pcd.shape[0] for frac in self.fracs])[-4:]
            pairs = [(top4_idx[i], top4_idx[j]) for i in range(4) for j in range(i+1, 4)]

        new_frac_set_lst = []
        for i, j in pairs:
            if i == j:
                continue
            new_frac_set = deepcopy(self)
            new_frac_set.merge(i, j)
            new_frac_set_lst.append(new_frac_set)
            assert self.depth + 1 == new_frac_set.depth

        return new_frac_set_lst

    def search(self):
        assert self.depth == 0
        
        que = PriorityQueue()
        que.put([self.depth, -self.n_removed, self])

        # for top k search
        search_count = [0] * self.num_pcd

        while not que.empty():
            
            depth, n_removed, frac_set = que.get()
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
                data = [new_frac_set.depth, -new_frac_set.n_removed, new_frac_set]
                if is_item_in_priority_queue(que, frac_set=new_frac_set):
                    continue
                que.put(data)

    def __str__(self) -> str:
        return str([str(frac) for frac in self.fracs])

    def __gt__(self, other):
        if self == other:
            return False
        return str(self) > str(other)

    def __lt__(self, other):
        if self == other:
            return False
        return str(self) < str(other)

    def __eq__(self, other):
        return self.depth == other.depth and set(self.fracs) == set(other.fracs)


def is_item_in_priority_queue(pq, frac_set):
    for _, _, frac_set_compare in pq.queue:
        if frac_set == frac_set_compare:
            return True
    return False


def pointcloud_xor(src: torch.Tensor, ref: torch.Tensor, threshold=0.01):
    n_origin = src.shape[0] + ref.shape[0]

    dist_src, _ = knn(src, ref)
    dist_ref, _ = knn(ref, src)
    src = src[dist_src > threshold]
    ref = ref[dist_ref > threshold]

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


def reproduce(pcd_list, merge_state):
    # TODO: class frac_set로 바꾸기
    left, right = merge_state

    if not isinstance(left, int):
        pcd1, n_removed1 = reproduce(pcd_list, left)
    else:
        pcd1 = pcd_list[left]
        n_removed1 = 0

    if not isinstance(right, int):
        pcd2, n_removed2 = reproduce(pcd_list, right)
    else:
        pcd2 = pcd_list[right]
        n_removed2 = 0

    T = geo_transformer(pcd1, pcd2)
    pcd2_transformed = matrix_transform(T, pcd2)

    pcd_xored, n_removed = pointcloud_xor(pcd1, pcd2_transformed)
    n_removed += n_removed1 + n_removed2
    return pcd_xored, n_removed


def test_reproduce(n_iter=10, n_obj_threshold=5):
    cfg = load_yaml("/data/wlsgur4011/part_assembly/yamls/data_example.yaml")
    train_loader, val_loader = build_geometry_dataloader(cfg, use_saved=True)

    i = 0
    for data in val_loader:
        pcd_list = data["broken_pcs"]
        if len(pcd_list) > n_obj_threshold:
            continue
        if i == n_iter:
            break
        i += 1

        # TODO: point cloud는 gpu memory에서만 다루기
        # pcd_list = to_cuda(pcd_list)

        result = FractureSet(pcd_list).search()
        final_frac = result.fracs[0]

        import jhutil; jhutil.jhprint(1111, final_frac.merge_state)
        import jhutil; jhutil.jhprint(2222, final_frac.T_dic)
        import jhutil; jhutil.jhprint(5555, result.n_removed)

        n_removed = result.n_removed
        pcd_xored_re, n_removed_re = reproduce(pcd_list, final_frac.merge_state)
        # assert torch.sum(pcd_xored - pcd_xored_re) < 1e-6, f"{torch.sum(pcd_xored - pcd_xored_re)}"
        if n_removed != n_removed_re:
            import jhutil; jhutil.jhprint(0000, f"{n_removed} is differs from {n_removed_re}")

    import jhutil; jhutil.jhprint(0000, "test_reproduce done")


def visualize_result():
    # TODO: result를 visulaize해서 사진으로 저장해 두기
    pass