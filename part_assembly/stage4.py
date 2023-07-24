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
from multi_part_assembly.datasets.geometry_data import build_geometry_dataset, build_geometry_dataloader
from jhutil import open3d_icp
import numpy as np

from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port

from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights

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
        """merge two nodes

        Args:
            other (Node): other node
            T (torch.Tensor): homofrac_sety transformation matrix

        Returns:
            new_node: merged node
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
    def __init__(self, pcd_list, use_similarity=False, use_icp=True):
        self.fracs = [Fracture(pcd, i, 0, 0) for i, pcd in enumerate(pcd_list)]
        self.k = 3
        self.use_similarity = use_similarity
        self.use_icp = use_icp
        if self.use_similarity:
            self.update_similarity()

    def update_similarity(self):
        # TODO: 기존에 있는 similarity를 업데이트하는 방식으로 바꾸기
        self.feature_lst = torch.stack([pointnext(node.pcd) for node in self.fracs], dim=0)
        self.similarity = self.feature_lst @ self.feature_lst.T  # (n, n)
        self.similarity = (self.similarity + 1) / 2
        self.similarity = self.similarity - torch.eye(self.similarity.shape[0])

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
        src = self.fracs[i].pcd
        ref = self.fracs[j].pcd
        T = geo_transformer(src, ref)
        if self.use_icp:
            T = open3d_icp(ref, src, trans_init=T)
        new_frac = self.fracs[i].merge(self.fracs[j], T)
        del self.fracs[max(i, j)]
        del self.fracs[min(i, j)]
        self.fracs.append(new_frac)
        if self.use_similarity:
            self.update_similarity()

    def search_one_step(self):
        if self.use_similarity:
            pairs = top_k_indices(self.similarity, self.k)
        else:
            # TODO : object 개수 바탕으로 pair 개수 조절하기
            pairs = [(i, j) for i in range(len(self.fracs)) for j in range(i, len(self.fracs))]

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
            logger.info(f"state={frac_set.state}   depth={depth}   n_removed={n_removed}   ")
            
            if depth == self.num_pcd - 1:
                if search_count != [1] + [self.k] * (self.num_pcd - 2) + [1]:
                    raise Warning(f"search_count is incorrect: {search_count}")
                return frac_set

            new_frac_set_lst = frac_set.search_one_step()
            for new_frac_set in new_frac_set_lst:
                data = [new_frac_set.depth, -new_frac_set.n_removed, new_frac_set]
                if is_item_in_priority_queue(que, data):
                    continue
                que.put(data)

    def __str__(self) -> str:
        return str([str(node) for node in self.fracs])

    def __gt__(self, other):
        if self == other:
            return False
        return str(self) > str(other)

    def __lt__(self, other):
        if self == other:
            return False
        return str(self) < str(other)

    def __eq__(self, other):
        return set(self.fracs) == set(other.nodes)


def is_item_in_priority_queue(pq, item):
    for element in pq.queue:
        if element == item:
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
        final_node = result.nodes[0]
        pcd_xored = final_node.pcd

        import jhutil; jhutil.jhprint(1111, final_node.merge_state)
        import jhutil; jhutil.jhprint(2222, final_node.T_dic)
        import jhutil; jhutil.jhprint(5555, result.n_removed)

        n_removed = result.n_removed
        pcd_xored_re, n_removed_re = reproduce(pcd_list, final_node.merge_state)
        # assert torch.sum(pcd_xored - pcd_xored_re) < 1e-6, f"{torch.sum(pcd_xored - pcd_xored_re)}"
        if n_removed != n_removed_re:
            import jhutil; jhutil.jhprint(0000, f"{n_removed} is differs from {n_removed_re}")

    import jhutil; jhutil.jhprint(0000, "test_reproduce done")


def visualize_result():
    # TODO: result를 visulaize해서 사진으로 저장해 두기
    pass