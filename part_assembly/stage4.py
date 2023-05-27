import torch
from jhutil import knn
from typing import Union
import random
from queue import PriorityQueue
from jhutil import matrix_transform
from copy import deepcopy


class Node:
    def __init__(self, pcd, id, n_removed, n_last_removed):
        self.id = id
        self.n_removed = n_removed
        self.n_last_removed = n_last_removed
        self.pcd = pcd
        # TODO: original pcd와 transformation matrix도 저장해 두기

    @property
    def num_pcd(self):
        def count(id):
            if isinstance(id, int):
                return 1
            else:
                return count(id[0]) + count(id[1])

        return count(self.id)

    def merge(self, other, T):
        """merge two nodes

        Args:
            other (Node): other node
            T (torch.Tensor): homography transformation matrix

        Returns:
            new_node: merged node
        """
        pcd_transformed = matrix_transform(T, other.pcd)
        new_pcd, n_last_removed = pointcloud_xor(self.pcd, pcd_transformed)
        n_removed = self.n_removed + other.n_removed + n_last_removed

        # since merging is commutitive, fix order.
        if self < other:
            id = [self.id, other.id]
        else:
            id = [other.id, self.id]

        return Node(new_pcd, id, n_removed, n_last_removed)

    def __eq__(self, other):
        return self.id == other.id

    def __gt__(self, other):
        return str(self.id) > str(other.id)

    def __lt__(self, other):
        return str(self.id) < str(other.id)

    def __hash__(self) -> int:
        return str(self.id).__hash__()

    def __str__(self):
        return str(self.id)


class Graph:
    def __init__(self, pcd_list, full_pair=False):
        nodes = [Node(pcd, i, 0, 0) for i, pcd in enumerate(pcd_list)]
        self.nodes = nodes
        self.k = 3
        self.update_similarity()
        self.full_pair = full_pair

    def update_similarity(self):
        # TODO: 기존에 있는 similarity를 업데이트하는 방식으로 바꾸기
        self.feature_lst = torch.stack([pointnext(node.pcd) for node in self.nodes], dim=0)
        self.similarity = self.feature_lst @ self.feature_lst.T  # (n, n)
        self.similarity = (self.similarity + 1) / 2
        self.similarity = self.similarity - torch.eye(self.similarity.shape[0])

    @property
    def n_removed(self):
        return sum([node.n_removed for node in self.nodes])

    @property
    def num_pcd(self):
        return sum(node.num_pcd for node in self.nodes)

    @property
    def depth(self):
        return self.num_pcd - len(self.nodes)

    def merge(self, i, j):
        T = geotransformer(self.nodes[i].pcd, self.nodes[j].pcd)
        # TODO: add icp
        new_node = self.nodes[i].merge(self.nodes[j], T)
        del self.nodes[max(i, j)]
        del self.nodes[min(i, j)]
        self.nodes.append(new_node)
        self.update_similarity()

    def search_one_step(self):
        # TODO : top p sample로 바꾸기
        if self.full_pair:
            pairs = [(i, j) for i in range(len(self.nodes)) for j in range(i, len(self.nodes))]
            pairs = [(i, j) for i in range(5) for j in range(i, 5)]
        else:
            pairs = top_k_indices(self.similarity, self.k)

        new_graph_lst = []
        for i, j in pairs:
            if i == j:
                continue
            new_graph = deepcopy(self)
            new_graph.merge(i, j)
            new_graph_lst.append(new_graph)

            print()
            assert self.depth + 1 == new_graph.depth

        return new_graph_lst

    def search(self):
        assert self.depth == 0
        que = PriorityQueue()
        que.put([self.depth, -self.n_removed, self])
        # TODO : top p sample로 바꾸기
        search_count = [0] * self.num_pcd

        while not que.empty():
            depth, n_removed, graph = que.get()
            n_removed = -n_removed
            if search_count[depth] >= self.k:
                continue
            else:
                search_count[depth] += 1
            if depth == self.num_pcd - 1:
                if search_count != [1] + [self.k] * (self.num_pcd - 2) + [1]:
                    raise Warning(f"search_count is not correct: {search_count}")
                return graph

            new_graph_lst = graph.search_one_step()
            for new_graph in new_graph_lst:
                que.put([new_graph.depth, -new_graph.n_removed, new_graph])

    def __str__(self) -> str:
        return str([str(node) for node in self.nodes])

    def __gt__(self, other):
        if self == other:
            return False
        return str(self) > str(other)

    def __lt__(self, other):
        if self == other:
            return False
        return str(self) < str(other)

    def __eq__(self, other: object):
        return set(self.nodes) == set(other.nodes)


def pointnext(pcd):
    feature = torch.ones(64)
    feature = feature / torch.norm(feature)
    return feature


def geotransformer(src, ref):
    T = torch.eye(4)
    return T


def pointcloud_xor(src: torch.Tensor, ref: torch.Tensor, threshold=0.01):
    n_origin = src.shape[0] + ref.shape[0]

    distance, _ = knn(src, ref)
    src = src[distance > threshold]
    distance, _ = knn(ref, src)
    ref = ref[distance > threshold]

    n_after_xor = src.shape[0] + ref.shape[0]
    n_removed = n_origin - n_after_xor

    return torch.cat((src, ref), dim=0), n_removed


def top_k_indices(matrix, k):
    assert matrix.shape[0] == matrix.shape[1]
    n = matrix.size(0)
    k = min(k, n * (n - 1) / 2)
    k = int(k)

    # for upper triangle matrix replace into -1
    matrix = torch.triu(matrix, diagonal=1)

    # Flatten the matrix and get the values and indices of top-k elements
    values, indices = torch.topk(matrix.view(-1), k)

    # Convert to 2D indices
    row_indices = torch.div(indices, n, rounding_mode='floor')
    column_indices = indices % n
    indices = torch.stack((row_indices, column_indices), dim=1).tolist()

    return indices


def reproduce(pcd_list, id):
    left, right = id
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

    final_pcd, n_removed = pointcloud_xor(pcd1, pcd2)
    n_removed += n_removed1 + n_removed2
    return final_pcd, n_removed


def test_reproduce():
    pcd_list = [torch.randn(1000, 3) for i in range(4)]
    result = Graph(pcd_list).search()
    pcd_xored = result.nodes[0].pcd
    
    
    import jhutil; jhutil.jhprint(0000, "BFS done")
    
    n_removed = result.n_removed
    pcd_xored_re, n_removed_re = reproduce(pcd_list, result.nodes[0].id)
    assert torch.sum(pcd_xored - pcd_xored_re) < 1e-6, f"{torch.sum(pcd_xored - pcd_xored_re)}"
    assert n_removed == n_removed_re, f"{n_removed} is differs from {n_removed_re}"
    
    import jhutil; jhutil.jhprint(0000, "test_reproduce done")


if __name__ == '__main__':
    test_reproduce()
    