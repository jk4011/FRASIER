import torch
from torch_geometric.data import InMemoryDataset, Dataset
from functools import lru_cache
import os
from scipy.spatial.transform import Rotation as R
from part_assembly.data_util import create_mesh_info, sample_broken_noisy, recenter_pc, rotate_pc
from tqdm import tqdm
from torch.utils.data import DataLoader

from pytorch3d.transforms import matrix_to_quaternion, matrix_to_axis_angle, \
    quaternion_to_matrix, quaternion_to_axis_angle, \
    axis_angle_to_quaternion, axis_angle_to_matrix


class Stage3PairDataset(Dataset):
    def __init__(self,
                 data_dir,
                 data_fn,
                 category,
                 overlap_threshold,
                 sample_weight,
                 overfit,
                 scale,
                 dataset_name,
                 mode,
                 ):

        self.data_fn = data_fn
        self.overlap_threshold = overlap_threshold
        self.category = category
        self.sample_weight = sample_weight
        self.overfit = overfit
        self.scale = scale
        self.dataset_name = dataset_name
        self.mode = mode
        super().__init__(root=data_dir, transform=None, pre_transform=None)
        self.pair_files = os.listdir(self.pcd_pair_dir_path)
        self.pair_files = [os.path.join(self.pcd_pair_dir_path, fn) for fn in self.pair_files]

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
                if 2 <= num_parts:
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
        return os.path.join(self.processed_dir, self.dataset_name, f"pcd_broken_noisy_{self.mode}")

    def len(self):
        return len(self.pair_files)

    def process(self):
        assert len(self.mesh_info_paths) == len(self.pcd_broken_noisy_paths)

        # print('1. prcessing mesh_info.pt...')
        # create_mesh_info(self.raw_paths, self.mesh_info_paths)

        print('2. prcessing pcd_broken_noisy.pt...')
        if not os.path.exists(self.pcd_pair_dir_path):
            os.mkdir(self.pcd_pair_dir_path)

        i = 0
        for mesh_info_path, pcd_broken_noisy_path in tqdm(list(zip(self.mesh_info_paths, self.pcd_broken_noisy_paths))):
            assert mesh_info_path.endswith("mesh_info.pt")
            assert pcd_broken_noisy_path.endswith("pcd_broken_noisy.pt")
            assert os.path.dirname(mesh_info_path) == os.path.dirname(pcd_broken_noisy_path)

            # if os.path.exists(pcd_broken_noisy_path):
            #     continue
            
            mesh_info = torch.load(mesh_info_path)

            pcd_broken_noisy = sample_broken_noisy(mesh_info, scale=self.scale)
            
            directory = os.path.dirname(pcd_broken_noisy_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(pcd_broken_noisy, pcd_broken_noisy_path)

            # save pcd_broken_noisy.pt as pcd_broken_noisy/{i}.pt
            broken_pcs = pcd_broken_noisy["broken_pcs"]
            
                
            overlap_ratio = mesh_info["overlap_ratio"]
            adjacent_pair = get_adjacent_pair(overlap_ratio, self.overlap_threshold)
            # TODO: overlap ratio 기반으로 pair 만들기

            for pair_idx in adjacent_pair:
                pair_data = get_pair_data(broken_pcs, overlap_ratio, pair_idx)

                pair_file = os.path.join(self.pcd_pair_dir_path, f"{i}.pt")
                torch.save(pair_data, pair_file)
                try:
                    torch.load(pair_file)
                except:
                    os.remove(pair_file)
                    continue
                i += 1

    def get(self, idx):
        try:
            return torch.load(self.pair_files[idx])
        except:
            os.remove(self.pair_files[idx])
            return None


def get_pair_data(broken_pcs, overlap_ratio, pair_idx):
    i, j = pair_idx
    src = broken_pcs[i]
    ref = broken_pcs[j]
    overlap_score = overlap_ratio[i, j] * overlap_ratio[j, i]

    rot_mat = R.random().as_matrix()
    src, src_trans = recenter_pc(src.float())
    src, src_quat = rotate_pc(src.float(), rot_mat)

    rot_mat = R.random().as_matrix()
    ref, ref_trans = recenter_pc(ref.float())
    ref, ref_quat = rotate_pc(ref.float(), rot_mat)

    transform = relative_transform_matrix(
        src_quat, ref_quat, src_trans, ref_trans)

    out = {
        "ref_points": ref.contiguous(),
        "src_points": src.contiguous(),
        "ref_feats": torch.ones(len(src), 1),
        "src_feats": torch.ones(len(ref), 1),
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


def get_adjacent_pair(overlap_ratio, overlap_threshold=0.01):
    assert isinstance(overlap_ratio, torch.Tensor)
    n = overlap_ratio.shape[0]
    assert overlap_ratio.shape == (n, n)

    adjacent_pair = []

    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            if overlap_ratio[i, j] * overlap_ratio[j, i] < overlap_threshold:
                continue
            adjacent_pair.append([i, j])

    return adjacent_pair


def build_sample_broken_train_dataset(cfg):

    print("building sample_broekn_data train dataset...")
    print(f"overfit: {cfg.overfit}")
    data_dict = dict(
        data_dir=cfg.data_dir,
        data_fn=cfg.data_fn.format('train'),
        category=cfg.category,
        overlap_threshold=cfg.overlap_threshold,
        sample_weight=cfg.sample_weight,
        overfit=cfg.overfit,
        scale=cfg.scale,
        dataset_name=cfg.data_fn.split('.')[0],
        mode='train',
    )
    return Stage3PairDataset(**data_dict)


def build_sample_broken_val_dataset(cfg):

    print("building sample_broekn_data val dataset...")
    print(f"overfit: {cfg.overfit}")

    data_dict = dict(
        data_dir=cfg.data_dir,
        data_fn=cfg.data_fn.format('val'),
        category=cfg.category,
        overlap_threshold=cfg.overlap_threshold,
        sample_weight=cfg.sample_weight,
        overfit=cfg.overfit,
        scale=cfg.scale,
        dataset_name=cfg.data_fn.split('.')[0],
        mode='val',
    )
    return Stage3PairDataset(**data_dict)


def build_sample_broken_dataloader(cfg):
    train_set = build_sample_broken_train_dataset(cfg)
    val_set = build_sample_broken_val_dataset(cfg)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=1,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader
