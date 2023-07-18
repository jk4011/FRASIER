from multi_part_assembly.datasets.geometry_data import build_geometry_dataset, build_geometry_dataloader, save_geometry_dataset
from multi_part_assembly.datasets.geometry_data import GeometryPartDataset
import torch
from torch.utils.data import Dataset, DataLoader
import trimesh
import os
from copy import copy
import numpy as np

class Stage1SingleDataset(Dataset):
    def __init__(self,
                 data_root,
                 scale=7,
                 sample_weight=50000,
                 overfit=-1):
        self.dataset = torch.load(data_root)

        self.n_part_objs = []
        for i, data in enumerate(self.dataset):
            if overfit == i:
                break
            self.n_part_objs.append(len(data['file_names']))

        self.sample_weight = sample_weight
        self.scale = scale

    def __len__(self):
        return sum(self.n_part_objs)

    def get_idx(self, idx):
        for group_idx, n_part_obj in enumerate(self.n_part_objs):
            if idx < n_part_obj:
                return group_idx, idx
            else:
                idx -= n_part_obj

        raise IndexError

    def __getitem__(self, index):
        group_idx, part_idx = self.get_idx(index)
        group_obj = self.dataset[group_idx]

        is_broken_vertices = group_obj['is_broken_vertices'][part_idx]

        dir_name = group_obj['dir_name']
        file_name = group_obj['file_names'][part_idx]
        mesh_path = os.path.join(dir_name, file_name)
        mesh = trimesh.load_mesh(mesh_path)

        faces = torch.Tensor(mesh.faces)  # (f_i, 3)
        area_faces = torch.Tensor(mesh.area_faces)  # (f_i)

        is_face_broken = torch.zeros(len(faces), dtype=torch.bool)
        for i, vertex_indice in enumerate(faces):
            vertex_indice = vertex_indice.long()
            is_vertex_broken = is_broken_vertices[vertex_indice]  # (3, )
            is_face_broken[i] = torch.all(is_vertex_broken).item()

        total_area_broken = torch.sum(area_faces[is_face_broken])
        n_broken_sample = int(self.sample_weight * total_area_broken)
        n_broken_sample = max(1, n_broken_sample)

        total_area_skin = torch.sum(area_faces[(is_face_broken.logical_not())])
        n_skin_sample = int(self.sample_weight * total_area_skin)

        # if part object is too small, scale it.
        scale = copy(self.scale)
        while n_broken_sample + n_skin_sample < 1024:
            n_broken_sample *= 2
            n_skin_sample *= 2
            scale *= 1.414

        broken_face_weight = area_faces * is_face_broken
        broken_face_weight = broken_face_weight / torch.sum(broken_face_weight)
        broken_sample, face_indices = trimesh.sample.sample_surface(
            mesh, n_broken_sample, broken_face_weight.numpy())
        broken_sample = torch.tensor(broken_sample)
        broken_normal = torch.tensor(mesh.face_normals[face_indices])

        skin_face_weight = area_faces * is_face_broken.logical_not()
        skin_face_weight = skin_face_weight / torch.sum(skin_face_weight)
        skin_sample, face_indices = trimesh.sample.sample_surface(
            mesh, n_skin_sample, skin_face_weight.numpy())
        skin_sample = torch.tensor(skin_sample)
        skin_normal = torch.tensor(mesh.face_normals[face_indices])

        sample = torch.cat([broken_sample, skin_sample], dim=0)
        sample *= scale
        broken_label = torch.cat([torch.ones(n_broken_sample), torch.zeros(n_skin_sample)], dim=0).bool()
        normal = torch.cat([broken_normal, skin_normal], dim=0)

        # shuffle
        perm = torch.randperm(len(sample))
        sample = sample[perm]
        broken_label = broken_label[perm]
        normal = normal[perm]

        assert len(skin_sample) == n_skin_sample
        assert len(broken_sample) == n_broken_sample
        assert len(sample) >= 1024, f'{len(sample)}'
        data = {
            'sample': sample,  # (N, 3)
            'normal': normal,
            'broken_label': broken_label,  # (N, )
            'path': mesh_path,
        }
        return data

