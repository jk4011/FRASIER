from multi_part_assembly.datasets.geometry_data import build_geometry_dataset, build_geometry_dataloader, save_geometry_dataset
from multi_part_assembly.datasets.geometry_data import GeometryPartDataset
import torch
from torch.utils.data import Dataset, DataLoader
import trimesh
import os


class DatasetStage1(Dataset):
    def __init__(self,
                 datapath,
                 scale=7,
                 sample_weight=50000):
        self.dataset = torch.load(datapath)
        
        self.n_part_objs = []
        for data in self.dataset:
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
        raise ValueError('idx is out of range')
    
    
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
        broken_face_weight = area_faces * is_face_broken
        broken_face_weight = broken_face_weight / torch.sum(broken_face_weight)
        broken_sample = trimesh.sample.sample_surface(
            mesh, n_broken_sample, broken_face_weight.numpy())[0]
        broken_sample = torch.tensor(broken_sample)
        
        total_area_skin = torch.sum(area_faces[(is_face_broken.logical_not())])
        n_skin_sample = int(self.sample_weight * total_area_skin)
        skin_face_weight = area_faces * is_face_broken.logical_not()
        skin_face_weight = skin_face_weight / torch.sum(skin_face_weight)
        skin_sample = trimesh.sample.sample_surface(
            mesh, n_skin_sample, skin_face_weight.numpy())[0]
        skin_sample = torch.tensor(skin_sample)
        
        sample = torch.cat([broken_sample, skin_sample], dim=0)
        # sample = broken_sample
        broken_label = torch.cat([torch.ones(n_broken_sample), torch.zeros(n_skin_sample)], dim=0).bool()
        
        # shuffle
        perm = torch.randperm(len(sample))
        sample = sample[perm]
        broken_label = broken_label[perm]
        
        data = {
            'sample': sample,  # (N, 3)
            'broken_label': broken_label,  # (N, )
            'path': mesh_path,
        }
        return data
