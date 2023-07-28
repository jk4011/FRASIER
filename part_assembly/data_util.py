import os
import torch
import trimesh
import jhutil
from tqdm import tqdm


def _get_broken_pcs_idxs(points, threshold=0.0001):
    broken_pcs_idxs = []

    for i in range(len(points)):
        idx_i = torch.zeros(len(points[i]))
        idx_i = idx_i.to(torch.bool)

        for j in range(len(points)):
            if i == j:
                continue
            if not _box_overlap(points[i], points[j]):
                continue
            distances, _ = jhutil.knn(points[i], points[j])
            idx_i = torch.logical_or(idx_i, distances < threshold)
        broken_pcs_idxs.append(idx_i)

    return broken_pcs_idxs


def _box_overlap(src, ref):
    # src : (N, 3)
    # ref : (M, 3)
    src_min = src.min(axis=0)[0]  # (3,)
    src_max = src.max(axis=0)[0]  # (3,)
    ref_min = ref.min(axis=0)[0]  # (3,)
    ref_max = ref.max(axis=0)[0]  # (3,)

    # Check x-axis overlap
    if src_max[0] < ref_min[0] or src_min[0] > ref_max[0]:
        return False

    # Check y-axis overlap
    if src_max[1] < ref_min[1] or src_min[1] > ref_max[1]:
        return False

    # Check z-axis overlap
    if src_max[2] < ref_min[2] or src_min[2] > ref_max[2]:
        return False

    return True


def _parse_data(data_folder):
    """Read mesh and parse it"""
    file_names = os.listdir(data_folder)
    file_names = [fn for fn in file_names if fn.endswith('.obj')]

    if len(file_names) == 0:
        return None

    # read mesh and sample points
    meshes = [
        trimesh.load(os.path.join(data_folder, mesh_file))
        for mesh_file in file_names
    ]

    # parsing into list
    vertices_all = []  # (N, v_i, 3)
    faces_all = []  # (N, f_i, 3)
    area_faces_all = []  # (N, f_i)
    for mesh in meshes:
        faces = torch.Tensor(mesh.faces)  # (f_i, 3)
        vertices = torch.Tensor(mesh.vertices)  # (v_i, 3)
        area_faces = torch.Tensor(mesh.area_faces)  # (f_i)

        faces_all.append(faces)
        vertices_all.append(vertices)
        area_faces_all.append(area_faces)

    is_pts_broken_all = _get_broken_pcs_idxs(vertices_all, threshold=0.0001)  # (N, v_i)
    is_broken_face_all = []  # (N, f_i)
    for faces, is_pts_broken, vertices in zip(faces_all, is_pts_broken_all, vertices_all):

        is_face_broken = []  # (f_i, )
        for vertex_idx in faces:
            vertex_idx = vertex_idx.long()
            is_vertex_broken = is_pts_broken[vertex_idx]  # (3, )
            is_face_broken.append(torch.all(is_vertex_broken))
        is_broken_face_all.append(torch.tensor(is_face_broken))

    # TODO: error나면 copy 때문인지 확인하기
    data = {
        "meshes": meshes,
        'is_broken_vertices': is_pts_broken_all,  # (N, v_i)
        'is_broken_face': is_broken_face_all,  # (N, f_i)
        'file_names': file_names,  # (N, )
        'dir_name': data_folder,
    }
    return data


def create_mesh_info(raw_paths, processed_paths):
    assert len(raw_paths) == len(processed_paths)
    for data_folder, processed_path in tqdm(list(zip(raw_paths, processed_paths))):
        assert processed_path.endswith("mesh_info.pt")
        if os.path.exists(processed_path):
            continue

        data = _parse_data(data_folder)
        if data is None:
            continue

        directory = os.path.dirname(processed_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(data, processed_path)


def sample_pcd(mesh, is_face_broken,
               sample_weight=15000,
               max_n_sample=None,
               omit_large_n=False,
               min_n_sample=None,
               omit_small_n=False,
               scale=7., ):

    area_faces = torch.Tensor(mesh.area_faces)  # (f_i)

    total_area_broken = torch.sum(area_faces[is_face_broken])

    n_broken_sample = int(sample_weight * total_area_broken)
    n_broken_sample = max(1, n_broken_sample)

    total_area_skin = torch.sum(area_faces[(is_face_broken.logical_not())])
    n_skin_sample = int(sample_weight * total_area_skin)

    if max_n_sample is not None and n_broken_sample + n_skin_sample > max_n_sample:
        if omit_large_n:
            return None
        # make the sum of two equal to max_sample
        n_broken_sample = int(n_broken_sample * (max_n_sample / (n_broken_sample + n_skin_sample)))
        n_skin_sample = max_n_sample - n_broken_sample
    if min_n_sample is not None and n_broken_sample + n_skin_sample < min_n_sample:
        if omit_small_n:
            return None
        n_broken_sample = int(n_broken_sample * (min_n_sample / (n_broken_sample + n_skin_sample)))
        n_skin_sample = min_n_sample - n_broken_sample

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
    data = {
        'sample': sample,  # (N, 3)
        'normal': normal,
        'broken_label': broken_label,  # (N, )
    }
    return data


def sample_from_mesh_info(mesh_info, **kwargs):
    pcd_20k = {'sample': [], 'normal': [], 'broken_label': []}
    for i in range(len(mesh_info['meshes'])):
        mesh = mesh_info['meshes'][i]
        is_broken_face = mesh_info['is_broken_face'][i]
        data = sample_pcd(mesh=mesh, is_face_broken=is_broken_face, **kwargs)
        if data is None:
            continue
        pcd_20k['sample'].append(data['sample'])
        pcd_20k['normal'].append(data['normal'])
        pcd_20k['broken_label'].append(data['broken_label'])

    return pcd_20k
