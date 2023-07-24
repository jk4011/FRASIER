import torch
import trimesh
import os
from copy import copy
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardFlatShader,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex,
    MeshRendererWithFragments,
)
# from pytorch3d.renderer.mesh.shader import NormalShader
from pytorch3d.io import load_objs_as_meshes, load_obj

from random import randint
from torch_geometric.data import Dataset, InMemoryDataset, download_url


from pytorch3d.ops import interpolate_face_attributes


from pytorch3d.renderer.mesh.shader import ShaderBase, Fragments, Meshes


class NormalShader(ShaderBase):
    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        faces = meshes.faces_packed()  # (F, 3)

        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]

        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )

        return pixel_normals


class Stage1NormalDataset(InMemoryDataset):
    def __init__(self, root="/data/wlsgur4011/part_assembly/data", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["artifact.val.pth"]  # , "artifact.train.pth", "everyday.val.pth", "everyday.train.pth"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def random_dir_render(self, data):
        meshes = load_objs_as_meshes([os.path.join(data["dir_name"], file_name)
                                     for file_name in data["file_names"]], device="cpu")
        R, T = look_at_view_transform(0.7, randint(0, 90), randint(0, 90))

        cameras = FoVPerspectiveCameras(device='cpu', R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=NormalShader()
        )
        images, fragment = renderer(meshes)
        images = images.squeeze()
        return images

    def process(self):

        dataset_list = [torch.load(path) for path in self.raw_paths]

        if self.pre_filter is not None:
            dataset_list = [data for data in dataset_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset_list = [self.pre_transform(data) for data in dataset_list]

        image_all = []
        for dataset in dataset_list:
            for data in dataset:
                images = self.random_dir_render(data)
                image_all.append(images)

        image_all = torch.concat(image_all, dim=0)
        torch.save(image_all, self.processed_paths[0])
