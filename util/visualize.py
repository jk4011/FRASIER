import numpy as np
import meshplot as mp
import math
import random
import plotly.graph_objs as go
import trimesh
import matplotlib.pyplot as plt

import os

# FIXME : This is a hack to make it work on the server



def sample_point_cloud_from_mesh(mesh_file, num_points):
    # Load the mesh
    mesh = trimesh.load_mesh(mesh_file)

    # Sample the point cloud
    point_cloud, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    return point_cloud
