import numpy as np
import meshplot as mp
import math
import random
import plotly.graph_objs as go
import numpy as np
import trimesh
import numpy as np
import matplotlib.pyplot as plt

import os



def parse_obj_file(filename):
    positions = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Extract vertex position
                parts = line.split()
                position = [float(parts[1]), float(parts[2]), float(parts[3])]
                positions.append(position)
            elif line.startswith('f '):
                # Extract face indices
                parts = line.split()
                v1 = int(parts[1].split('/')[0]) - 1  # Subtract 1 because OBJ indices are 1-based
                v2 = int(parts[2].split('/')[0]) - 1
                v3 = int(parts[3].split('/')[0]) - 1
                faces.append([v1, v2, v3])

    positions = np.array(positions, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    return positions, faces

    

def show_obj(obj_file, color=[1, 0, 0], library="go"):
    vertices, faces = parse_obj_file(obj_file)
    if library == "meshplot":
        mp.plot(vertices, faces, c=color)
    elif library == "go":
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k)
        layout = go.Layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                                       aspectmode='data',
                                       aspectratio=dict(x=1, y=1, z=1)))

        fig = go.Figure(data=[mesh], layout=layout)
        fig.show()

def random_rotate(vertices):
    theta_x = random.uniform(0, math.pi * 2)
    theta_y = random.uniform(0, math.pi * 2)
    theta_z = random.uniform(0, math.pi * 2)
    rotation_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta_x), -math.sin(theta_x)],
        [0, math.sin(theta_x), math.cos(theta_x)]
    ])
    rotation_y = np.array([
        [math.cos(theta_y), 0, math.sin(theta_y)],
        [0, 1, 0],
        [-math.sin(theta_y), 0, math.cos(theta_y)]
    ])
    rotation_z = np.array([
        [math.cos(theta_z), -math.sin(theta_z), 0],
        [math.sin(theta_z), math.cos(theta_z), 0],
        [0, 0, 1]
    ])
    rotation = np.matmul(rotation_z, np.matmul(rotation_y, rotation_x))
    vertices = np.matmul(vertices, rotation)
    return vertices


def show_multiple_objs(obj_files, colors=None, is_random_rotate=False, library="go"):
    if library == "meshplot":
        v, f = parse_obj_file(obj_files[0])
        p = mp.plot(v, f)
        
        for i, obj_file in enumerate(obj_files):
            if i == 0:
                continue
            vertices, faces = parse_obj_file(obj_file)
            if is_random_rotate:
                vertices = random_rotate(vertices)
            # color = colors[i] if colors and len(colors) > i else None
            color = np.array(colors[i]) if colors and len(colors) > i else None
            p.add_mesh(vertices, faces, c=color) # shading={"wireframe": True}
    elif library == "go":
        meshes = []

        for idx, obj_file in enumerate(obj_files):
            vertices, faces = parse_obj_file(obj_file)
            x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

            if colors is None:
                def rgb_to_hex(rgb):
                    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
                pallete = [[0, 204, 0], [204, 0, 0], [0, 0, 204], [127, 127, 0], [127, 0, 127], [0, 127, 127], [76, 153, 0], [153, 0, 76], [76, 0, 153], [153, 76, 0], [76, 0, 153], [153, 0, 76], [204, 51, 127], [204, 51, 127], [51, 204, 127], [51, 127, 204], [127, 51, 204], [127, 204, 51], [76, 76, 178], [76, 178, 76], [178, 76, 76]]
                pallete = [rgb_to_hex(color) for color in pallete]
                color = pallete[idx % len(pallete)]

            mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color)
            meshes.append(mesh)
        
        layout = go.Layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                xaxis=dict(visible=False),  # Disable x-axis
                yaxis=dict(visible=False),  # Disable y-axis
                zaxis=dict(visible=False)   # Disable z-axis
            ),
        )

        fig = go.Figure(data=meshes, layout=layout)
        fig.show()



def show_meshes(folder_path):
    file_list = os.listdir(folder_path)
    file_list = [os.path.join(folder_path, file) for file in file_list]
    show_multiple_objs(file_list)

def sample_point_cloud_from_mesh(mesh_file, num_points):
    # Load the mesh
    mesh = trimesh.load_mesh(mesh_file)

    # Sample the point cloud
    point_cloud, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    return point_cloud

def show_point_clouds(point_clouds, colors=None, normals=None, is_random_rotate=False, s=None, range=((-1.1), (-1.1), (-1.1))):
    # type check point_clouds is list
    # if isinstance(point_clouds, list):
    #     point_clouds = np.array(point_clouds)
    if s is None:
        s = [0.3] * len(point_clouds)

    if colors is None:
        def rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
        colors = [[0, 204, 0], [204, 0, 0], [0, 0, 204], [127, 127, 0], [127, 0, 127], [0, 127, 127], [76, 153, 0], [153, 0, 76], [76, 0, 153], [153, 76, 0], [76, 0, 153], [153, 0, 76], [204, 51, 127], [204, 51, 127], [51, 204, 127], [51, 127, 204], [127, 51, 204], [127, 204, 51], [76, 76, 178], [76, 178, 76], [178, 76, 76]]
        colors = [rgb_to_hex(color) for color in colors]
        colors = colors[:len(point_clouds)]
    
    # Set up the figure and axis for the 3D plot
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    if normals is None:
        # Scatter plot for the 3D coordinates
        for points, color, s_ in zip(point_clouds, colors, s):
            if is_random_rotate:
                points = random_rotate(points)
            
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], '.', color=color, s=s_)
    else:
        # Scatter plot for the 3D coordinates
        for points, normal, color, s_ in zip(point_clouds, normals, colors, s):
            if is_random_rotate:
                points = random_rotate(points)
            
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], '.', color=color, s=s_)
            ax.quiver(points[:, 0], points[:, 1], points[:, 2], normal[:, 0], normal[:, 1], normal[:, 2], length=0.05 ,normalize=True, color=color, alpha=0.2)
            location_mean = points.mean(axis=0)
            normal_mean = normal.mean(axis=0)
            ax.quiver(location_mean[0], location_mean[1], location_mean[2], 
                      normal_mean[0], normal_mean[1], normal_mean[2], length=0.5, color=color, alpha=1)

    # Set labels for the axis
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Set the range for each axis
    ax.set_xlim(range[0])
    ax.set_ylim(range[1])
    ax.set_zlim(range[2])
    
    # Display the plot
    plt.show()