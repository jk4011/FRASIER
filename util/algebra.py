import torch


def get_rotation_matrix(v1, v2):
    """Returns the rotation matrix required to rotate v1 to v2."""
    v1 = v1 / torch.norm(v1)  # normalize v1
    v2 = v2 / torch.norm(v2)  # normalize v2
    cos_angle = torch.dot(v1, v2)
    sin_angle = torch.sqrt(1 - cos_angle**2)
    axis = torch.cross(v1, v2)
    axis_norm = torch.norm(axis)
    if axis_norm != 0:
        axis = axis / axis_norm
    else:
        axis = torch.tensor([1, 0, 0], dtype=torch.float)  # default axis
    k = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=torch.float)
    rotation_matrix = torch.eye(3) + sin_angle * k + (1 - cos_angle) * torch.matmul(k, k)
    return rotation_matrix
