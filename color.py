import numpy as np
import open3d as o3d


def project(image):
    diagonal = np.ones(3).reshape((1, 1, 3))
    image = np.sum(image * diagonal, axis=2) / 3

    return image.round().astype(np.uint8)


def reset_diagonal(image, head, tail):
    diagonal = np.array([255, 255, 255], dtype=float)
    norm_d = np.linalg.norm(diagonal)

    v = np.array([x2 - x1 for x1, x2 in zip(head, tail)], dtype=float)
    norm_v = np.linalg.norm(v)

    axis = np.cross(v, diagonal)
    axis /= np.linalg.norm(axis)

    cos = np.sum(diagonal * v) / (norm_d * norm_v)
    angle = np.arccos(cos)

    rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle).T
    scale = norm_d / norm_v

    return (image - head) @ rotation[np.newaxis] * scale


def discard_cubic(image):
    return image.round().clip(0, 255).astype(np.uint8)


def discard_cylindrical(image, radius):
    z = 0.0, 0.0, 1.0
    diagonal = np.array([255, 255, 255], dtype=float)
    diagonal /= np.linalg.norm(diagonal)

    axis = np.cross(diagonal, z)
    axis /= np.linalg.norm(axis)

    cos = np.sum(diagonal * z)
    angle = np.arccos(cos)

    rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle).T

    image = image @ rotation
    norm = np.linalg.norm(image[:, :, :2], axis=2, keepdims=True)
    image = np.where(norm <= radius, image, 0.0)

    return (image @ rotation.T).round().clip(0, 255).astype(np.uint8)
