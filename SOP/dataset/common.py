import numpy as np
import open3d
import random
import torch
import pickle
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


class FarthestSampler:
    '''
    A class for farthest point sampling (FPS)
    '''
    def __init__(self):
        pass

    def calc_distance(self, p0, points):
        '''
        Calculate one-to-all distances
        :param p0: single point
        :param points: point clouds
        :return: one-to-all distances
        '''
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, points, k):
        '''
        Downsample give point cloud (nx3) into smaller point cloud (kx3) via farthest point sampling
        :param points: input point cloud in shape (nx3)
        :param k: number of points after downsampling
        :return: farthest_points: point cloud in shape (kx3) after farthest point sampling
        '''
        farthest_points = np.zeros(shape=(k, 3))
        farthest_points[0] = points[np.random.randint(points.shape[0])]
        distances = self.calc_distance(farthest_points[0], points)
        for i in range(1, k):
            farthest_points[i] = points[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distance(farthest_points[i], points))
        return farthest_points


def make_rotation_matrix(augment_axis, augment_rotation):
    '''
    Generate a random rotaion matrix
    :param augment_axis: 0 or 1, if set to 1, returned matrix will describe a rotation around a random selected axis from {x, y, z}
    :param augment_rotation: float number scales the generated rotation
    :return: rotaion matrix
    '''
    angles = np.random.rand(3) * 2. * np.pi * augment_rotation
    rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = rx @ ry @ rz
    if augment_axis == 1:
        return random.choice([rx, ry, rz])
    return rx @ ry @ rz


def make_translation_vector(augment_translation):
    '''
    Generate a random translation vector
    :param augment_translation: float number scales the generated translation
    :return: translation vector
    '''
    T = np.random.rand(3) * augment_translation
    return T


def to_tsfm(rot, trans):
    '''
    Make (4, 4) transformation matrix from rotation matrix and translation vector
    :param rot: (3, 3) rotation matrix
    :param trans: (3, 1) translation matrix
    :return: (4, 4) transformation matrix
    '''
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def to_o3d_pcd(pts):
    '''
    From numpy array, make point cloud in open3d format
    :param pts: point cloud (nx3) in numpy array
    :return: pcd: point cloud in open3d format
    '''
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    return pcd


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    '''
    Give source & target point clouds as well as the relative transformation between them, calculate correspondences according to give threshold
    :param src_pcd: source point cloud
    :param tgt_pcd: target point cloud
    :param trans: relative transformation between source and target point clouds
    :param search_voxel_size: given threshold
    :param K: if k is not none, select top k nearest neighbors from candidate set after radius search
    :return: (m, 2) torch tensor, consisting of m correspondences
    '''
    src_pcd.transform(trans)
    pcd_tree = open3d.geometry.KDTreeFlann(tgt_pcd)


    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences


def compute_overlap_and_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    # Apply transformation to source point cloud
    src_pcd.transform(trans)

    # Build KD tree for target point cloud
    pcd_tree = open3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    src_points = np.asarray(src_pcd.points)

    # Compute overlap masks
    has_corr_src = np.zeros(len(src_points), dtype=bool)
    has_corr_tgt = np.zeros(len(tgt_pcd.points), dtype=bool)

    for i, src_point in enumerate(src_points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(src_point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])
            has_corr_src[i] = True
            has_corr_tgt[j] = True

    # Convert correspondences to numpy array and then to torch tensor
    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)

    return has_corr_src, has_corr_tgt, correspondences

def load_info(path):
    '''
    read a dictionary from a pickle file
    :param path: path to the pickle file
    :return: loaded info
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)

def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    """Random rotation matrix generated from uniformly distributed eular angles

    """
    # uniform angle_z, angle_y, angle_x
    euler = np.random.rand(
        3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation

def uniform_sample_rotation() -> np.ndarray:
    """Random rotation matrix generated from QR decomposition

    Rotation generated by this function is uniformly distributed on SO(3) w.r.t Haar measure

    NOTE: RRE of the rotation generated by this function is NOT uniformly distributed

    """
    # QR decomposition
    z = np.random.randn(3, 3)
    while np.linalg.matrix_rank(z) != z.shape[0]:
        z = np.random.randn(3, 3)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = np.diag(d / np.absolute(d))
    q = np.matmul(q, ph)
    # # if det(rotation) == -1, project q to so(3)
    # #rotation = np.linalg.det(q) * q # det(q) 有误差，用乘法可能放大误差 而用除法则可以一定程度上修正误差
    rotation = q / np.linalg.det(
        q)  # 虽然根据行列式数乘的性质，除以det(q)的3次方根更合适，但是感觉从精度上讲没有必要

    return rotation

def np_get_transform_from_rotation_translation(
        rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

def np_transform(g: np.ndarray, pts: np.ndarray):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([B,] 3/4, 4)
        pts: Points to be transformed ([B,] N, 3)

    Returns:
        transformed points of size (N, 3)
    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    transformed = pts[..., :3] @ np.swapaxes(rot, -1, -2) + trans[..., None, :]
    return transformed

def get_overlap_mask(ref_points, src_points, transform, matching_radius):
    r"""Compute overlap region mask

    Returns:
        ref_overlap: Whether each reference point is in the overlap region
        src_overlap: Whether each source point is in the overlap region
    """
    src_points = np_transform(transform, src_points)

    # Mask for reference's overlap region
    src_tree = cKDTree(src_points)
    _, indices_list = src_tree.query(
        ref_points,
        distance_upper_bound=matching_radius,
        workers=-1,
    )
    invalid_index = src_points.shape[0]
    ref_overlap = indices_list < invalid_index

    # Mask for source's overlap region
    ref_tree = cKDTree(ref_points)
    _, indices_list = ref_tree.query(
        src_points,
        distance_upper_bound=matching_radius,
        workers=-1,
    )
    invalid_index = ref_points.shape[0]
    src_overlap = indices_list < invalid_index

    return ref_overlap, src_overlap