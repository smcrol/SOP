from typing import Union, Tuple, List
from dataset.common import to_o3d_pcd
from common.se3_numpy import *

import numpy as np
import torch
import open3d as o3d


def compute_overlap(src: Union[np.ndarray, o3d.geometry.PointCloud],
                    tgt: Union[np.ndarray, o3d.geometry.PointCloud],
                    search_voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Computes region of overlap between two point clouds.

    Args:
        src: Source point cloud, either a numpy array of shape (N, 3) or
          Open3D PointCloud object
        tgt: Target point cloud similar to src.
        search_voxel_size: Search radius

    Returns:
        has_corr_src: Whether each source point is in the overlap region
        has_corr_tgt: Whether each target point is in the overlap region
        src_tgt_corr: Indices of source to target correspondences
    """

    if isinstance(src, np.ndarray):
        src_pcd = to_o3d_pcd(src)
        src_xyz = src
    else:
        src_pcd = src
        src_xyz = np.asarray(src.points)

    if isinstance(tgt, np.ndarray):
        tgt_pcd = to_o3d_pcd(tgt)
        tgt_xyz = tgt
    else:
        tgt_pcd = tgt
        tgt_xyz = tgt.points

    # Check which points in tgt has a correspondence (i.e. point nearby) in the src,
    # and then in the other direction. As long there's a point nearby, it's
    # considered to be in the overlap region. For correspondences, we require a stronger
    # condition of being mutual matches
    tgt_corr = np.full(tgt_xyz.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(src_pcd)
    for i, t in enumerate(tgt_xyz):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(t, search_voxel_size)
        if num_knn > 0:
            tgt_corr[i] = knn_indices[0]
    src_corr = np.full(src_xyz.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    for i, s in enumerate(src_xyz):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(s, search_voxel_size)
        if num_knn > 0:
            src_corr[i] = knn_indices[0]

    has_corr_src = src_corr >= 0
    has_corr_tgt = tgt_corr >= 0

    return has_corr_src, has_corr_tgt


def compute_overlaps(batch):
    """Compute groundtruth overlap for each point+level. Note that this is a
    approximation since
    1) it relies on the pooling indices from the preprocessing which caps the number of
       points considered
    2) we do a unweighted average at each level, without considering the
       number of points used to generate the estimate at the previous level
    """

    overlaps = (batch['src_overlap'], batch['tgt_overlap'])
    # kpconv_meta = batch['kpconv_meta']
    n_pyr = len(batch['points'])

    overlap_pyr = {'pyr_0': torch.cat(overlaps, dim=0).type(torch.float)}
    invalid_indices = [s.sum() for s in batch['stack_lengths']]
    for p in range(1, n_pyr):
        pooling_indices = batch['pools'][p - 1].clone()
        valid_mask = pooling_indices < invalid_indices[p - 1]
        pooling_indices[~valid_mask] = 0

        # Average pool over indices
        overlap_gathered = overlap_pyr[f'pyr_{p-1}'][pooling_indices] * valid_mask
        overlap_gathered = torch.sum(overlap_gathered, dim=1) / torch.sum(valid_mask, dim=1)
        overlap_gathered = torch.clamp(overlap_gathered, min=0, max=1)
        overlap_pyr[f'pyr_{p}'] = overlap_gathered

    return overlap_pyr


def trans_loss(gt_transformed_src, pred_transformed_src):

    loss = torch.abs(pred_transformed_src - gt_transformed_src)

    return loss


def overlap(batch):
    src_xyz, tgt_xyz = batch['src_pcd_raw'], batch['tgt_pcd_raw']
    rot, trans = batch['rot'], batch['trans']
    rot, trans, src_xyz, tgt_xyz = rot.cpu(), trans.cpu(), src_xyz.cpu(), tgt_xyz.cpu()
    pose = se3_init(rot, trans)
    src_trans = se3_transform(pose, src_xyz)
    src_overlap, tgt_overlap = compute_overlap(src_trans, np.array(tgt_xyz), search_voxel_size=0.0375)
    src_overlap, tgt_overlap = torch.tensor(src_overlap).cuda(), torch.tensor(tgt_overlap).cuda()
    overlap_pyr = compute_overlaps(batch, src_overlap, tgt_overlap)

    return overlap_pyr