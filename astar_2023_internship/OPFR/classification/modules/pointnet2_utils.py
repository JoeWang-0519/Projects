"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch

try:
    from modules.pointops.functions.pointops import furthestsampling, gathering, ballquery, knnquery, \
        grouping, interpolation, nearestneighbor
except:
    raise Exception('Failed to load pointops')


def square_distance(src, dst):
    """
    Calculate Squared distance between each two points.

    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def square_distance_4dim(src, dst):
    '''
    :param src: [B, N, k1, 3]
    :param dst: [B, N, k2, 3]
    :return: square distance: [B, N, k1, k2]
    '''
    B, N, k1, _ = src.shape
    _, _, k2, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 1, 3, 2)) # [B, N, k1, k2]
    dist += torch.sum(src ** 2, -1).view(B, N, k1, 1)
    dist += torch.sum(dst ** 2, -1).view(B, N, 1, k2)
    return dist

def index_points(points, idx, cuda=False, is_group=False):
    '''
    :param points: [B, N, 3] / [B, N, k1, 3]
    :param idx: [B, N, k] / [B, N, k2, k3]
        Here, k3 is sampled from k1
    :return: [B, N, k, 3] / [B, N, k2, k3, 3]
    '''
    if cuda:
        if is_group:
            points = grouping(points.transpose(1, 2).contiguous(), idx)
            return points.permute(0, 2, 3, 1).contiguous()
        else:
            points = gathering(points.transpose(1, 2).contiguous(), idx)
            return points.permute(0, 2, 1).contiguous()

    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape) # [B, N, k]
    view_shape[1:] = [1] * (len(view_shape) - 1) # [B, 1, 1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1 # [1, N, k]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def index_points_4dim(points, idx, cuda=False, is_group=False):
    '''
    :param points: [B, N, k1, 3]
    :param idx: [B, N, k2, k3]
        Here, k3 is sampled from k1
    :return: [B, N, k2, k3, 3]
    '''
    if cuda:
        if is_group:
            points = grouping(points.transpose(1, 2).contiguous(), idx)
            return points.permute(0, 2, 3, 1).contiguous()
        else:
            points = gathering(points.transpose(1, 2).contiguous(), idx)
            return points.permute(0, 2, 1).contiguous()

    device = points.device
    B, N, _, _ = points.shape

    batch_shape = list(idx.shape) # [B, N, k2, k3]
    batch_shape[1:] = [1] * (len(batch_shape) - 1) # [B, 1, 1, 1]
    batch_repeat_shape = list(idx.shape)
    batch_repeat_shape[0] = 1 # [1, N, k2, k3]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(batch_shape).repeat(batch_repeat_shape)

    sample_shape = list(idx.shape)
    sample_shape[0], sample_shape[2:] = 1, [1] * (len(batch_shape) - 2) # [1, N, 1, 1]
    sample_repeat_shape = list(idx.shape)
    sample_repeat_shape[1] = 1 # [B, 1, k2, k3]
    sample_indices = torch.arange(N, dtype=torch.long).to(device).view(sample_shape).repeat(sample_repeat_shape)

    new_points = points[batch_indices, sample_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint, cuda=False, random_seed=None):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]

    FLOPs:
        S * (3 + 3 + 2)
    """
    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        return furthestsampling(xyz, npoint)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if random_seed:
        torch.manual_seed(random_seed)
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def farthest_point_sample_constructor(xyz, npoint, cuda=False, random_seed=None):
    """
        Input:
            xyz: pointcloud data, [B, N, k, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, N, npoint, 3]
    """
    if cuda:
        # cuda = False
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        return furthestsampling(xyz, npoint)
    device = xyz.device
    B, N, k, C = xyz.shape
    centroids = torch.zeros(B, N, npoint, 3, dtype=torch.float32).to(device)
    distance = torch.ones(B, N, k).to(device) * 1e10

    if random_seed:
        torch.manual_seed(random_seed)
        farthest = torch.randint(0, k, (B, N), dtype=torch.long).to(device)
    else:
        farthest = torch.randint(0, k, (B, N), dtype=torch.long).to(device)

    batch_indices = torch.arange(B, device=device).view([B, 1]).repeat([1, N])
    sample_indices = torch.arange(N, device=device).view([1, N]).repeat([B, 1])
    for i in range(npoint):
        centroids[:, :, i] = xyz[batch_indices, sample_indices, farthest, :]
        centroid = xyz[batch_indices, sample_indices, farthest, :].view(B, N, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids



def query_ball_point(radius, nsample, xyz, new_xyz, debug=False, cuda=False):
    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        if not new_xyz.is_contiguous():
            new_xyz = new_xyz.contiguous()
        return ballquery(radius, nsample, xyz, new_xyz)
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    # .sort()[0]: sorted tensor
    # .sort()[1]: sorted index
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    if debug:
        num_miss = torch.sum(mask)
        num_over = torch.sum(torch.clamp(torch.sum(sqrdists < radius ** 2, dim=2) - nsample, min=0))
        return num_miss, num_over
    return group_idx


def query_knn_point(k, xyz, new_xyz, cuda=False):
    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        if not new_xyz.is_contiguous():
            new_xyz = new_xyz.contiguous()
        return knnquery(k, xyz, new_xyz)
    dist = square_distance(new_xyz, xyz)
    group_idx = dist.sort(descending=False, dim=-1)[1][:, :, :k]
    return group_idx

def query_knn_point_4dim(k3, xyz, new_xyz, cuda=False):
    '''
    :param k3: number of neighbors
    :param xyz: [B, N, k1, 3]
    :param new_xyz: [B, N, k2, 3]
    :return: [B, N, k2, k3]
    '''

    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        if not new_xyz.is_contiguous():
            new_xyz = new_xyz.contiguous()
        return knnquery(k3, xyz, new_xyz)

    dist = square_distance_4dim(new_xyz, xyz) # [B, N, k2, k1]
    group_idx = dist.sort(descending=False, dim=-1)[1][:, :, :, :k3] # [B, N, k2, k3]
    return group_idx


def sample(nsample, feature, cuda=False):
    feature = feature.permute(0, 2, 1)  # [B, N, 3+K]
    xyz = feature[:, :, :3]

    fps_idx = farthest_point_sample(xyz, nsample, cuda=cuda)  # [B, npoint]
    torch.cuda.empty_cache()
    feature = index_points(feature, fps_idx, cuda=cuda, is_group=False)
    torch.cuda.empty_cache()
    feature = feature.permute(0, 2, 1) # [B, 3+K, npoint]

    return feature
