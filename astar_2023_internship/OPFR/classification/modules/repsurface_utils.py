"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.pointnet2_utils import farthest_point_sample_constructor, farthest_point_sample, index_points, index_points_4dim, query_knn_point, query_knn_point_4dim, query_ball_point
from modules.polar_utils import xyz2sphere
from modules.recons_utils import cal_const, cal_normal, cal_center, cal_angle, check_nan_umb, check_nan_umb_v2, check_nan_pfh, check_nan_pfh_v2


def sample_and_group(npoint, radius, nsample, center, normal, feature, return_normal=True, return_polar=False, cuda=False):
    """
    Input:
        center: input points position data
        normal: input points normal data
        feature: input points feature
    Return:
        new_center: sampled points position data
        new_normal: sampled points normal data
        new_feature: sampled points feature
    """
    # sample
    fps_idx = farthest_point_sample(center, npoint, cuda=cuda)  # [B, npoint, A]
    torch.cuda.empty_cache()
    # sample center
    new_center = index_points(center, fps_idx, cuda=cuda, is_group=False)
    torch.cuda.empty_cache()
    # sample normal
    if normal is not None:
        new_normal = index_points(normal, fps_idx, cuda=cuda, is_group=False)
        torch.cuda.empty_cache()
    else:
        new_normal = None

    # group
    idx = query_ball_point(radius, nsample, center, new_center, cuda=cuda)
    torch.cuda.empty_cache()
    # group normal
    if normal is not None:
        group_normal = index_points(normal, idx, cuda=cuda, is_group=True)  # [B, npoint, nsample, B]
        torch.cuda.empty_cache()
    # group center
    group_center = index_points(center, idx, cuda=cuda, is_group=True)  # [B, npoint, nsample, A]
    torch.cuda.empty_cache()
    group_center_norm = group_center - new_center.unsqueeze(2)
    torch.cuda.empty_cache()

    # group polar
    if return_polar:
        group_polar = xyz2sphere(group_center_norm)
        group_center_norm = torch.cat([group_center_norm, group_polar], dim=-1)
    if feature is not None:
        group_feature = index_points(feature, idx, cuda=cuda, is_group=True)
        new_feature = torch.cat([group_center_norm, group_normal, group_feature], dim=-1) if normal is not None \
            else torch.cat([group_center_norm, group_feature], dim=-1)
    else:
        new_feature = torch.cat([group_center_norm, group_normal], dim=-1) if normal is not None \
            else group_center_norm
    # new_center: [B, g, 3]
    # new_normal: [B, g, 10]
    # new_feature: [B, g, k, 6+10+c]
    return new_center, new_normal, new_feature


def sample_and_group_all(center, normal, feature, return_normal=True, return_polar=False):
    """
    Input:
        center: input centroid position data
        normal: input normal data
        feature: input feature data
    Return:
        new_center: sampled points position data
        new_normal: sampled points position data
        new_feature: sampled points data
    """
    device = center.device
    B, N, _ = center.shape
    if normal is not None:
        B, N, C = normal.shape

    new_center = torch.zeros(B, 1, 3).to(device)
    new_normal = new_center
    if normal is not None:
        group_normal = normal.view(B, 1, N, C)
    group_center = center.view(B, 1, N, 3)

    if return_polar:
        group_polar = xyz2sphere(group_center)
        group_center = torch.cat([group_center, group_polar], dim=-1)

    new_feature = torch.cat([group_center, group_normal, feature.view(B, 1, N, -1)], dim=-1) if normal is not None \
        else torch.cat([group_center, feature.view(B, 1, N, -1)], dim=-1)

    return new_center, new_normal, new_feature


def resort_points(points, idx):
    """
    Resort Set of points along G dim

    """
    device = points.device
    B, N, G, _ = points.shape

    view_shape = [B, 1, 1]
    repeat_shape = [1, N, G]
    b_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    view_shape = [1, N, 1]
    repeat_shape = [B, 1, G]
    n_indices = torch.arange(N, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[b_indices, n_indices, idx, :]

    return new_points

def resort_points_new(points, idx):
    '''
    :param points: [B，N, k2, k3-1, 3]
    :param idx: [B, N, k2, k3-1]
    :return: ordered points: [B, N, k2, k3-1, 3]
    '''
    device = points.device
    B, N, k2, k3, _ = points.shape

    b_shape = [B, 1, 1, 1]
    b_repeat_shape = [1, N, k2, k3]
    b_indices = torch.arange(B, dtype=torch.long).to(device).view(b_shape).repeat(b_repeat_shape)

    n_shape = [1, N, 1, 1]
    n_repeat_shape = [B, 1, k2, k3]
    n_indices = torch.arange(N, dtype=torch.long).to(device).view(n_shape).repeat(n_repeat_shape)

    k2_shape = [1, 1, k2, 1]
    k2_repeat_shape = [B, N, 1, k3]
    k2_indices = torch.arange(k2, dtype=torch.long).to(device).view(k2_shape).repeat(k2_repeat_shape)
    new_points = points[b_indices, n_indices, k2_indices, idx, :]
    # note: new_points [B, N, k2, k3-1, 3], the k3-1 dimension is ordered according to the
    # 'phi' in the ascending manner
    return new_points

def group_by_hierachical_pfh(xyz, new_xyz, k1=21, k2=3, k3=5, cuda=False):
    """
    Group a set of points into local reference
    xyz: [B, N, 3]
    new_xyz: [B, N, 3] (generally xyz == new_xyz)
    k1: determine (# of neighbors) for one target point we are interested in
    k2: determine (# of centroids) for one target point to characterize itself
    k3: determine (# of neighbors) for one centroid to characterize itself

    return:
    umbrella_group_xyz: [B, N, k2, k3-1, 4, 3]
    """
    # determine neighborhood (index) of target point
    B, N, _ = xyz.shape
    target_neighb_idx = query_knn_point(k1, xyz, new_xyz, cuda=cuda) # [B, N, k1]
    torch.cuda.empty_cache()
    # attain neighborhood coordinates of target point
    group_xyz = index_points(xyz, target_neighb_idx, cuda=cuda, is_group=True)[:, :, 1:]  # [B, N, k1-1, 3]
    torch.cuda.empty_cache()
    # determine neighborhood centroid coordinate of target point
    ### Note: at this moment, I am not sure whether need to fix the randomness here.
    group_xyz_centroid = farthest_point_sample_constructor(group_xyz, k2, cuda=False) # [B, N, k2, 3]
    torch.cuda.empty_cache()
    # determine neighborhood of each centroid
    #centroid_neighb_index = query_knn_point_4dim(k3, group_xyz, group_xyz_centroid, cuda=False) # [B, N, k2, k3]
    centroid_neighb_index = query_knn_point_4dim(k3, xyz.reshape(B, 1, N, -1).repeat([1, N, 1, 1]), group_xyz_centroid, cuda=False)  # [B, N, k2, k3]
    # torch.cuda.empty_cache()
    # attain coordinates
    # group_xyz_centorid_ngb: [B，N, k2, k3-1, 3]
    group_xyz_centorid_ngb = index_points_4dim(xyz.reshape(B,1,N,-1).repeat([1, N, 1, 1]), centroid_neighb_index, cuda=False, is_group=True)[:, :, :, 1:]
    # torch.cuda.empty_cache()
    # [B，N, k2, k3-1, 3]
    group_xyz_norm = group_xyz_centorid_ngb - group_xyz_centroid.unsqueeze(-2)
     
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [B, N, k2, k3-1]
    # ascending by default
    # sort_idx[B,N,k2,i]: the i-th smallest entry index with respect to group_phi [B,N,k2,k3-1]
    # that is, with respect to group_xyz_norm [B,N,k2,k3-1,3]
    sort_idx = group_phi.argsort(dim=-1)  # [B, N, k2, k3-1]

    # [B, N, k2, k3-1, 1, 3]
    # the order is: [smallest, ..., largest] with respect to phi (in -3 dimension)
    sorted_group_xyz = resort_points_new(group_xyz_norm, sort_idx).unsqueeze(-2)

    # [B, N, k2, k3-1, 1, 3]
    # the order turns to [2-nd smallest, ..., largest, smallest] with respect to phi
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    sorted_group_xyz_roll2 = torch.roll(sorted_group_xyz, -2, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    # [B, N, k2, k3-1, 4, 3]
    # there are k3-1 triangles for each B, N, k2
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll, sorted_group_xyz_roll2], dim=-2)

    return umbrella_group_xyz


# modified for pfh
def group_by_hierarchical_umbrella(xyz, new_xyz, k1=21, k2=3, k3=5, cuda=False):
    """
    Group a set of points into local reference
    xyz: [B, N, 3]
    new_xyz: [B, N, 3] (generally xyz == neew_xyz)
    k1: determine (# of neighbors) for one target point we are interested in
    k2: determine (# of centroids) for one target point to characterize itself
    k3: determine (# of neighbors) for one centroid to characterize itself

    """
    B, N, _ = xyz.shape
    # determine neighborhood (index) of target point
    target_neighb_idx = query_knn_point(k1, xyz, new_xyz, cuda=cuda) # [B, N, k1]
    torch.cuda.empty_cache()
    # attain neighborhood coordinates of target point
    group_xyz = index_points(xyz, target_neighb_idx, cuda=cuda, is_group=True)[:, :, 1:]  # [B, N, k1-1, 3]
    torch.cuda.empty_cache()
    # determine neighborhood centroid coordinate of target point
    ### Note: at this moment, I am not sure whether need to fix the randomness here.
    group_xyz_centroid = farthest_point_sample_constructor(group_xyz, k2, cuda=False) # [B, N, k2, 3]
    torch.cuda.empty_cache()
    # determine neighborhood of each centroid
    # centroid_neighb_index = query_knn_point_4dim(k3, group_xyz, group_xyz_centroid, cuda=False) # [B, N, k2, k3]
    centroid_neighb_index = query_knn_point_4dim(k3, xyz.reshape(B, 1, N, -1).repeat([1, N, 1, 1]), group_xyz_centroid, cuda=False)  # [B, N, k2, k3]

    torch.cuda.empty_cache()
    # attain coordinates
    # group_xyz_centorid_ngb: [B，N, k2, k3-1, 3]
    group_xyz_centorid_ngb = index_points_4dim(xyz.reshape(B,1,N,-1).repeat([1, N, 1, 1]), centroid_neighb_index, cuda=False, is_group=True)[:, :, :, 1:]
    torch.cuda.empty_cache()

    # [B，N, k2, k3-1, 3]
    group_xyz_norm = group_xyz_centorid_ngb - group_xyz_centroid.unsqueeze(-2)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [B, N, k2, k3-1]
    # ascending by default
    # sort_idx[B,N,k2,i]: the i-th smallest entry index with respect to group_phi [B,N,k2,k3-1]
    # that is, with respect to group_xyz_norm [B,N,k2,k3-1,3]
    sort_idx = group_phi.argsort(dim=-1)  # [B, N, k2, k3-1]

    # [B, N, k2, k3-1, 1, 3]
    # the order is: [smallest, ..., largest] with respect to phi (in -3 dimension)
    sorted_group_xyz = resort_points_new(group_xyz_norm, sort_idx).unsqueeze(-2)

    # [B, N, k2, k3-1, 1, 3]
    # the order turns to [2-nd smallest, ..., largest, smallest] with respect to phi
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    # [B, N, k2, k3-1, 3, 3]
    # there are k3-1 triangles for each B, N, k2
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz

def group_by_pfh(xyz, new_xyz, k=9, cuda=False):
    """
    Group a set of points into umbrella surfaces

    """
    idx = query_knn_point(k, xyz, new_xyz, cuda=cuda)
    torch.cuda.empty_cache()
    group_xyz = index_points(xyz, idx, cuda=cuda, is_group=True)[:, :, 1:]  # [B, N', K-1, 3]
    torch.cuda.empty_cache()

    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [B, N', K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [B, N', K-1]

    # [B, N', K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    sorted_group_xyz_roll2 = torch.roll(sorted_group_xyz, -2, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    # [B, N'. K-1 ,4, 3]
    pfh_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll, sorted_group_xyz_roll2], dim=-2)

    return pfh_group_xyz


def group_by_umbrella(xyz, new_xyz, k=9, cuda=False):
    """
    Group a set of points into umbrella surfaces

    """
    idx = query_knn_point(k, xyz, new_xyz, cuda=cuda)
    torch.cuda.empty_cache()
    group_xyz = index_points(xyz, idx, cuda=cuda, is_group=True)[:, :, 1:]  # [B, N', K-1, 3]
    torch.cuda.empty_cache()

    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [B, N', K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [B, N', K-1]

    # [B, N', K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz


class SurfaceAbstraction(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, return_polar=True, return_normal=True, cuda=False):
        super(SurfaceAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.cuda = cuda
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, center, normal, feature):
        normal = normal.permute(0, 2, 1)
        center = center.permute(0, 2, 1)
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        if self.group_all:
            new_center, new_normal, new_feature = sample_and_group_all(center, normal, feature,
                                                                       return_polar=self.return_polar,
                                                                       return_normal=self.return_normal)
        else:
            new_center, new_normal, new_feature = sample_and_group(self.npoint, self.radius, self.nsample, center,
                                                                   normal, feature, return_polar=self.return_polar,
                                                                   return_normal=self.return_normal, cuda=self.cuda)

        new_feature = new_feature.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        new_center = new_center.permute(0, 2, 1)
        new_normal = new_normal.permute(0, 2, 1)

        return new_center, new_normal, new_feature


class SurfaceAbstractionCD(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, npoint, radius, nsample, feat_channel, pos_channel, mlp, group_all,
                 return_normal=True, return_polar=False, cuda=False):
        super(SurfaceAbstractionCD, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.cuda = cuda
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_channel = pos_channel
        self.group_all = group_all

        self.mlp_l0 = nn.Conv2d(self.pos_channel, mlp[0], 1)
        self.mlp_f0 = nn.Conv2d(feat_channel, mlp[0], 1)
        self.bn_l0 = nn.BatchNorm2d(mlp[0])
        self.bn_f0 = nn.BatchNorm2d(mlp[0])

        # mlp_l0+mlp_f0 can be considered as the first layer of mlp_convs
        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, center, normal, feature):
        center = center.permute(0, 2, 1)
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        if normal is not None:
            normal = normal.permute(0, 2, 1)

        if self.group_all:
            new_center, new_normal, new_feature = sample_and_group_all(center, normal, feature,
                                                                       return_normal=self.return_normal,
                                                                       return_polar=self.return_polar)
        else:
            new_center, new_normal, new_feature = sample_and_group(self.npoint, self.radius, self.nsample, center,
                                                                   normal, feature, return_normal=self.return_normal,
                                                                   return_polar=self.return_polar, cuda=self.cuda)

        new_feature = new_feature.permute(0, 3, 2, 1)

        # init layer
        _, C, _, _ = new_feature.shape
        if C == self.pos_channel:
            loc = self.bn_l0(self.mlp_l0(new_feature[:, :self.pos_channel]))
            new_feature = loc
        else:    
            loc = self.bn_l0(self.mlp_l0(new_feature[:, :self.pos_channel]))
            feat = self.bn_f0(self.mlp_f0(new_feature[:, self.pos_channel:]))
            new_feature = loc + feat
        
        new_feature = F.relu(new_feature)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        new_center = new_center.permute(0, 2, 1)
        if new_normal is not None:
            new_normal = new_normal.permute(0, 2, 1)

        return new_center, new_normal, new_feature


class UmbrellaSurfaceConstructor(nn.Module):
    """
    Umbrella-based Surface Abstraction Module

    """

    def __init__(self, k, in_channel, aggr_type='sum', return_dist=False, random_inv=True, cuda=False):
        super(UmbrellaSurfaceConstructor, self).__init__()
        self.k = k
        self.return_dist = return_dist
        self.random_inv = random_inv
        self.aggr_type = aggr_type
        self.cuda = cuda

        self.mlps = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
        )

    def forward(self, center):
        center = center.permute(0, 2, 1)
        # surface construction
        group_xyz = group_by_umbrella(center, center, k=self.k, cuda=self.cuda)  # [B, N, K-1, 3 (points), 3 (coord.)]

        # normal
        group_normal = cal_normal(group_xyz, random_inv=self.random_inv, is_group=True)
        # coordinate
        group_center = cal_center(group_xyz)
        # polar
        group_polar = xyz2sphere(group_center)
        if self.return_dist:
            group_pos = cal_const(group_normal, group_center)
            group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)
            new_feature = torch.cat([group_center, group_polar, group_normal, group_pos], dim=-1)  # N+P+CP: 10
        else:
            group_normal, group_center = check_nan_umb(group_normal, group_center)
            new_feature = torch.cat([group_center, group_polar, group_normal], dim=-1)
        new_feature = new_feature.permute(0, 3, 2, 1)  # [B, C, G, N]

        # mapping
        new_feature = self.mlps(new_feature)

        # aggregation
        if self.aggr_type == 'max':
            new_feature = torch.max(new_feature, 2)[0]
        elif self.aggr_type == 'avg':
            new_feature = torch.mean(new_feature, 2)
        else:
            new_feature = torch.sum(new_feature, 2)

        return new_feature


class UmbrellaSurfaceConstructorV2(nn.Module):
    """
    PFH-based Surface Abstraction Module
    """

    def __init__(self, k1, k2, k3, in_channel, aggr_type1='max', aggr_type2='avg', return_dist=False, random_inv=True, cuda=False):
        super(UmbrellaSurfaceConstructorV2, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.return_dist = return_dist
        self.random_inv = random_inv
        self.aggr_type1 = aggr_type1
        self.aggr_type2 = aggr_type2
        self.cuda = cuda

        self.mlps = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
        )

    def forward(self, center):
        # "center: [B, 3, N]"
        # [B, N, 3]
        center = center.permute(0, 2, 1)
        # surface construction
        # [B, N, k2, k3-1, 3, 3]
        group_xyz = group_by_hierarchical_umbrella(center, center, k1=self.k1, k2=self.k2, k3=self.k3, cuda=self.cuda)
        B, N, k2, k3, _, _ = group_xyz.shape
        # [B, N, k2, k3-1, 3]
        group_normal = cal_normal(group_xyz, normal_shape=5, random_inv=self.random_inv, is_group=True)
        # [B, N, k2, k3-1, 3]
        group_center = cal_center(group_xyz)
        # [B, N, k2, k3-1, 3], rho, phi and theta
        group_polar = xyz2sphere(group_center)
        if self.return_dist:
            group_pos = cal_const(group_normal, group_center)
            group_normal, group_center, group_pos = check_nan_umb_v2(group_normal, group_center, group_pos)
            # [B, N, k2, k3-1, 10]
            new_feature = torch.cat([group_center, group_polar, group_normal, group_pos], dim=-1)  # N+P+CP: 10
        else:
            group_normal, group_center = check_nan_umb_v2(group_normal, group_center)
            new_feature = torch.cat([group_center, group_polar, group_normal], dim=-1)
        # [B, 10, k3-1, k2*N]
        new_feature = new_feature.permute(0, 4, 3, 2, 1).reshape([B, 10, k3, -1])

        # mapping
        # [B, 10, k3-1, k2*N]
        new_feature = self.mlps(new_feature)
        # [B, 10, k3-1, k2, N]
        new_feature = new_feature.reshape([B, 10, k3, k2, N])

        # aggregation over neighborhoods [B, 10, k2, N]
        if self.aggr_type1 == 'max':
            new_feature = torch.max(new_feature, 2)[0]
        elif self.aggr_type1 == 'avg':
            new_feature = torch.mean(new_feature, 2)
        else:
            new_feature = torch.sum(new_feature, 2)

        # aggregation over centroids [B, 10, N]
        if self.aggr_type2 == 'max':
            new_feature = torch.max(new_feature, 2)[0]
        elif self.aggr_type2 == 'avg':
            new_feature = torch.mean(new_feature, 2)
        else:
            new_feature = torch.sum(new_feature, 2)

        return new_feature


class PFHSurfaceConstructor(nn.Module):
    """
    Umbrella-based Surface Abstraction Module

    """
    def __init__(self, k, in_channel, aggr_type='sum', return_dist=False, random_inv=True, cuda=False):
        super(PFHSurfaceConstructor, self).__init__()
        self.k = k
        self.return_dist = return_dist
        self.random_inv = random_inv
        self.aggr_type = aggr_type
        self.cuda = cuda

        self.mlps = nn.Sequential(
            nn.Conv2d(in_channel, 64, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 64, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, in_channel, 1, bias=True),
        )

    def forward(self, center):
        center = center.permute(0, 2, 1)
        # surface construction
        group_xyz = group_by_pfh(center, center, k=self.k, cuda=self.cuda)  # [B, N, K-1, 4 (points), 3 (coord.)]

        # normal
        group_normal = cal_normal(group_xyz, pfh=True, normal_shape=4, random_inv=self.random_inv, is_group=True)
        # coordinate
        group_center = cal_center(group_xyz)
        # polar
        group_polar = xyz2sphere(group_center)
        # angle
        group_angle = cal_angle(group_xyz, group_normal)

        if self.return_dist:
            group_pos = cal_const(group_normal, group_center)
            group_normal, group_center, group_angle, group_pos = check_nan_pfh(group_normal, group_center, group_angle, group_pos)
            new_feature = torch.cat([group_center, group_polar, group_normal, group_angle, group_pos], dim=-1)  # 14
            ### now repsurf channel = 30
            # new_feature = group_angle
        else:
            group_normal, group_center, group_angle = check_nan_pfh(group_normal, group_center, group_angle)
            new_feature = torch.cat([group_center, group_polar, group_normal, group_angle], dim=-1)
        new_feature = new_feature.permute(0, 3, 2, 1)  # [B, C, G, N]

        # mapping
        new_feature = self.mlps(new_feature)

        # aggregation
        if self.aggr_type == 'max':
            new_feature = torch.max(new_feature, 2)[0]
        elif self.aggr_type == 'avg':
            new_feature = torch.mean(new_feature, 2)
        else:
            new_feature = torch.sum(new_feature, 2)

        return new_feature



class PFHSurfaceConstructorV2(nn.Module):
    """
    PFH-based Surface Abstraction Module
    """

    def __init__(self, k1, k2, k3, in_channel, aggr_type1='max', aggr_type2='avg', return_dist=False, random_inv=True, cuda=False):
        super(PFHSurfaceConstructorV2, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.return_dist = return_dist
        self.random_inv = random_inv
        self.aggr_type1 = aggr_type1
        self.aggr_type2 = aggr_type2
        self.cuda = cuda

        self.mlps = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
        )

    def forward(self, center):
        # "center: [B, 3, N]"
        # [B, N, 3]
        center = center.permute(0, 2, 1)
        # surface construction
        # [B, N, k2, k3-1, 4, 3]
        group_xyz = group_by_hierachical_pfh(center, center, k1=self.k1, k2=self.k2, k3=self.k3, cuda=self.cuda)
        B, N, k2, k3, _, _ = group_xyz.shape
        # [B, N, k2, k3-1, 3]
        group_normal = cal_normal(group_xyz, pfh=True, normal_shape=5, random_inv=self.random_inv, is_group=True)
        # [B, N, k2, k3-1, 3]
        group_center = cal_center(group_xyz)
        # [B, N, k2, k3-1, 3], rho, phi and theta
        group_polar = xyz2sphere(group_center)
        # [B, N, k2, k3-1, 4]
        group_angle = cal_angle(group_xyz, group_normal)
        if self.return_dist:
            group_pos = cal_const(group_normal, group_center)
            group_normal, group_center, group_angle, group_pos = check_nan_pfh_v2(group_normal, group_center, group_angle, group_pos)
            # [B, N, k2, k3-1, 14]
            new_feature = torch.cat([group_center, group_polar, group_normal, group_angle, group_pos], dim=-1)  # N+P+CP: 10
        else:
            group_normal, group_center, group_angle = check_nan_pfh_v2(group_normal, group_center, group_angle)
            new_feature = torch.cat([group_center, group_polar, group_normal, group_angle], dim=-1)
        # [B, 14, k3-1, k2*N]
        new_feature = new_feature.permute(0, 4, 3, 2, 1).reshape([B, 14, k3, -1])

        # mapping
        # [B, 14, k3-1, k2*N]
        new_feature = self.mlps(new_feature)
        # [B, 14, k3-1, k2, N]
        new_feature = new_feature.reshape([B, 14, k3, k2, N])

        # aggregation over neighborhoods [B, 10, k2, N]
        if self.aggr_type1 == 'max':
            new_feature = torch.max(new_feature, 2)[0]
        elif self.aggr_type1 == 'avg':
            new_feature = torch.mean(new_feature, 2)
        else:
            new_feature = torch.sum(new_feature, 2)

        # aggregation over centroids [B, 10, N]
        if self.aggr_type2 == 'max':
            new_feature = torch.max(new_feature, 2)[0]
        elif self.aggr_type2 == 'avg':
            new_feature = torch.mean(new_feature, 2)
        else:
            new_feature = torch.sum(new_feature, 2)

        return new_feature
