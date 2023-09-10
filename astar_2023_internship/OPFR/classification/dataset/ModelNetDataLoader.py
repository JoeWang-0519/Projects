'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import torch

# here we do not need PFH
from dataset.pfh import FPFH

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def normal_normalization(normal):
    centroid = np.mean(normal, axis = 0)
    normal  = normal - centroid
    std = np.sqrt(np.mean(normal**2,axis = 0))
    normal = normal / std
    return normal


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, pfh_source, normal_source, args, split='train', process_data=False):
        # data path
        self.root = root
        self.pfh_source = pfh_source
        self.normal_source = normal_source

        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.num_category = args.num_category

        # load more features
        self.use_pfh = args.use_pfh
        self.use_normals = args.use_normals
        self.estimate_normals = args.estimate_normals

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip()for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        ## shape_ids: record the tag like 'bathtub_004'
        ## shape_names: 'bathtub'
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        ## based on the shape_names, recover the direct path of each image
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt',
                          os.path.join(self.pfh_source, shape_ids[split][i]) + '.txt', os.path.join(self.normal_source, shape_ids[split][i]) + '.txt') for i in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[fn[0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set[:, 3:6] = normal_normalization(point_set[:, 3:6])


        icp_normal = FPFH(10, 30, use_normal=True)
        icp_pfh = FPFH(10, 18, use_normal=True)
        
        if not self.use_normals:
            # sample 4000 points
            point_set = point_set[:4000, 0:3]
        else: 
            if self.estimate_normals:
                normal_filename = fn[3][0:-4] + '_normal' + '.txt'
                point_set = point_set[:, :3]
                if os.path.exists(normal_filename):
                    normal = np.loadtxt(normal_filename, delimiter=',').astype(np.float32)
                else:
                    normal, _ = icp_normal.calc_normals(torch.tensor(point_set))
                    normal = normal.numpy()
                    # print(normal.dtype)
                    np.savetxt(normal_filename, normal, delimiter=',')
                point_set = np.concatenate((point_set, normal), axis = 1)
                point_set = point_set[:4000, :]
            else:
                point_set = point_set[:4000, :]

        if self.use_pfh and self.use_normals:
            # fn[2] : root/modelnet40_pfh/xxxx_xxxx.txt
            pfh_filename = fn[2][0:-4] + '_pfh' + '.txt'
            if os.path.exists(pfh_filename):
                point_set_pfh = np.loadtxt(pfh_filename, delimiter=',').astype(np.float32)
            else:
                # N * 30 numpy
                point_set_pfh = icp_pfh.feature_solver(torch.tensor(point_set)).numpy()
                # print(point_set_pfh.dtype)
                np.savetxt(pfh_filename, point_set_pfh, delimiter=",")
            point_set = np.concatenate((point_set[:, 0:3], point_set_pfh), axis = 1)

        return point_set, label[0]

    # encapsulate the '__getitem__' function in '_get_item'
    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
