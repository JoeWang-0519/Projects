"""
Author: Haoxi Ran
Date: 05/10/2022
"""

from dataset.pfh import FPFH
import os
import h5py
import warnings
from torch.utils.data import Dataset
import numpy as np
import torch

warnings.filterwarnings('ignore')


class ScanObjectNNDataLoader(Dataset):
    def __init__(self, root, use_normals=False, use_pfh=False, split='training', bg=True):
        self.root = root # ./data/ScanObjectNN
        self.use_normals = use_normals
        self.use_pfh = use_pfh

        assert (split == 'training' or split == 'test')
        if bg:
            print('Use data with background points')
            dir_name = 'main_split'
        else:
            print('Use data without background points')
            dir_name = 'main_split_nobg'
        file_name = '_objectdataset_augmentedrot_scale75.h5'
        h5_name = '{}/{}/{}'.format(self.root, dir_name, split + file_name)
        with h5py.File(h5_name, mode="r") as f:
            self.data = f['data'][:].astype('float32')
            self.label = f['label'][:].astype('int64')
        print('The size of %s data is %d' % (split, self.data.shape[0]))


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        normal_root = os.path.join(self.root, 'normal_30ngb')
        pfh_root = os.path.join(self.root, 'pfh_10ngb_normal_30ngb')
        point = np.array(self.data[index]) # [2048, 3]
        
        icp_normal = FPFH(10, 30, use_normal=True)
        icp = FPFH(10, 10, use_normal=True)

        if self.use_normals:
            filename = str(index) + '.txt'
            filename_normal_root = os.path.join(normal_root, filename)
            if os.path.exists(filename_normal_root):
                normal = np.loadtxt(filename_normal_root, delimiter=',').astype(np.float32)
            else:
                normal, _ = icp_normal.calc_normals(torch.tensor(point))
                np.savetxt(filename_normal_root, normal.numpy(), delimiter=",")
            if self.use_pfh:
                filename = str(index) + '.txt'
                filename_pfh_root = os.path.join(pfh_root, filename)
                if os.path.exists(filename_pfh_root):
                    pfh = np.loadtxt(filename_pfh_root, delimiter=',').astype(np.float32)
                else:
                    point_normal = np.concatenate((point, normal), axis=1)
                    pfh = icp.feature_solver(torch.tensor(point_normal)).numpy() # [2048, 30]
                    np.savetxt(filename_pfh_root, pfh, delimiter=",")


        if self.use_normals:
            if self.use_pfh:
                # [33, 2048]
                return np.concatenate((point, pfh), axis=1).transpose(), self.label[index]
            else:
                return point.transpose(), self.label[index]
        else:
            return point.transpose(), self.label[index]
