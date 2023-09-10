'''
Contains a generic algorithm object which can do vanilla ICP

Then, create plugin functions which can do naive, simple, and fast PFH_raw_version
'''

# modified version
import math
import torch
import time
import numpy as np
#import matplotlib.pyplot as plt
#import scipy.special as sp

#from . import utils

class PFH(object):

    """Parent class for PFH_raw_version"""

    def __init__(self, div, nneighbors, use_normal = False):
        """Pass in parameters """
        #self._e = e
        self._div = div
        self._nneighbors = nneighbors
        #self._radius = rad
        # we can choice use normal or not
        self.use_normal = use_normal


    def feature_solver(self, P):
        """Main solver
        :P: point cloud
        # note, P is n * (3 + C) tensor
        :div: Number of divisions for binning PFH_raw_version
        :nneighbors: Number of k neighbors for surface normal estimation and PFH_raw_version neighbors
        :returns: feature (n * div**3 numpy)

        """
        k = self._nneighbors

        if self.use_normal == False:
            normP, indP = self.calc_normals(P)
            histP = self.calcHistArray(P, normP, indP)
        else:
            P_xyz = P[:, :3]
            normP = P[:, 3:]

            # we still need the neighbour index

            N = P.shape[0]
            ind_of_neighbors = torch.zeros((N, k), dtype=torch.int64)

            for i in range(N):
                indN = self.getNeighbors(P_xyz[i, :], P_xyz)[1]
                ind_of_neighbors[i, :] = indN

            # hist P: [N, div*3]
            histP = self.calcHistArray(P_xyz, normP, ind_of_neighbors)

        return histP

    def getNeighbors(self, pq, pc):
        """Get k nearest neighbors of the query point pq from pc, within the radius

        :pq: query point, which is of size 3 tensor
        :pc: total point cloud data, which is of size N * 3 tensor
        :returns: distance and corresponding index

        """
        k = self._nneighbors
        N = pc.shape[0]
        dists = torch.norm(pq - pc, dim=1)
        #mask = (dists <= self._radius).nonzero().squeeze()
        values, idxs = torch.topk(dists, k=k + 1, largest=False)
        return values[1:], idxs[1:]

    def calc_normals(self, pc):
        """calculate normal vector for point cloud pc

        :pc: tensor of size N * 3
        :returns: normal vector N * 3, indexs of neighbours N * k

        """

        k = self._nneighbors
        N = pc.shape[0]
        normals = torch.zeros((N, 3), dtype=torch.float32)
        ind_of_neighbors = torch.zeros((N, k), dtype=torch.int16)

        for i in range(N):

            indN = self.getNeighbors(pc[i, :], pc)[1]
            ind_of_neighbors[i,:] = indN

            # PCA
            # [k, 3]
            X = pc[indN, :]
            X = X - torch.mean(X, dim=0, keepdim=True)
            
            # cov = torch.matmul(X, X.T) / k
            # previously use torch.svd(), incorrect
            _, _, V = torch.svd(X)
            Vt = V.permute(1, 0)
            normal = Vt[2, :]

            # Re-orient normal vectors
            if torch.matmul(normal, -1. * pc[i, :]) < 0:
                normal = -1. * normal
            normals[i,:] = normal

        return normals, ind_of_neighbors

    def calcHistArray(self, pc, norm, indNeigh):
        """override this function with custom Histogram"""
        ## input:
        # pc: N * 3 tensor
        # norm: N * 3 tensor
        # indNeigh : N * k tensor
        ## output:
        # histArray : N * div**3 tensor
        # calculate PFH_raw_version feature representation
        print("\tCalculating histograms naive method \n")

        k = self._nneighbors
        N = pc.shape[0]
        histArray = torch.zeros((N, self._div**3))
        s = self.calc_thresholds()

        for i in range(N):
            u = norm[i]
            n = k + 1
            # may need to change type
            # N_features = sp.comb(n, 2)

            p_list = torch.cat((torch.tensor([i]), indNeigh[i]), dim = 0)
            # total_len = k + 1
            # N_features = total_pair
            total_len = p_list.shape[0]
            total_pair = total_len * (total_len - 1) / 2
            features = torch.zeros((total_pair, 3))
            index = 0
            for idx_z in range(0, total_len - 1):
                z = p_list[idx_z]
                for idx_p in range(idx_z + 1, total_len):
                    p = p_list[idx_p]
                    pi = pc[p, :]
                    pj = pc[z, :]
                    if torch.arccos(torch.dot(norm[p], pj - pi)) <= torch.arccos(torch.dot(norm[z], pi - pj)):
                        ps = pi
                        pt = pj
                        ns = norm[p]
                        nt = norm[z]
                    else:
                        ps = pj
                        pt = pi
                        ns = norm[z]
                        nt = norm[p]

                    u = ns
                    difV = pt - ps
                    dist = torch.norm(difV)
                    difV = difV / dist
                    v = torch.cross(difV, u)
                    w = torch.cross(u, v)

                    # features we want for each point
                    alpha = torch.dot(v, nt)
                    phi = torch.dot(u, difV)
                    theta = torch.atan(torch.dot(w, nt) / torch.dot(u, nt))

                    features[index, 0], features[index, 1], features[index, 2] = alpha, phi, theta
                    index = index + 1

            # features for one point cloud data
            pfh_hist, bin_edges = self.calc_pfh_hist(features, s)
            histArray[i, :] = pfh_hist / (total_pair)
        return histArray
    '''
    def step(self, si, fi):
        """Helper function for calc_pfh_hist. Depends on selection of div

        :si: TODO
        :fi: TODO
        :returns: TODO

        """
        # si size: self._div
        result = 0
        if fi < si[0]:
            result = 0
        else:
            for i in range(0, self._div-1):
                start, end = si[i] , si[i+1]
                if fi >= start and fi < end:
                    result = i+1
                    break
        # result belongs to [0, 1, ..., self._div-1]
        return result

    def calc_thresholds(self):
        """
        :returns: 3 x (div) torch where each row is a feature's thresholds
        """
        delta = (torch.pi) / self._div
        s1 = torch.tensor([0. + i * delta for i in range(1, self._div+1)], dtype=torch.float32)

        delta = (torch.pi) / self._div
        s3 = torch.tensor([0. + i * delta for i in range(1, self._div+1)], dtype=torch.float32)

        delta = (torch.pi) / self._div
        s4 = torch.tensor([- torch.pi / 2 + i * delta for i in range(1, self._div+1)], dtype=torch.float32)

        s = torch.stack([s1, s3, s4])
        return s
    '''
    def calc_pfh_hist(self, f):
        """Calculate histogram and bin edges.
        # no need of s!!
        :f: feature vector of f1,f3,f4 (Nx3)
        :returns:
            pfh_hist - array of length 3 * div, represents number of samples per bin
            bin_edges - range(0, 1, 2, ..., div; div, ..., 2*div; 2*div, ..., 3*div)
        """
        # preallocate array sizes, create bin_edges
        pfh_hist = torch.zeros(self._div * 3)

        bin_edges = torch.arange(0, self._div * 3 + 1)

        # find the division thresholds for the histogram
        # s is of size 3 * (div)
        # s = self.calc_thresholds()


        # Loop for every row in f from 0 to N
        #for j in range(0, f.shape[0]):
            # calculate the bin index to increment
        #    for i in range(0,3):
        #        starting_index = i * self._div
        #        index = starting_index + self.step(s[i, :], f[j, i])
        #        pfh_hist[index] += 1
            # Increment histogram at that index


        for i in range(3):
            feature_i = f[:,i]
            if i==2:
                min, max = -torch.tensor(math.pi)/2, torch.tensor(math.pi)/2
            else:
                min, max = 0, torch.tensor(math.pi)
            starting_index, ending_index = i * self._div, (i+1) * self._div
            pfh_hist[starting_index:ending_index] = torch.histc(feature_i, min=min, max=max, bins=self._div)

        return pfh_hist, bin_edges

class FPFH(PFH):

    """Child class of PFH_raw_version to implement a different calcHistArray"""

    def calcHistArray(self, pc, norm, indNeigh):
        """Overriding base PFH_raw_version to FPFH"""

        # print("\tCalculating histograms fast method \n")
        k = self._nneighbors
        N = pc.shape[0]
        histArray = torch.zeros((N, 3 * self._div))
        #fast_histArray = torch.zeros((N, 3 * self._div))
        # for specific point cloud data, its distance with each neighborhood
        #distArray = torch.zeros(k)
        # for total point cloud data, N * k distance tansor
        #distList = torch.zeros((N, k))
        #s = self.calc_thresholds()
        for i in range(N):
            default = norm[i]
            features = torch.zeros((k, 3))
            for j in range(k):
                pi = pc[i, :]
                pj = pc[indNeigh[i,j], :]
                dif = torch.linalg.norm(pi - pj)
                # normalization first
                if torch.dot(default, (pj - pi)/dif) >= torch.dot(norm[indNeigh[i,j], :], (pi - pj)/dif):
                    ps = pi
                    pt = pj
                    ns = norm[i]
                    nt = norm[indNeigh[i,j]]
                else:
                    ps = pj
                    pt = pi
                    ns = norm[indNeigh[i,j]]
                    nt = norm[i]

                u = ns

                difV = pt - ps
                dist = torch.linalg.norm(difV)
                difV = difV / dist
                v = torch.cross(u, difV)
                v = v / torch.linalg.norm(v)

                w = torch.cross(u, v)
                w = w / torch.linalg.norm(w)

                alpha = torch.dot(v, nt) # [0, pi]
                phi = torch.dot(u, difV) # [0, pi]
                theta = torch.dot(w, nt) / torch.dot(u, nt) #[-pi/2, pi/2]

                features[j, 0], features[j, 1], features[j, 2] = alpha, phi, theta
                #distArray[j] = dist

            #distList[i, :] = distArray
            features[:,:2] = torch.arccos(features[:,:2])
            features[:,2] = torch.arctan((features[:,2]))
            pfh_hist, _ = self.calc_pfh_hist(features)
            #pfh_hist, _ = self.calc_pfh_hist(features, s)
            histArray[i, :] = pfh_hist / k
        # histArray is SPFH feature
        '''
        for i in range(N):
            spfh_sum = torch.zeros(self._div * 3)
            for j in range(k):
                ## spfh_sum is div*3 tensor
                neighbor_idx = indNeigh[i,j]
                # might be modifed
                spfh_sum += histArray[neighbor_idx, :]
            fast_histArray[i, :] = (histArray[i, :] + (1/k) * spfh_sum)/2
        '''
        #total: [N, N, 3 * self._div]
        total = histArray.unsqueeze(0).repeat(N, 1, 1)
        new_idx = indNeigh.unsqueeze(-1).expand(-1, -1, 3 * self._div)
        neighbor_pfh = torch.gather(total, dim=1, index=new_idx) # [N, k, 3 * self._div]
        spfh_avg = torch.mean(neighbor_pfh, dim=1)
        fast_histArray2 = (histArray + spfh_avg) / 2


        return fast_histArray2
