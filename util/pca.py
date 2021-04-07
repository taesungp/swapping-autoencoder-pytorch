import time
import numpy as np
import torch
import util

np.set_printoptions(precision=4, suppress=True, edgeitems=10)


class PCA:
    def __init__(self, X, ndim=128, var_fraction=0.99, l2_normalized=True, first_direction=None):

        self.l2_normalized = l2_normalized
        if l2_normalized:
            X = X[:, :-1]

        assert len(X.shape) == 2
        torch.cuda.synchronize()
        start_time = time.time()

        self.mean = torch.mean(X, dim=0, keepdim=True)
        #self.mean = 0
        #self.std = torch.std(X, dim=0, keepdim=True) + 1e-6
        self.std = 1
        #print("std is ", self.std[:, :10].cpu().numpy())
        #X_orig = X
        X = (X - self.mean) / self.std

        U, S, V = torch.svd(X)
        S = S[:ndim]
        V = V[:, :ndim]
        self.proj = V
        scale = torch.mm(X, self.proj).std(dim=0)
        torch.cuda.synchronize()
        print("PCA time taken on vectors of size %s : %f" % (str(X.size()), time.time() - start_time))
        print("largest std of each PC: ", scale[:10].cpu().numpy())
        print("smallest std of each PC: ", scale[-10:].cpu().numpy())
        self.sinvals = S
        print("largest sinvals: ", self.sinvals[:10].cpu().numpy())
        self.inv_proj = V.transpose(0, 1)
        self.N = X.size(0)

    def project(self, x):
        if self.l2_normalized:
            last_dim = x[:, -1:]
            x = x[:, :-1]
        #x = (x - self.mean) / self.std
        z = torch.mm(x, self.proj)
        if self.l2_normalized:
            return torch.cat([z, last_dim], dim=1)
        else:
            return z

    def scale(self):
        return self.sinvals / np.sqrt(self.N)

    def pc(self, idx):
        # return self.inv_proj[idx:idx + 1] * (self.std * np.sqrt(self.inv_proj.size(1)))
        return self.inv_proj[idx:idx + 1]

    def inverse(self, z):
        if self.l2_normalized:
            last_dim = z[:, -1:]
            z = z[:, :-1]
        x = torch.mm(z, self.inv_proj)
        #x = x * self.std + self.mean
        if self.l2_normalized:
            return torch.cat([x, last_dim], dim=1)
        else:
            return x
