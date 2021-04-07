import random
import os.path
from data.base_dataset import BaseDataset
from data.lmdb_dataset import LMDBDataset
import util


class UnalignedLMDBDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.dataset_A = LMDBDataset(util.copyconf(opt, dataroot=self.dir_A))
        self.dataset_B = LMDBDataset(util.copyconf(opt, dataroot=self.dir_B))
        self.B_indices = list(range(len(self.dataset_B)))


    def __len__(self):
        return max(len(self.dataset_A), len(self.dataset_B))

    def __getitem__(self, index):
        if index == 0 and self.opt.isTrain:
            random.shuffle(self.B_indices)

        result = self.dataset_A.__getitem__(index % len(self.dataset_A))
        B_index = self.B_indices[index % len(self.dataset_B)]
        B_result = self.dataset_B.__getitem__(B_index)
        result["real_B"] = B_result["real_A"]
        result["path_B"] = B_result["path_A"]
        return result
