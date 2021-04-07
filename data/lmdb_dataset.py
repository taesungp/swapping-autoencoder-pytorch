import random
import sys
import os.path
from PIL import Image
from data.base_dataset import BaseDataset, get_transform
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision.transforms as transforms


class LMDBDataset(BaseDataset):
    def __init__(self, opt):
        import lmdb
        self.opt = opt
        write_cache = True
        root = opt.dataroot
        self.root = os.path.expanduser(root)
        self.env = lmdb.open(root, readonly=True, lock=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        print('lmdb file at %s opened.' % root)
        cache_file = os.path.join(root, '_cache_')
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        elif write_cache:
            print('generating keys')
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))
            print('cache file generated at %s' % cache_file)
        else:
            self.keys = []

        random.Random(0).shuffle(self.keys)

        self.transform = get_transform(self.opt, grayscale=False)
        if "lsun" in self.opt.dataroot.lower():
            print("Seems like a LSUN dataset, so we will apply BGR->RGB conversion")


    def __getitem__(self, index):
        path = self.keys[index]
        return self.getitem_by_path(path)

    def getitem_by_path(self, path):
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(path)
        try:
            img = cv2.imdecode(
                np.fromstring(imgbuf, dtype=np.uint8), 1)
        except cv2.error as e:
            print(path, e)
            return self.__getitem__(random.randint(0, self.length - 1))
        if "lsun" in self.opt.dataroot.lower():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        return {"real_A": self.transform(img), "path_A": path.decode("utf-8")}

    def set_phase(self, phase):
        super().set_phase(phase)
        pass

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'
