import random
import numpy as np
import os.path
from data.base_dataset import BaseDataset, get_transform
import torchvision


class CIFAR100Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=32, crop_size=32, preprocess_crop_padding=0,
                            preprocess='crop', num_classes=100, use_class_labels=True)
        opt, _ = parser.parse_known_args()
        assert opt.preprocess == 'crop' and opt.load_size == 32 and opt.crop_size == 32
        return parser

    def __init__(self, opt):
        self.opt = opt
        self.torch_dataset = torchvision.datasets.CIFAR100(
            opt.dataroot, train=opt.isTrain, download=True
        )
        self.transform = get_transform(self.opt, grayscale=False)
        self.class_list = self.create_class_list()

    def create_class_list(self):
        cache_path = os.path.join(self.opt.dataroot, "%s_classlist.npy" % self.opt.phase)
        if os.path.exists(cache_path):
            cache = np.load(cache_path)
            classlist = {i: [] for i in range(100)}
            for i, c in enumerate(cache):
                classlist[c].append(i)
            return classlist

        print("creating cache list of classes...")
        classes = np.zeros((len(self.torch_dataset)), dtype=int)
        for i in range(len(self.torch_dataset)):
            _, class_id = self.torch_dataset[i]
            classes[i] = class_id
            if i % 100 == 0:
                print("%d/%d\r" % (i, len(self.torch_dataset)), end="", flush=True)
        np.save(cache_path, classes)
        print("cache saved at %s" % cache_path)
        return self.create_class_list()

    def __getitem__(self, index):
        index = index % len(self.torch_dataset)
        image, class_id = self.torch_dataset[index]

        another_image_index = random.choice(self.class_list[class_id])
        another_image, another_class_id = self.torch_dataset[another_image_index]
        assert class_id == another_class_id
        return {"real_A": self.transform(image),
                "real_B": self.transform(another_image),
                "class_A": class_id, "class_B": class_id}

    def __len__(self):
        return len(self.torch_dataset)
        
