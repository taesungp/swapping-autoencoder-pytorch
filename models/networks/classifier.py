import torch
from models.networks import BaseNetwork
from models.networks.pyramidnet import PyramidNet


class PyramidNetClassifier(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--pyramid_alpha", type=int, default=240)
        parser.add_argument("--pyramid_depth", type=int, default=200)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        assert "cifar" in opt.dataset_mode
        self.net = PyramidNet(
            opt.dataset_mode, depth=opt.pyramid_depth, alpha=opt.pyramid_alpha,
            num_classes=opt.num_classes, bottleneck=True)

        mean = torch.tensor([x / 127.5 - 1.0 for x in [125.3, 123.0, 113.9]], dtype=torch.float)
        std = torch.tensor([x / 127.5 for x in [63.0, 62.1, 66.7]], dtype=torch.float)
        self.register_buffer("mean", mean[None, :, None, None])
        self.register_buffer("std", std[None, :, None, None])

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.net(x)

