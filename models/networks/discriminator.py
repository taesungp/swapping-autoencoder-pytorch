from models.networks import BaseNetwork
from models.networks.stylegan2_layers import Discriminator as OriginalStyleGAN2Discriminator


class StyleGAN2Discriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--netD_scale_capacity", default=1.0, type=float)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.stylegan2_D = OriginalStyleGAN2Discriminator(
            opt.crop_size,
            2.0 * opt.netD_scale_capacity,
            blur_kernel=[1, 3, 3, 1] if self.opt.use_antialias else [1]
        )

    def forward(self, x):
        pred = self.stylegan2_D(x)
        return pred

    def get_features(self, x):
        return self.stylegan2_D.get_features(x)

    def get_pred_from_features(self, feat, label):
        assert label is None
        feat = feat.flatten(1)
        out = self.stylegan2_D.final_linear(feat)
        return out


