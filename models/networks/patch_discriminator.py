from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
from models.networks import BaseNetwork
from models.networks.stylegan2_layers import ConvLayer, ResBlock, EqualLinear


class BasePatchDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--netPatchD_scale_capacity", default=4.0, type=float)
        parser.add_argument("--netPatchD_max_nc", default=256 + 128, type=int)
        parser.add_argument("--patch_size", default=128, type=int)
        parser.add_argument("--max_num_tiles", default=8, type=int)
        parser.add_argument("--patch_random_transformation",
                            type=util.str2bool, nargs='?', const=True, default=False)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        #self.visdom = util.Visualizer(opt)

    def needs_regularization(self):
        return False

    def extract_features(self, patches):
        raise NotImplementedError()

    def discriminate_features(self, feature1, feature2):
        raise NotImplementedError()

    def apply_random_transformation(self, patches):
        B, ntiles, C, H, W = patches.size()
        patches = patches.view(B * ntiles, C, H, W)
        before = patches
        transformer = util.RandomSpatialTransformer(self.opt, B * ntiles)
        patches = transformer.forward_transform(patches, (self.opt.patch_size, self.opt.patch_size))
        #self.visdom.display_current_results({'before': before,
        #                                     'after': patches}, 0, save_result=False)
        return patches.view(B, ntiles, C, H, W)

    def sample_patches_old(self, img, indices):
        B, C, H, W = img.size()
        s = self.opt.patch_size
        if H % s > 0 or W % s > 0:
            y_offset = torch.randint(H % s, (), device=img.device)
            x_offset = torch.randint(W % s, (), device=img.device)
            img = img[:, :,
                      y_offset:y_offset + s * (H // s),
                      x_offset:x_offset + s * (W // s)]
        img = img.view(B, C, H//s, s, W//s, s)
        ntiles = (H // s) * (W // s)
        tiles = img.permute(0, 2, 4, 1, 3, 5).reshape(B, ntiles, C, s, s)
        if indices is None:
            indices = torch.randperm(ntiles, device=img.device)[:self.opt.max_num_tiles]
            return self.apply_random_transformation(tiles[:, indices]), indices
        else:
            return self.apply_random_transformation(tiles[:, indices])

    def forward(self, real, fake, fake_only=False):
        assert real is not None
        real_patches, patch_ids = self.sample_patches(real, None)
        if fake is None:
            real_patches.requires_grad_()
        real_feat = self.extract_features(real_patches)

        bs = real.size(0)
        if fake is None or not fake_only:
            pred_real = self.discriminate_features(
                real_feat,
                torch.roll(real_feat, 1, 1))
            pred_real = pred_real.view(bs, -1)


        if fake is not None:
            fake_patches = self.sample_patches(fake, patch_ids)
            #self.visualizer.display_current_results({'real_A': real_patches[0],
            #                                         'real_B': torch.roll(fake_patches, 1, 1)[0]}, 0, False, max_num_images=16)
            fake_feat = self.extract_features(fake_patches)
            pred_fake = self.discriminate_features(
                real_feat,
                torch.roll(fake_feat, 1, 1))
            pred_fake = pred_fake.view(bs, -1)

        if fake is None:
            return pred_real, real_patches
        elif fake_only:
            return pred_fake
        else:
            return pred_real, pred_fake
  


class StyleGAN2PatchDiscriminator(BasePatchDiscriminator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        BasePatchDiscriminator.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        channel_multiplier = self.opt.netPatchD_scale_capacity
        size = self.opt.patch_size
        channels = {
            4: min(self.opt.netPatchD_max_nc, int(256 * channel_multiplier)),
            8: min(self.opt.netPatchD_max_nc, int(128 * channel_multiplier)),
            16: min(self.opt.netPatchD_max_nc, int(64 * channel_multiplier)),
            32: int(32 * channel_multiplier),
            64: int(16 * channel_multiplier),
            128: int(8 * channel_multiplier),
            256: int(4 * channel_multiplier),
        }

        log_size = int(math.ceil(math.log(size, 2)))

        in_channel = channels[2 ** log_size]

        blur_kernel = [1, 3, 3, 1] if self.opt.use_antialias else [1]

        convs = [('0', ConvLayer(3, in_channel, 3))]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            layer_name = str(7 - i) if i <= 6 else "%dx%d" % (2 ** i, 2 ** i)
            convs.append((layer_name, ResBlock(in_channel, out_channel, blur_kernel)))

            in_channel = out_channel

        convs.append(('5', ResBlock(in_channel, self.opt.netPatchD_max_nc * 2, downsample=False)))
        convs.append(('6', ConvLayer(self.opt.netPatchD_max_nc * 2, self.opt.netPatchD_max_nc, 3, pad=0)))

        self.convs = nn.Sequential(OrderedDict(convs))

        out_dim = 1

        pairlinear1 = EqualLinear(channels[4] * 2 * 2 * 2, 2048, activation='fused_lrelu')
        pairlinear2 = EqualLinear(2048, 2048, activation='fused_lrelu')
        pairlinear3 = EqualLinear(2048, 1024, activation='fused_lrelu')
        pairlinear4 = EqualLinear(1024, out_dim)
        self.pairlinear = nn.Sequential(pairlinear1, pairlinear2, pairlinear3, pairlinear4)

    def extract_features(self, patches, aggregate=False):
        if patches.ndim == 5:
            B, T, C, H, W = patches.size()
            flattened_patches = patches.flatten(0, 1)
        else:
            B, C, H, W = patches.size()
            T = patches.size(1)
            flattened_patches = patches
        features = self.convs(flattened_patches)
        features = features.view(B, T, features.size(1), features.size(2), features.size(3))
        if aggregate:
            features = features.mean(1, keepdim=True).expand(-1, T, -1, -1, -1)
        return features.flatten(0, 1)

    def extract_layerwise_features(self, image):
        feats = [image]
        for m in self.convs:
            feats.append(m(feats[-1]))

        return feats

    def discriminate_features(self, feature1, feature2):
        feature1 = feature1.flatten(1)
        feature2 = feature2.flatten(1)
        out = self.pairlinear(torch.cat([feature1, feature2], dim=1))
        return out
