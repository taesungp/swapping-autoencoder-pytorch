import math
import torch
import util
import torch.nn.functional as F
from models.networks import BaseNetwork
from models.networks.stylegan2_layers import ConvLayer, ToRGB, EqualLinear, StyledConv


class UpsamplingBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim,
                 blur_kernel=[1, 3, 3, 1], use_noise=False):
        super().__init__()
        self.inch, self.outch, self.styledim = inch, outch, styledim
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=True,
                                blur_kernel=blur_kernel, use_noise=use_noise)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False,
                                use_noise=use_noise)

    def forward(self, x, style):
        return self.conv2(self.conv1(x, style), style)


class ResolutionPreservingResnetBlock(torch.nn.Module):
    def __init__(self, opt, inch, outch, styledim):
        super().__init__()
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=False)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=False, bias=False)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style):
        skip = self.skip(x)
        res = self.conv2(self.conv1(x, style), style)
        return (skip + res) / math.sqrt(2)


class UpsamplingResnetBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim, blur_kernel=[1, 3, 3, 1], use_noise=False):
        super().__init__()
        self.inch, self.outch, self.styledim = inch, outch, styledim
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=True, blur_kernel=blur_kernel, use_noise=use_noise)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False, use_noise=use_noise)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=True, bias=True)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style):
        skip = F.interpolate(self.skip(x), scale_factor=2, mode='bilinear', align_corners=False)
        res = self.conv2(self.conv1(x, style), style)
        return (skip + res) / math.sqrt(2)


class GeneratorModulation(torch.nn.Module):
    def __init__(self, styledim, outch):
        super().__init__()
        self.scale = EqualLinear(styledim, outch)
        self.bias = EqualLinear(styledim, outch)

    def forward(self, x, style):
        if style.ndimension() <= 2:
            return x * (1 * self.scale(style)[:, :, None, None]) + self.bias(style)[:, :, None, None]
        else:
            style = F.interpolate(style, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            return x * (1 * self.scale(style)) + self.bias(style)


class StyleGAN2ResnetGenerator(BaseNetwork):
    """ The Generator (decoder) architecture described in Figure 18 of
        Swapping Autoencoder (https://arxiv.org/abs/2007.00653).
        
        At high level, the architecture consists of regular and 
        upsampling residual blocks to transform the structure code into an RGB
        image. The global code is applied at each layer as modulation.
        
        Here's more detailed architecture:
        
        1. SpatialCodeModulation: First of all, modulate the structure code 
        with the global code.
        2. HeadResnetBlock: resnets at the resolution of the structure code,
        which also incorporates modulation from the global code.
        3. UpsamplingResnetBlock: resnets that upsamples by factor of 2 until
        the resolution of the output RGB image, along with the global code
        modulation.
        4. ToRGB: Final layer that transforms the output into 3 channels (RGB).
        
        Each components of the layers borrow heavily from StyleGAN2 code,
        implemented by Seonghyeon Kim.
        https://github.com/rosinality/stylegan2-pytorch
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--netG_scale_capacity", default=1.0, type=float)
        parser.add_argument(
            "--netG_num_base_resnet_layers",
            default=2, type=int,
            help="The number of resnet layers before the upsampling layers."
        )
        parser.add_argument("--netG_use_noise", type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--netG_resnet_ch", type=int, default=256)

        return parser

    def __init__(self, opt):
        super().__init__(opt)
        num_upsamplings = opt.netE_num_downsampling_sp
        blur_kernel = [1, 3, 3, 1] if opt.use_antialias else [1]

        self.global_code_ch = opt.global_code_ch + opt.num_classes

        self.add_module(
            "SpatialCodeModulation",
            GeneratorModulation(self.global_code_ch, opt.spatial_code_ch))

        in_channel = opt.spatial_code_ch
        for i in range(opt.netG_num_base_resnet_layers):
            # gradually increase the number of channels
            out_channel = (i + 1) / opt.netG_num_base_resnet_layers * self.nf(0)
            out_channel = max(opt.spatial_code_ch, round(out_channel))
            layer_name = "HeadResnetBlock%d" % i
            new_layer = ResolutionPreservingResnetBlock(
                opt, in_channel, out_channel, self.global_code_ch)
            self.add_module(layer_name, new_layer)
            in_channel = out_channel

        for j in range(num_upsamplings):
            out_channel = self.nf(j + 1)
            layer_name = "UpsamplingResBlock%d" % (2 ** (4 + j))
            new_layer = UpsamplingResnetBlock(
                in_channel, out_channel, self.global_code_ch,
                blur_kernel, opt.netG_use_noise)
            self.add_module(layer_name, new_layer)
            in_channel = out_channel

        last_layer = ToRGB(out_channel, self.global_code_ch,
                           blur_kernel=blur_kernel)
        self.add_module("ToRGB", last_layer)

    def nf(self, num_up):
        ch = 128 * (2 ** (self.opt.netE_num_downsampling_sp - num_up))
        ch = int(min(512, ch) * self.opt.netG_scale_capacity)
        return ch

    def forward(self, spatial_code, global_code):
        spatial_code = util.normalize(spatial_code)
        global_code = util.normalize(global_code)

        x = self.SpatialCodeModulation(spatial_code, global_code)
        for i in range(self.opt.netG_num_base_resnet_layers):
            resblock = getattr(self, "HeadResnetBlock%d" % i)
            x = resblock(x, global_code)

        for j in range(self.opt.netE_num_downsampling_sp):
            key_name = 2 ** (4 + j)
            upsampling_layer = getattr(self, "UpsamplingResBlock%d" % key_name)
            x = upsampling_layer(x, global_code)
        rgb = self.ToRGB(x, global_code, None)

        return rgb
