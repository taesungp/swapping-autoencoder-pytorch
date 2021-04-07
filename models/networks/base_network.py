import torch


class BaseNetwork(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def print_architecture(self, verbose=False):
        name = type(self).__name__
        result = '-------------------%s---------------------\n' % name
        total_num_params = 0
        for i, (name, child) in enumerate(self.named_children()):
            num_params = sum([p.numel() for p in child.parameters()])
            total_num_params += num_params
            if verbose:
                result += "%s: %3.3fM\n" % (name, (num_params / 1e6))
            for i, (name, grandchild) in enumerate(child.named_children()):
                num_params = sum([p.numel() for p in grandchild.parameters()])
                if verbose:
                    result += "\t%s: %3.3fM\n" % (name, (num_params / 1e6))
        result += '[Network %s] Total number of parameters : %.3f M\n' % (name, total_num_params / 1e6)
        result += '-----------------------------------------------\n'
        print(result)

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def collect_parameters(self, name):
        params = []
        for m in self.modules():
            if type(m).__name__ == name:
                params += list(m.parameters())
        return params

    def fix_and_gather_noise_parameters(self):
        params = []
        device = next(self.parameters()).device
        for m in self.modules():
            if type(m).__name__ == "NoiseInjection":
                assert m.image_size is not None, "One forward call should be made to determine size of noise parameters"
                m.fixed_noise = torch.nn.Parameter(torch.randn(m.image_size[0], 1, m.image_size[2], m.image_size[3], device=device))
                params.append(m.fixed_noise)
        return params

    def remove_noise_parameters(self, name):
        for m in self.modules():
            if type(m).__name__ == "NoiseInjection":
                m.fixed_noise = None

    def forward(self, x):
        return x
