import torch
import util
from .base_network import BaseNetwork


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    netE_cls = find_network_using_name(opt.netE, 'encoder')
    assert netE_cls is not None
    parser = netE_cls.modify_commandline_options(parser, is_train)

    netG_cls = find_network_using_name(opt.netG, 'generator')
    assert netG_cls is not None
    parser = netG_cls.modify_commandline_options(parser, is_train)

    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    parser = netD_cls.modify_commandline_options(parser, is_train)

    if opt.netPatchD is not None:
        netD_cls = find_network_using_name(opt.netPatchD, 'patch_discriminator')
        assert netD_cls is not None
        parser = netD_cls.modify_commandline_options(parser, is_train)

    return parser


def create_network(opt, network_name, mode, verbose=True):
    if network_name is None:
        return None
    net_cls = find_network_using_name(network_name, mode)
    net = net_cls(opt)
    if verbose:
        net.print_architecture(verbose=True)
    return net
