import os
import importlib
from optimizers.base_optimizer import BaseOptimizer
import torch


def find_optimizer_using_name(optimizer_name):
    """Import the module "optimizers/[optimizer_name]_optimizer.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseOptimizer,
    and it is case-insensitive.
    """
    optimizer_filename = "optimizers." + optimizer_name + "_optimizer"
    optimizerlib = importlib.import_module(optimizer_filename)
    optimizer = None
    target_optimizer_name = optimizer_name.replace('_', '') + 'optimizer'
    for name, cls in optimizerlib.__dict__.items():
        if name.lower() == target_optimizer_name.lower() \
           and issubclass(cls, BaseOptimizer):
            optimizer = cls

    if optimizer is None:
        print("In %s.py, there should be a subclass of BaseOptimizer with class name that matches %s in lowercase." % (optimizer_filename, target_optimizer_name))
        exit(0)

    return optimizer


def get_option_setter(optimizer_name):
    """Return the static method <modify_commandline_options> of the optimizer class."""
    optimizer_class = find_optimizer_using_name(optimizer_name)
    return optimizer_class.modify_commandline_options


def create_optimizer(opt, model):
    """Create a optimizer given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from optimizers import create_optimizer
        >>> optimizer = create_optimizer(opt)
    """
    optimizer = find_optimizer_using_name(opt.optimizer)
    instance = optimizer(model)
    return instance
