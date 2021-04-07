"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import os
import importlib
from models.base_model import BaseModel
import torch
from torch.nn.parallel import DataParallel


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    instance.initialize()
    multigpu_instance = MultiGPUModelWrapper(opt, instance)
    print("model [%s] was created" % type(instance).__name__)
    return multigpu_instance


class MultiGPUModelWrapper():
    def __init__(self, opt, model: BaseModel):
        self.opt = opt
        if opt.num_gpus > 0:
            model = model.to('cuda:0')
        self.parallelized_model = torch.nn.parallel.DataParallel(model)
        self.parallelized_model(command="per_gpu_initialize")
        self.singlegpu_model = self.parallelized_model.module
        self.singlegpu_model(command="per_gpu_initialize")

    def get_parameters_for_mode(self, mode):
        return self.singlegpu_model.get_parameters_for_mode(mode)

    def save(self, total_steps_so_far):
        self.singlegpu_model.save(total_steps_so_far)

    def __call__(self, *args, **kwargs):
        """ Calls are forwarded to __call__ of BaseModel through DataParallel, and corresponding methods specified by |command| will be called. Please see BaseModel.forward() to see how it is done. """
        return self.parallelized_model(*args, **kwargs)


class StateVariableStorage():
    pass


_state_variables = StateVariableStorage()
_state_variables.fix_noise = False


def fixed_noise():
    return _state_variables.fix_noise


def fix_noise(set=True):
    _state_variables.fix_noise = set
