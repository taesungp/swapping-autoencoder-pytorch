import os
import torch


class BaseModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = torch.device('cuda:0') if opt.num_gpus > 0 else torch.device('cpu')

    def initialize(self):
        pass

    def per_gpu_initialize(self):
        pass

    def compute_generator_losses(self, data_i):
        return {}

    def compute_discriminator_losses(self, data_i):
        return {}

    def get_visuals_for_snapshot(self, data_i):
        return {}

    def get_parameters_for_mode(self, mode):
        return {}

    def save(self, total_steps_so_far):
        savedir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        checkpoint_name = "%dk_checkpoint.pth" % (total_steps_so_far // 1000)
        savepath = os.path.join(savedir, checkpoint_name)
        torch.save(self.state_dict(), savepath)
        sympath = os.path.join(savedir, "latest_checkpoint.pth")
        if os.path.exists(sympath):
            os.remove(sympath)
        os.symlink(checkpoint_name, sympath)

    def load(self):
        if self.opt.isTrain and self.opt.pretrained_name is not None:
            loaddir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
        else:
            loaddir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        checkpoint_name = "%s_checkpoint.pth" % self.opt.resume_iter
        checkpoint_path = os.path.join(loaddir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            print("\n\ncheckpoint %s does not exist!" % checkpoint_path)
            assert self.opt.isTrain, "In test mode, the checkpoint file must exist"
            print("Training will start from scratch")
            return
        state_dict = torch.load(checkpoint_path,
                                map_location=str(self.device))
        # self.load_state_dict(state_dict)
        own_state = self.state_dict()
        skip_all = False
        for name, own_param in own_state.items():
            if not self.opt.isTrain and (name.startswith("D.") or name.startswith("Dpatch.")):
                continue
            if name not in state_dict:
                print("Key %s does not exist in checkpoint. Skipping..." % name)
                continue
            # if name.startswith("C.net"):
            #    continue
            param = state_dict[name]
            if own_param.shape != param.shape:
                message = "Key [%s]: Shape does not match the created model (%s) and loaded checkpoint (%s)" % (name, str(own_param.shape), str(param.shape))
                if skip_all:
                    print(message)
                    min_shape = [min(s1, s2) for s1, s2 in zip(own_param.shape, param.shape)]
                    ms = min_shape
                    if len(min_shape) == 1:
                        own_param[:ms[0]].copy_(param[:ms[0]])
                        own_param[ms[0]:].copy_(own_param[ms[0]:] * 0)
                    elif len(min_shape) == 2:
                        own_param[:ms[0], :ms[1]].copy_(param[:ms[0], :ms[1]])
                        own_param[ms[0]:, ms[1]:].copy_(own_param[ms[0]:, ms[1]:] * 0)
                    elif len(ms) == 4:
                        own_param[:ms[0], :ms[1], :ms[2], :ms[3]].copy_(param[:ms[0], :ms[1], :ms[2], :ms[3]])
                        own_param[ms[0]:, ms[1]:, ms[2]:, ms[3]:].copy_(own_param[ms[0]:, ms[1]:, ms[2]:, ms[3]:] * 0)
                    else:
                        print("Skipping min_shape of %s" % str(ms))
                    continue
                userinput = input("%s. Force loading? (yes, no, all) " % (message))
                if userinput.lower() == "yes":
                    pass
                elif userinput.lower() == "no":
                    #assert own_param.shape == param.shape
                    continue
                elif userinput.lower() == "all":
                    skip_all = True
                else:
                    raise ValueError(userinput)
                min_shape = [min(s1, s2) for s1, s2 in zip(own_param.shape, param.shape)]
                ms = min_shape
                if len(min_shape) == 1:
                    own_param[:ms[0]].copy_(param[:ms[0]])
                    own_param[ms[0]:].copy_(own_param[ms[0]:] * 0)
                elif len(min_shape) == 2:
                    own_param[:ms[0], :ms[1]].copy_(param[:ms[0], :ms[1]])
                    own_param[ms[0]:, ms[1]:].copy_(own_param[ms[0]:, ms[1]:] * 0)
                elif len(ms) == 4:
                    own_param[:ms[0], :ms[1], :ms[2], :ms[3]].copy_(param[:ms[0], :ms[1], :ms[2], :ms[3]])
                    own_param[ms[0]:, ms[1]:, ms[2]:, ms[3]:].copy_(own_param[ms[0]:, ms[1]:, ms[2]:, ms[3]:] * 0)
                else:
                    print("Skipping min_shape of %s" % str(ms))
                continue
            own_param.copy_(param)
        print("checkpoint loaded from %s" % os.path.join(loaddir, checkpoint_name))

    def forward(self, *args, command=None, **kwargs):
        """ wrapper for multigpu training. BaseModel is expected to be
        wrapped in nn.parallel.DataParallel, which distributes its call to
        the BaseModel instance on each GPU """
        if command is not None:
            method = getattr(self, command)
            assert callable(method), "[%s] is not a method of %s" % (command, type(self).__name__)
            return method(*args, **kwargs)
        else:
            raise ValueError(command)
