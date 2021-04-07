"""
experiment launcher using tmux panes
"""
import os
import math
import GPUtil
import re

available_gpu_devices = None


class Options():
    def __init__(self):
        self.args = []
        self.kvs = {}
        self.tag_str = None

    def set(self, *args, **kwargs):
        for a in args:
            self.args.append(a)
        for k, v in kwargs.items():
            self.kvs[k] = v

        return self

    def remove(self, *args):
        for a in args:
            if a in self.args:
                self.args.remove(a)
            if a in self.kvs:
                del self.kvs[a]

        return self

    def update(self, opt):
        self.args += opt.args
        self.kvs.update(opt.kvs)
        return self

    def __str__(self):
        final = " ".join(self.args)
        for k, v in self.kvs.items():
            final += " --{} {}".format(k, v)

        return final

    def clone(self):
        opt = Options()
        opt.args = self.args.copy()
        opt.kvs = self.kvs.copy()
        opt.tag_str = self.tag_str
        return opt

    def specify(self, *args, **kwargs):
        return self.clone().set(*args, **kwargs)
    
    def tag(self, tag):
        self.tag_str = tag
        return self


def grab_pattern(pattern, text):
    found = re.search(pattern, text)
    if found is not None:
        return found[1]
    else:
        None

# http://code.activestate.com/recipes/252177-find-the-common-beginning-in-a-list-of-strings/


def findcommonstart(strlist):
    prefix_len = ([min([x[0] == elem for elem in x])
                   for x in zip(*strlist)] + [0]).index(0)
    prefix_len = max(1, prefix_len - 4)
    return strlist[0][:prefix_len]


class TmuxLauncher():
    def __init__(self):
        super().__init__()
        self.tmux_prepared = False

    def commands(self):
        opts = self.train_options()
        return ["python train.py " + str(opt) for opt in opts]

    def test_commands(self):
        opts = self.test_options()
        return ["python test.py " + str(opt) for opt in opts]

    def options(self):
        return []

    def train_options(self):
        return self.options()

    def test_options(self):
        return self.options()
    
    def find_tag(self, options, tag):
        for i, opt in enumerate(options):
            if opt.tag_str == tag:
                return i
        for i, opt in enumerate(options):
            if opt.kvs["name"] == tag:
                return i
        raise ValueError("Tag %s does not exist in the command lists" % tag)

    def prepare_tmux_panes(self, num_experiments, dry=False):
        self.pane_per_window = 1
        self.n_windows = int(math.ceil(num_experiments / self.pane_per_window))
        print('preparing {} tmux panes'.format(num_experiments))
        for w in range(self.n_windows):
            if dry:
                continue
            window_name = "experiments_{}".format(w)
            os.system("tmux new-window -n {}".format(window_name))
            #os.system("tmux split-window -t {} -h".format(window_name))
            #os.system("tmux split-window -t {} -v".format(window_name))
            #os.system("tmux split-window -t {}.0 -v".format(window_name))
        self.tmux_prepared = True

    def refine_command(self, command, resume_iter, continue_train, gpu_id=None):
        command = str(command)
        if "--num_gpus" in command:
            num_gpus = int(re.search(r'--num_gpus ([\d,?]+)', command)[1])
        else:
            num_gpus = 1

        global available_gpu_devices
        if available_gpu_devices is None and gpu_id is None:
            available_gpu_devices = [str(g) for g in GPUtil.getAvailable(limit=8, maxMemory=0.2)]
        if gpu_id is not None:
            available_gpu_devices = [i for i in str(gpu_id)]
        if len(available_gpu_devices) < num_gpus:
            raise ValueError("{} GPU(s) required for the command {} is not available".format(num_gpus, command))
        active_devices = ",".join(available_gpu_devices[:num_gpus])
        if resume_iter is not None:
            resume_iter = " --resume_iter %s " % resume_iter
        else:
            resume_iter = ""
        command = "CUDA_VISIBLE_DEVICES={} {} {}".format(active_devices, command, resume_iter)
        if continue_train:
            command += " --continue_train "

        # available_gpu_devices = [str(g) for g in GPUtil.getAvailable(limit=8, maxMemory=0.8)]
        available_gpu_devices = available_gpu_devices[num_gpus:]

        return command

    def send_command(self, exp_id, command, dry=False, continue_train=False):
        command = self.refine_command(command, None, continue_train, gpu_id=None)
        pane_name = "experiments_{windowid}.{paneid}".format(windowid=exp_id // self.pane_per_window,
                                                             paneid=exp_id % self.pane_per_window)
        if dry is False:
            os.system("tmux send-keys -t {} \"{}\" Enter".format(pane_name, command))

        print("{}: {}".format(pane_name, command))
        return pane_name
    
    def run_command(self, command, ids, resume_iter=None, continue_train=False, gpu_id=None):
        if type(command) is not list:
            command = [command]
        if ids is None:
            ids = list(range(len(command)))
        if type(ids) is not list:
            ids = [ids]

        for id in ids:
            this_command = command[id]
            refined_command = self.refine_command(this_command, resume_iter, continue_train=continue_train, gpu_id=gpu_id)
            num_repeats = 1
            for trial_id in range(num_repeats):
                if trial_id > 0:
                    print("Running the command again since last command returned nonzero")
                print(refined_command)
                result = os.system(refined_command)
                if result == 0:
                    break

    def launch(self, ids, test=False, dry=False, continue_train=False):
        commands = self.test_commands() if test else self.commands()
        if type(ids) is list:
            commands = [commands[i] for i in ids]
        if not self.tmux_prepared:
            self.prepare_tmux_panes(len(commands), dry)
            assert self.tmux_prepared

        for i, command in enumerate(commands):
            self.send_command(i, command, dry, continue_train=continue_train)

    def dry(self):
        self.launch(dry=True)

    def stop(self):
        num_experiments = len(self.commands())
        self.pane_per_window = 4
        self.n_windows = int(math.ceil(num_experiments / self.pane_per_window))
        for w in range(self.n_windows):
            window_name = "experiments_{}".format(w)
            for i in range(self.pane_per_window):
                os.system("tmux send-keys -t {window}.{pane} C-c".format(window=window_name, pane=i))

    def close(self):
        num_experiments = len(self.commands())
        self.pane_per_window = 1
        self.n_windows = int(math.ceil(num_experiments / self.pane_per_window))
        for w in range(self.n_windows):
            window_name = "experiments_{}".format(w)
            os.system("tmux kill-window -t {}".format(window_name))

    def print_names(self, ids, test=False):
        if test:
            cmds = self.test_commands()
        else:
            cmds = self.commands()
        if type(ids) is list:
            cmds = [cmds[i] for i in ids]

        for cmdid, cmd in enumerate(cmds):
            name = grab_pattern(r'--name ([^ ]+)', cmd)
            print(name)

    def create_comparison_html(self, expr_name, ids, subdir, title, phase):
        cmds = self.test_commands()
        if type(ids) is list:
            cmds = [cmds[i] for i in ids]

        no_easy_label = True
        dirs = []
        labels = []
        for cmdid, cmd in enumerate(cmds):
            name = grab_pattern(r'--name ([^ ]+)', cmd)
            resume_iter = grab_pattern(r'--resume_iter ([^ ]+)', cmd)
            if resume_iter is None:
                resume_iter = "latest"
            label = grab_pattern(r'--easy_label "([^"]+)"', cmd)
            if label is None:
                label = name
            else:
                no_easy_label = False
            labels.append(label)
            dir = "results/%s/%s_%s/%s/" % (name, phase, resume_iter, subdir)
            dirs.append(dir)

        commonprefix = findcommonstart(labels) if no_easy_label else ""
        labels = ['"' + label[len(commonprefix):] + '"' for label in labels]
        dirstr = ' '.join(dirs)
        labelstr = ' '.join(labels)

        command = "python ~/tools/html.py --web_dir_prefix results/comparison_ --name %s --dirs %s --labels %s --image_width 256" % (expr_name + '_' + title, dirstr, labelstr)
        print(command)
        os.system(command)

    def plot_loss(self, ids, mode, name):
        from .plotter import plot_entrypoint
        plot_entrypoint(self, ids, mode, name)
        return

    def gather_metrics(self, ids, mode, name):
        from .plotter import gather_metrics
        gather_metrics(self, ids, mode, name)
