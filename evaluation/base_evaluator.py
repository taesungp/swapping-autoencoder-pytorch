import os


class BaseEvaluator():
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, target_phase):
        super().__init__()
        self.opt = opt
        self.target_phase = target_phase

    def output_dir(self):
        evaluator_name = str(type(self).__name__).lower().replace('evaluator', '')
        expr_name = self.opt.name
        if self.opt.isTrain:
            result_dir = os.path.join(self.opt.checkpoints_dir, expr_name, "snapshots")
        else:
            result_dir = os.path.join(self.opt.result_dir, expr_name, evaluator_name)
        return result_dir

    def evaluate(self, model, dataset, nsteps=None):
        pass
