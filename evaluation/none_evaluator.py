from .base_evaluator import BaseEvaluator


class NoneEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, target_phase):
        super().__init__(opt, target_phase)
        self.opt = opt

    def evaluate(self, model, dataset, nsteps):
        return {}
