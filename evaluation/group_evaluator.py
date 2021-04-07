import torch
from .base_evaluator import BaseEvaluator
import util

def find_evaluator_using_name(filename):
    target_class_name = filename
    module_name = 'evaluation.' + filename
    eval_class = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(eval_class, BaseEvaluator), \
        "Class %s should be a subclass of BaseEvaluator" % eval_class

    return eval_class


def find_evaluator_classes(opt):
    if len(opt.evaluation_metrics) == 0:
        return []

    eval_metrics = opt.evaluation_metrics.split(",")

    all_classes = []
    target_phases = []
    for metric in eval_metrics:
        if metric.startswith("train"):
            target_phases.append("train")
            metric = metric[len("train"):]
        elif metric.startswith("test"):
            target_phases.append("test")
            metric = metric[len("test"):]
        else:
            target_phases.append("test")

        metric_class = find_evaluator_using_name("%s_evaluator" % metric)
        all_classes.append(metric_class)

    return all_classes, target_phases


class GroupEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--evaluation_metrics", default="none")

        opt, _ = parser.parse_known_args()
        evaluator_classes, _ = find_evaluator_classes(opt)

        for eval_class in evaluator_classes:
            parser = eval_class.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt, target_phase=None):
        super().__init__(opt, target_phase=None)
        self.opt = opt
        evaluator_classes, target_phases = find_evaluator_classes(opt)
        self.evaluators = [cls(opt, target_phase=phs) for cls, phs in zip(evaluator_classes, target_phases)]

    def evaluate(self, model, dataset, nsteps=None):
        original_phase = dataset.phase
        metrics = {}
        for i, evaluator in enumerate(self.evaluators):
            print("Entering evaluation using %s on %s images" % (type(evaluator).__name__, evaluator.target_phase))
            dataset.set_phase(evaluator.target_phase)
            with torch.no_grad():
                new_metrics = evaluator.evaluate(model, dataset, nsteps)
                metrics.update(new_metrics)
            print("Finished evaluation of %s" % type(evaluator).__name__)
        dataset.set_phase(original_phase)
        return metrics
