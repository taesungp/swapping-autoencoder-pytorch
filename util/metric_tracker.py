from collections import OrderedDict


class MetricTracker:
    def __init__(self, opt):
        self.opt = opt
        self.metrics = {}

    def moving_average(self, old, new):
        s = 0.98
        return old * (s) + new * (1 - s)

    def update_metrics(self, metric_dict, smoothe=True):
        default_smoothe = smoothe
        for k, v in metric_dict.items():
            if k == "D_R1":
                smoothe = False
            else:
                smoothe = default_smoothe
            if k in self.metrics and smoothe:
                self.metrics[k] = self.moving_average(self.metrics[k], v)
            else:
                self.metrics[k] = v

    def current_metrics(self):
        keys = sorted(list(self.metrics.keys()))
        ordered_metrics = OrderedDict([(k, self.metrics[k]) for k in keys])
        return ordered_metrics
