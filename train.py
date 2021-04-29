import data
import models
import optimizers
from options import TrainOptions
from util import IterationCounter
from util import Visualizer
from util import MetricTracker
from evaluation import GroupEvaluator


opt = TrainOptions().parse()
dataset = data.create_dataset(opt)
opt.dataset = dataset
iter_counter = IterationCounter(opt)
visualizer = Visualizer(opt)
metric_tracker = MetricTracker(opt)
evaluators = GroupEvaluator(opt)

model = models.create_model(opt)
optimizer = optimizers.create_optimizer(opt, model)

while not iter_counter.completed_training():
    with iter_counter.time_measurement("data"):
        cur_data = next(dataset)

    with iter_counter.time_measurement("train"):
        losses = optimizer.train_one_step(cur_data, iter_counter.steps_so_far)
        metric_tracker.update_metrics(losses, smoothe=True)

    with iter_counter.time_measurement("maintenance"):
        if iter_counter.needs_printing():
            visualizer.print_current_losses(iter_counter.steps_so_far,
                                            iter_counter.time_measurements,
                                            metric_tracker.current_metrics())

        if iter_counter.needs_displaying():
            visuals = optimizer.get_visuals_for_snapshot(cur_data)
            visualizer.display_current_results(visuals,
                                               iter_counter.steps_so_far)

        if iter_counter.needs_evaluation():
            metrics = evaluators.evaluate(
                model, dataset, iter_counter.steps_so_far)
            metric_tracker.update_metrics(metrics, smoothe=False)

        if iter_counter.needs_saving():
            optimizer.save(iter_counter.steps_so_far)

        if iter_counter.completed_training():
            break

        iter_counter.record_one_iteration()

optimizer.save(iter_counter.steps_so_far)
print('Training finished.')
