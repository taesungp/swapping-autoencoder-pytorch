import torch
from models import MultiGPUModelWrapper
from optimizers.swapping_autoencoder_optimizer import SwappingAutoencoderOptimizer
import util


class ClassifierOptimizer(SwappingAutoencoderOptimizer):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = SwappingAutoencoderOptimizer.modify_commandline_options(parser, is_train)
        return parser

    def train_one_step(self, data_i, total_steps_so_far):
        images_minibatch, labels = self.prepare_images(data_i)
        c_losses = self.train_classifier_one_step(images_minibatch, labels)
        self.adjust_lr_if_necessary(total_steps_so_far)
        return util.to_numpy(c_losses)

    def train_classifier_one_step(self, images, labels):
        self.set_requires_grad(self.Gparams, False)
        self.optimizer_C.zero_grad()
        losses, metrics = self.model(images, labels, command="compute_classifier_losses")
        loss = sum([v.mean() for v in losses.values()])
        loss.backward()
        self.optimizer_C.step()
        losses.update(metrics)
        return losses

    def get_visuals_for_snapshot(self, data_i):
        images, labels = self.prepare_images(data_i)
        with torch.no_grad():
            return self.model(images, labels, command="get_visuals_for_snapshot")

    def save(self, total_steps_so_far):
        self.model.save(total_steps_so_far)
