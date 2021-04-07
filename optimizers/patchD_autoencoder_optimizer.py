import torch
from models import MultiGPUModelWrapper
from optimizers.base_optimizer import BaseOptimizer
import util


class PatchDAutoencoderOptimizer(BaseOptimizer):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--lr", default=0.002, type=float)
        parser.add_argument("--beta1", default=0.0, type=float)
        parser.add_argument("--beta2", default=0.99, type=float)
        parser.add_argument("--R1_once_every", default=16, type=int,
                            help="lazy R1 regularization. R1 loss is computed once in 1/R1_freq times")

        return parser

    def __init__(self, model: MultiGPUModelWrapper):
        self.opt = model.opt
        opt = self.opt
        self.model = model
        self.training_mode_index = 0

        self.Gparams = self.model.get_parameters_for_mode("generator")
        self.Dparams = self.model.get_parameters_for_mode("discriminator")

        self.num_discriminator_iters = 0
        self.optimizer_G = torch.optim.Adam(self.Gparams, lr=opt.lr,
                                            betas=(opt.beta1, opt.beta2))
        # StyleGAN2 Appendix B
        c = opt.R1_once_every / (1 + opt.R1_once_every)
        self.optimizer_D = torch.optim.Adam(self.Dparams,
                                            lr=opt.lr * c,
                                            betas=(opt.beta1 ** c,
                                                   opt.beta2 ** c))

    def set_requires_grad(self, params, requires_grad):
        for p in params:
            p.requires_grad_(requires_grad)

    def prepare_images(self, data_i):
        A = data_i["real_A"]
        if "real_B" in data_i:
            B = data_i["real_B"]
            A = torch.cat([A, B], dim=0)
            A = A[torch.randperm(A.size(0))]
        return A

    def toggle_training_mode(self):
        all_modes = ["generator", "discriminator"]
        self.training_mode_index = (self.training_mode_index + 1) % len(all_modes)
        return all_modes[self.training_mode_index]

    def train_one_step(self, data_i, total_steps_so_far):
        images_minibatch = self.prepare_images(data_i)
        if self.toggle_training_mode() == "generator":
            losses = self.train_discriminator_one_step(images_minibatch)
        else:
            losses = self.train_generator_one_step(images_minibatch)
        return util.to_numpy(losses)

    def train_generator_one_step(self, images):
        self.set_requires_grad(self.Dparams, False)
        self.set_requires_grad(self.Gparams, True)
        _, gl_ma = self.model(images, command="encode",
                              use_momentum_encoder=True)
        self.optimizer_G.zero_grad()
        g_losses, g_metrics = self.model(images, gl_ma,
                                         command="compute_generator_losses")
        g_loss = sum([v.mean() for v in g_losses.values()])
        g_loss.backward()
        self.optimizer_G.step()
        g_losses.update(g_metrics)
        return g_losses

    def train_discriminator_one_step(self, images):
        if self.opt.lambda_GAN == 0.0 and self.opt.lambda_PatchGAN == 0.0:
            return {}
        self.set_requires_grad(self.Dparams, True)
        self.set_requires_grad(self.Gparams, False)
        self.num_discriminator_iters += 1
        self.optimizer_D.zero_grad()

        d_losses, d_metrics, features = self.model(images,
                                                   command="compute_discriminator_losses")
        nce_losses, nce_metrics = self.model.singlegpu_model(*features,
                                                             command="compute_discriminator_nce_losses")
        d_losses.update(nce_losses)
        d_metrics.update(nce_metrics)
        d_loss = sum([v.mean() for v in d_losses.values()])
        d_loss.backward()
        self.optimizer_D.step()
        needs_R1 = (self.opt.lambda_R1 > 0.0 or self.opt.lambda_patch_R1) and \
                   (self.num_discriminator_iters % self.opt.R1_once_every == 0)
        if needs_R1:
            self.optimizer_D.zero_grad()
            r1_losses = self.model(images,
                                   command="compute_R1_loss")
            d_losses.update(r1_losses)
            r1_loss = sum([v.mean() for v in r1_losses.values()])
            r1_loss = r1_loss * self.opt.R1_once_every
            r1_loss.backward()
            self.optimizer_D.step()

        d_losses.update(d_metrics)
        return d_losses

    def get_visuals_for_snapshot(self, data_i):
        images = self.prepare_images(data_i)
        with torch.no_grad():
            return self.model(images, command="get_visuals_for_snapshot")

    def save(self, total_steps_so_far):
        self.model.save(total_steps_so_far)
