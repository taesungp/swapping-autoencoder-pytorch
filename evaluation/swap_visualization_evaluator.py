import os
from PIL import Image
import numpy as np
import torch
from evaluation import BaseEvaluator
import util


class SwapVisualizationEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--swap_num_columns", type=int, default=4,
                            help="number of images to be shown in the swap visualization grid. Setting this value will result in 4x4 swapping grid, with additional row and col for showing original images.")
        parser.add_argument("--swap_num_images", type=int, default=16,
                            help="total number of images to perform swapping. In the end, (swap_num_images / swap_num_columns) grid will be saved to disk")
        return parser

    def gather_images(self, dataset):
        all_images = []
        num_images_to_gather = max(self.opt.swap_num_columns, self.opt.num_gpus)
        exhausted = False
        while len(all_images) < num_images_to_gather:
            try:
                data = next(dataset)
            except StopIteration:
                print("Exhausted the dataset at %s" % (self.opt.dataroot))
                exhausted = True
                break
            for i in range(data["real_A"].size(0)):
                all_images.append(data["real_A"][i:i+1])
                if "real_B" in data:
                    all_images.append(data["real_B"][i:i+1])
                if len(all_images) >= num_images_to_gather:
                    break
        if len(all_images) == 0:
            return None, None, True
        return all_images, exhausted

    def generate_mix_grid(self, model, images):
        sps, gls = [], []
        for image in images:
            assert image.size(0) == 1
            sp, gl = model(image.expand(self.opt.num_gpus, -1, -1, -1), command="encode")
            sp = sp[:1]
            gl = gl[:1]
            sps.append(sp)
            gls.append(gl)
        gl = torch.cat(gls, dim=0)
        
        def put_img(img, canvas, row, col):
            h, w = img.shape[0], img.shape[1]
            start_x = int(self.opt.load_size * col + (self.opt.load_size - w) * 0.5)
            start_y = int(self.opt.load_size * row + (self.opt.load_size - h) * 0.5)
            canvas[start_y:start_y + h, start_x: start_x + w] = img
        grid_w = self.opt.load_size * (gl.size(0) + 1)
        grid_h = self.opt.load_size * (gl.size(0) + 1)
        grid_img = np.ones((grid_h, grid_w, 3), dtype=np.uint8)
        #images_np = util.tensor2im(images, tile=False)
        for i, image in enumerate(images):
            image_np = util.tensor2im(image, tile=False)[0]
            put_img(image_np, grid_img, 0, i + 1)
            put_img(image_np, grid_img, i + 1, 0)

        for i, sp in enumerate(sps):
            sp_for_current_row = sp.repeat(gl.size(0), 1, 1, 1)
            mix_row = model(sp_for_current_row, gl, command="decode")
            mix_row = util.tensor2im(mix_row, tile=False)
            for j, mix in enumerate(mix_row):
                put_img(mix, grid_img, i + 1, j + 1)

        final_grid = Image.fromarray(grid_img)
        return final_grid

    def evaluate(self, model, dataset, nsteps):
        nsteps = self.opt.resume_iter if nsteps is None else str(round(nsteps / 1000)) + "k"
        savedir = os.path.join(self.output_dir(), "%s_%s" % (self.target_phase, nsteps))
        os.makedirs(savedir, exist_ok=True)
        webpage_title = "Swap Visualization of %s. iter=%s. phase=%s" % \
                        (self.opt.name, str(nsteps), self.target_phase)
        webpage = util.HTML(savedir, webpage_title)
        num_repeats = int(np.ceil(self.opt.swap_num_images / max(self.opt.swap_num_columns, self.opt.num_gpus)))
        for i in range(num_repeats):
            images, should_break = self.gather_images(dataset)
            if images is None:
                break
            mix_grid = self.generate_mix_grid(model, images)
            webpage.add_images([mix_grid], ["%04d.png" % i])
            if should_break:
                break
        webpage.save()
        return {}
