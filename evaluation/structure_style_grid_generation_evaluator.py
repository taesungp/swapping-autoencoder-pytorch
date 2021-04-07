import os
import torch
from evaluation import BaseEvaluator
import util
import numpy as np
from PIL import Image


class StructureStyleGridGenerationEvaluator(BaseEvaluator):
    """ generate swapping images and save to disk """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def create_webpage(self, nsteps):
        nsteps = self.opt.resume_iter if nsteps is None else nsteps
        savedir = os.path.join(self.output_dir(), "%s_%s" % (self.target_phase, nsteps))
        os.makedirs(savedir, exist_ok=True)
        webpage_title = "%s. iter=%s. phase=%s" % \
                        (self.opt.name, str(nsteps), self.target_phase)
        self.webpage = util.HTML(savedir, webpage_title)

    def add_to_webpage(self, images, filenames, tile=1):
        converted_images = []
        for image in images:
            if isinstance(image, list):
                image = torch.stack(image, dim=0).flatten(0, 1)
            image = Image.fromarray(util.tensor2im(image, tile=min(image.size(0), tile)))
            converted_images.append(image)

        self.webpage.add_images(converted_images,
                                filenames)
        print("saved %s" % str(filenames))
        #self.webpage.save()

    def evaluate(self, model, dataset, nsteps=None):
        self.create_webpage(nsteps)

        structure_images, style_images = {}, {}
        for i, data_i in enumerate(dataset):
            bs = data_i["real_A"].size(0)
            sp, gl = model(data_i["real_A"].cuda(), command="encode")

            for j in range(bs):
                image = data_i["real_A"][j:j+1]
                path = data_i["path_A"][j]
                imagename = os.path.splitext(os.path.basename(path))[0]
                if "/structure/" in path:
                    structure_images[imagename] = image
                else:
                    style_images[imagename] = image

        gls = []
        style_paths = list(style_images.keys())
        for style_path in style_paths:
            style_image = style_images[style_path].cuda()
            gls.append(model(style_image, command="encode")[1])

        sps = []
        structure_paths = list(structure_images.keys())
        for structure_path in structure_paths:
            structure_image = structure_images[structure_path].cuda()
            sps.append(model(structure_image, command="encode")[0])

        # top row to show the input images
        blank_image = style_images[style_paths[0]] * 0.0 + 1.0
        self.add_to_webpage([blank_image] + [style_images[style_path] for style_path in style_paths],
                            ["blank.png"] + [style_path + ".png" for style_path in style_paths],
                            tile=1)

        # swapping
        for i, structure_path in enumerate(structure_paths):
            structure_image = structure_images[structure_path]
            swaps = []
            filenames = []
            for j, style_path in enumerate(style_paths):
                swaps.append(model(sps[i], gls[j], command="decode"))
                filenames.append(structure_path + "_" + style_path + ".png")
            self.add_to_webpage([structure_image] + swaps,
                                [structure_path + ".png"] + filenames,
                                tile=1)

            self.webpage.save()
        return {}
                
                
        
