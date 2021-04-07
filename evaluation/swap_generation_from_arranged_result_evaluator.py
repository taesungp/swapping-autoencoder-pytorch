import glob
import torchvision.transforms as transforms
import os
import torch
from evaluation import BaseEvaluator
import util
from PIL import Image


class InputDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot):
        structure_images = sorted(glob.glob(os.path.join(dataroot, "input_structure", "*.png")))
        style_images = sorted(glob.glob(os.path.join(dataroot, "input_style", "*.png")))

        for structure_path, style_path in zip(structure_images, style_images):
            assert structure_path.replace("structure", "style") == style_path, \
                "%s and %s do not match" % (structure_path, style_path)

        assert len(structure_images) == len(style_images)
        print("found %d images at %s" % (len(structure_images), dataroot))

        self.structure_images = structure_images
        self.style_images = style_images
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def __len__(self):
        return len(self.structure_images)

    def __getitem__(self, idx):
        structure_image = self.transform(Image.open(self.structure_images[idx]).convert('RGB'))
        style_image = self.transform(Image.open(self.style_images[idx]).convert('RGB'))
        return {'structure': structure_image,
                'style': style_image,
                'path': self.structure_images[idx]}


class SwapGenerationFromArrangedResultEvaluator(BaseEvaluator):
    """ Given two directories containing input structure and style (texture)
    images, respectively, generate reconstructed and swapped images.
    The input directories should contain the same set of image filenames. 
    It differs from StructureStyleGridGenerationEvaluator, which creates
    N^2 outputs (i.e. swapping of all possible pairs between the structure and
    style images).
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def image_save_dir(self, nsteps):
        return os.path.join(self.output_dir(), "%s_%s" % (self.target_phase, nsteps), "images")

    def create_webpage(self, nsteps):
        if nsteps is None:
            nsteps = self.opt.resume_iter
        elif isinstance(nsteps, int):
            nsteps = str(round(nsteps / 1000)) + "k"
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

    def set_num_test_images(self, num_images):
        self.num_test_images = num_images

    def evaluate(self, model, dataset, nsteps=None):
        input_dataset = torch.utils.data.DataLoader(
            InputDataset(self.opt.dataroot),
            batch_size=1,
            shuffle=False, drop_last=False, num_workers=0
        )

        self.num_test_images = None
        self.create_webpage(nsteps)
        image_num = 0
        for i, data_i in enumerate(input_dataset):
            structure = data_i["structure"].cuda()
            style = data_i["style"].cuda()
            path = data_i["path"][0]
            path = os.path.basename(path)
            #if "real_B" in data_i:
            #    image = torch.cat([image, data_i["real_B"].cuda()], dim=0)
            #    paths = paths + data_i["path_B"]
            sp, gl = model(structure, command="encode")
            rec = model(sp, gl, command="decode")

            _, gl = model(style, command="encode")
            swapped = model(sp, gl, command="decode")

            self.add_to_webpage([structure, style, rec, swapped],
                                ["%s_structure.png" % (path),
                                 "%s_style.png" % (path),
                                 "%s_rec.png" % (path),
                                 "%s_swap.png" % (path)],
                                tile=1)
            image_num += 1
            if self.num_test_images is not None and self.num_test_images <= image_num:
                self.webpage.save()
                return {}
                    
            self.webpage.save()
        return {}
                
                
        
