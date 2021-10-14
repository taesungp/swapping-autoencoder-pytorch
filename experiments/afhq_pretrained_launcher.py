from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="/mnt/localssd/datasets/afhq/afhq/train",
            dataset_mode="imagefolder",
            checkpoints_dir="./checkpoints/",
            num_gpus=8, batch_size=32,
            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="resize",
            load_size=256, crop_size=256,
        )

        return [
            opt.specify(
                name="afhq_pretrained",
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            display_freq=1600, print_freq=480,
            continue_train=True,
            evaluation_metrics="none") for opt in common_options]
        
    def test_options(self):
        opt = self.options()[0]
        return [
            # Swapping Grid Visualization. Fig 12 of the arxiv paper
            opt.tag("swapping_grid").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="./testphotos/afhq/",
                dataset_mode="imagefolder",
                evaluation_metrics="structure_style_grid_generation"
            ),
            
            # Simple Swapping code for quick testing
            opt.tag("simple_swapping").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.
                evaluation_metrics="simple_swapping",
                # Specify the two images here.
                input_structure_image="./testphotos/afhq/structure/flickr_dog_000846.jpg",
                input_texture_image="./testphotos/afhq/style/flickr_wild_001319.jpg",
                # alpha == 1.0 corresponds to full swapping.
                # 0 < alpha < 1 means interpolation
                texture_mix_alpha=1.0,
            ),
            
            # Simple interpolation images for quick testing
            opt.tag("simple_interpolation").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.
                evaluation_metrics="simple_swapping",
                # Specify the two images here.
                input_structure_image="./testphotos/afhq/structure/flickr_dog_000846.jpg",
                input_texture_image="./testphotos/afhq/style/flickr_wild_001319.jpg",
                texture_mix_alpha='0.0 0.25 0.5 0.75 1.0',
            )
        ]
