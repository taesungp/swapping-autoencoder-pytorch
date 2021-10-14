from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="~/datasets/flickr/mountain/train",
            dataset_mode="imagefolder",
            checkpoints_dir="./checkpoints/",
            num_gpus=8, batch_size=16,
            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="scale_shortside_and_crop",
            load_size=512, crop_size=512,
            display_freq=1600, print_freq=480,
        )

        return [
            opt.specify(
                name="mountain_default",
                lambda_patch_R1=10.0,
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=True,
            evaluation_metrics="none") for opt in common_options]
        
    def test_options(self):
        opt = self.options()[0]
        return [
            # Swapping Grid Visualization. Fig 12 of the arxiv paper
            opt.tag("swapping_grid").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="~/datasets/testphotos/images_used_in_swapping_autoencoder_arxiv/mountain/fig12/",
                dataset_mode="imagefolder",
                preprocess="scale_width",  # For testing, scale but don't crop
                load_size=1024, crop_size=1024,
                evaluation_metrics="structure_style_grid_generation"
            ),
        ]
