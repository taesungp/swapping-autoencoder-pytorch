from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="~/datasets/lsun/bedroom_train_lmdb",
            checkpoints_dir="./checkpoints/",
            dataset_mode="lmdb",
            num_gpus=7, batch_size=56,
            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="scale_shortside_and_crop",
            load_size=256, crop_size=256,
            display_freq=1600, print_freq=480,
        )

        return [
            opt.specify(
                name="bedroom_pretrained",
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
            # Swapping Grid Visualization. Fig 4 of the arxiv paper
            opt.tag("swapping_grid").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="./testphotos/bedroom/fig4",
                dataset_mode="imagefolder",
                preprocess="scale_shortside",  # For testing, scale but don't crop
                evaluation_metrics="structure_style_grid_generation"
            ),
        ]
