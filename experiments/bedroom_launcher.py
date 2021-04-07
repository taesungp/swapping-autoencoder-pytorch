from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="~/datasets/lsun/bedroom_train_lmdb",
            dataset_mode="lmdb",
            num_gpus=8, batch_size=32,
            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="scale_shortside_and_crop",
            load_size=256, crop_size=256,
            display_freq=1600, print_freq=480,
        ),

        return [
            opt.specify(
                name="bedroom_default",
                patch_use_aggregation=False,
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=True,
            evaluation_metrics="swap_visualization",
            evaluation_freq=50000,
        ) for opt in common_options]

    def test_options_fid(self):
        return []

    def test_options(self):
        common_options = self.options()

        return [opt.tag("fig4").specify(
            num_gpus=1,
            batch_size=1,
            dataroot="./testphotos/bedroom/fig4",
            dataset_mode="imagefolder",
            preprocess="scale_shortside",  # For testing, scale but don't crop
            evaluation_metrics="structure_style_grid_generation",
        ) for opt in common_options]
