from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            # I exported FFHQ dataset to 70,000 image files
            # and load them as images files.
            # Alternatively, the dataset can be prepared as
            # an LMDB dataset (like LSUN), and set dataset_mode = "lmdb".
            dataroot="~/datasets/ffhq/images1024x1024/",
            dataset_mode="imagefolder",
            checkpoints_dir="./checkpoints/",
            num_gpus=8, batch_size=16,
            preprocess="resize",
            load_size=512, crop_size=512,
        )

        return [
            opt.specify(
                name="ffhq512_pretrained",
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=True,
            evaluation_metrics="swap_visualization",
            evaluation_freq=50000) for opt in common_options]
        
    def test_options(self):
        opt = self.options()[0]
        return [
            # Fig 9 of Appendix
            opt.tag("swapping_grid").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="./testphotos/ffhq512/fig9/",
                dataset_mode="imagefolder",
                evaluation_metrics="structure_style_grid_generation"
            ),
        ]
