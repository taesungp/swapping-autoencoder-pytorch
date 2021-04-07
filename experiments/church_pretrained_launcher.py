from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="~/datasets/lsun/church_outdoor_train_lmdb",
            dataset_mode="lmdb",
            checkpoints_dir="./checkpoints/",
            num_gpus=8, batch_size=32,
            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="scale_shortside_and_crop",
            load_size=256, crop_size=256,
            display_freq=1600, print_freq=480,
        )

        return [
            opt.specify(
                name="church_pretrained",
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
                dataroot="./testphotos/church/fig4",
                dataset_mode="imagefolder",
                preprocess="scale_shortside",  # For testing, scale but don't crop
                evaluation_metrics="structure_style_grid_generation"
            ),

            # Fig 5 and Table 2 of Appendix
            # The selection of the images used for evaluation could not be
            # publicly shared due to copyright issue of redistribution.
            # We recommend randomly selecting the image pairs from the val set.
            # Alternatively, contact Taesung Park for the exact pairs of images.
            opt.tag("swapping_for_eval").specify(
                dataroot="./testphotos/church/fig5_tab2/",
                dataset_mode="imagefolder",
                num_gpus=1, batch_size=1,
                evaluation_metrics="swap_generation_from_arranged_result",
                preprocess="resize",
            )
        ]
