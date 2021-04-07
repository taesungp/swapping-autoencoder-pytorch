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
                name="ffhq512_default",
            ),
            opt.specify(
                name="ffhq1024_default",
                preprocess="resize_and_crop",
                load_size=1024, crop_size=512 + 256,
                # Adjust the capacity of the networks to fit on 16GB GPUs
                netG_scale_capacity=0.8,
                netE_num_downsampling_sp=5,
                netE_scale_capacity=0.4,
                global_code_ch=1024 + 512,
                patch_size=256,
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=True,
            evaluation_metrics="swap_visualization",
            evaluation_freq=50000) for opt in common_options]
        
    def test_options(self):
        opts = self.options()
        return [
            # Fig 9 of Appendix
            opts[0].tag("ffhq512_fig9").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="./testphotos/ffhq512/fig9/",
                dataset_mode="imagefolder",
                evaluation_metrics="structure_style_grid_generation"
            ),
            # Fig 5 and Table 2 of Appendix
            opts[1].tag("swapping_for_eval").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="./testphotos/ffhq1024/fig5_tab2/",
                evaluation_metrics="swap_generation_from_arranged_result",
                preprocess="resize", load_size=1024, crop_size=1024,
            )
        ]
