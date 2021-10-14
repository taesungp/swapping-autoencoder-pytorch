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
            
            # Load images at 1024x1024 size, and make random 
            # crops of 768x768
            preprocess="resize_and_crop",
            load_size=1024, crop_size=512 + 256,
            
            # Adjust the capacity of the networks to fit on 16GB GPUs
            netG_scale_capacity=0.8,
            netE_num_downsampling_sp=5,
            netE_scale_capacity=0.4,
            global_code_ch=1024 + 512,
            patch_size=256,
        )

        return [
            opt.specify(
                name="ffhq1024_pretrained",
            ),
            opt.specify(
                name="ffhq1024_patchDv7_L1_smallerstructurecode",
                netE_num_downsampling_sp=6,
                netE_scale_capacity=1.0,
                netE_nc_steepness=1.8,
                global_code_ch=1024,
                spatial_code_ch=64,
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=True,
            evaluation_metrics="swap_visualization",
            evaluation_freq=50000) for opt in common_options]
        
    def test_options(self):
        opt = self.options()[1]
        return [
            # Fig 5 and Table 2 of Appendix
            # The selection of the images used for evaluation could not be
            # publicly shared due to copyright issue of redistribution.
            # We recommend randomly selecting the image pairs from the val set.
            # Alternatively, contact Taesung Park for the exact pairs of images.
            opt.tag("swapping_for_eval").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="~/datasets/testphotos/images_used_in_swapping_autoencoder_arxiv/ffhq1024/fig5_tab2/",
                #dataroot="./testphotos/ffhq1024/fig5_tab2/",
                evaluation_metrics="swap_generation_from_arranged_result",
                preprocess="resize", load_size=1024, crop_size=1024,
            )
        ]
