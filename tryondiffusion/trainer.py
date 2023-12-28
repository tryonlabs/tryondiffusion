from .diffusion import Diffusion


class ArgParser:

    def __init__(self):

        self.run_name = "1"

        self.train_ip_folder = "data/test_flow/train/ip"
        self.train_jp_folder = "data/test_flow/train/jp"
        self.train_ia_folder = "data/test_flow/train/ia"
        self.train_ic_folder = "data/test_flow/train/ic"
        self.train_jg_folder = "data/test_flow/train/jg"

        self.validation_ip_folder = "data/test_flow/validation/ip"
        self.validation_jp_folder = "data/test_flow/validation/jp"
        self.validation_ia_folder = "data/test_flow/validation/ia"
        self.validation_ic_folder = "data/test_flow/validation/ic"
        self.validation_jg_folder = "data/test_flow/validation/jg"

        self.batch_size_train = 1
        self.batch_size_validation = 1

        self.lr = 0.0

        self.calculate_loss_frequency = 10
        self.image_logging_frequency = 10
        self.model_saving_frequency = 10

        self.total_steps = 100000
        self.start_lr = 0.0
        self.stop_lr = 0.0001
        self.pct_increasing_lr = 0.02


args = ArgParser()
diffusion = Diffusion(device="cuda",
                      pose_embed_dim=8,
                      time_steps=256,
                      beta_start=1e-4,
                      beta_end=0.02,
                      unet_dim=64,
                      noise_input_channel=3,
                      beta_ema=0.995)

diffusion.prepare(args)
diffusion.fit(args)
