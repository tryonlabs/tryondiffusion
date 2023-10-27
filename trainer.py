from diffusion import Diffusion


class ArgParser:

    def __init__(self):

        self.run_name = "1"

        self.train_ip_folder = "data/train/ip"
        self.train_jp_folder = "data/train/jp"
        self.train_ia_folder = "data/train/ia"
        self.train_ic_folder = "data/train/ic"

        self.validation_ip_folder = "data/validation/ip"
        self.validation_jp_folder = "data/validation/jp"
        self.validation_ia_folder = "data/validation/ia"
        self.validation_ic_folder = "data/validation/ic"

        self.batch_size_train = 8
        self.batch_size_validation = 1

        self.lr = 0.0001

        self.epochs = 100

        self.calculate_loss_frequency = 10
        self.image_logging_frequency = 10
        self.model_saving_frequency = 10


args = ArgParser()
diffusion = Diffusion(device="cuda", pose_embed_dim=16)

diffusion.prepare(args)
diffusion.fit(args)