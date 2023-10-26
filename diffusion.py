import torch


class Diffusion:

    def __init__(self, time_steps, beta_start, beta_end):
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.time_steps)

    def sample_time_steps(self, batch_size):
        return torch.randint(low=1, high=self.time_steps, size=(batch_size, ))

    # def add_noise_to_img(self, img, t):


