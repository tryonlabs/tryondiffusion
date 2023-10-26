import copy
import logging

import torch

from network import UNet64, UNet128


class Diffusion:

    def __init__(self,
                 time_steps,
                 beta_start,
                 beta_end,
                 device,
                 pose_embed_dim,
                 time_dim=256,
                 unet_dim=64,
                 noise_input_channel=3):
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.linear_beta_scheduler().to(device)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

        self.noise_input_channel = noise_input_channel
        self.unet_dim = unet_dim
        if unet_dim == 128:
            self.net = UNet128(pose_embed_dim, time_dim).to(device)
        elif unet_dim == 64:
            self.net = UNet64(pose_embed_dim, time_dim).to(device)

        self.ema_net = copy.deepcopy(self.net).eval().requires_grad_(False)

        self.device = device

    def linear_beta_scheduler(self):
        return torch.linspace(self.beta_start, self.beta_end, self.time_steps)

    def sample_time_steps(self, batch_size):
        return torch.randint(low=1, high=self.time_steps, size=(batch_size, ))

    def add_noise_to_img(self, img, t):
        sqrt_alpha_timestep = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_timestep = torch.sqrt(1 - self.alpha_cumprod[t])[:, None, None, None]
        epsilon = torch.randn_like(img)
        return (sqrt_alpha_timestep * epsilon) + (sqrt_one_minus_alpha_timestep * epsilon), epsilon

    @torch.inference_mode()
    def sample(self, use_ema, conditional_inputs):
        model = self.ema_net if use_ema else self.net
        ic, jp, jg, ia = conditional_inputs

        batch_size = len(ic)
        logging.info(f"Running inference for {batch_size} images")

        model.eval()
        with torch.inference_mode():
            x = torch.randn(batch_size, self.noise_input_channel, self.unet_dim, self.unet_dim).to(self.device)
            # concatenating noise with rgb agnostic image across channels
            x = torch.cat((x, ia), dim=1)
            for i in reversed(range(1, self.time_steps)):
                t = (torch.ones(batch_size) * i).long().to(self.device)
                predicted_noise = model(x, ic, jp, jg, t)
                # ToDo: Add Classifier-Free Guidance
                alpha = self.alpha[t][:, None, None, None]
                alpha_cumprod = self.alpha_cumprod[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)
                return x



