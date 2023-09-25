import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import math


class DownSample(nn.Module):

    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim * 4, dim_out, (1, 1))

    def forward(self, x):
        # slicing image into four pieces across h and w and appending pieces to channels
        # new_no_channel = c * 4, conserving the features instead of pooling.
        x = rearrange(x, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        x = self.conv(x)
        return x


class UpSample(nn.Module):

    def __init__(self, dim, dim_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim_out, (3, 3), padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class SinusodalPosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim//2
        pos_embed = math.log(10000)/(half_dim - 1)
        pos_embed = torch.exp(torch.arange(half_dim, device=device) * -pos_embed)
        pos_embed = t[:, None] * pos_embed[None, :]
        pos_embed = torch.cat((pos_embed.sin(), pos_embed.cos()), dim=-1)
        return pos_embed


class AttentionPool1D(nn.Module):

    def __init__(self, pose_embeb_dim, num_heads=2, output_dim=None):
        """
        Clip inspired 1D attention pooling
        :param pose_embeb_dim:
        :param num_heads:
        :param output_dim:
        """
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(2, pose_embeb_dim) / pose_embeb_dim ** 0.5)
        self.k_proj = nn.Linear(pose_embeb_dim, pose_embeb_dim)
        self.q_proj = nn.Linear(pose_embeb_dim, pose_embeb_dim)
        self.v_proj = nn.Linear(pose_embeb_dim, pose_embeb_dim)
        self.c_proj = nn.Linear(pose_embeb_dim, output_dim or pose_embeb_dim)
        self.num_heads = num_heads

        self.pos_embed_time = SinusodalPosEmbed(output_dim or pose_embeb_dim)

    def forward(self, x, time_step):
        # if x in format NP
        # N - Batch Dimension, P - Pose Dimension
        x = x[None, :, :]  # NN -> 1NP
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # 2NP
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # 2NP
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=8,
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        x += self.pos_embed_time(time_step)
        return x.squeeze(0)


class FiLM(nn.Module):

    def __init__(self, clip_dim, channels):
        super().__init__()
        self.channels = channels

        self.fc = nn.Linear(clip_dim, 2 * channels)
        self.activation = nn.ReLU(True)

    def forward(self, clip_pooled_embed, img_embed):
        clip_pooled_embed = self.fc(clip_pooled_embed)
        clip_pooled_embed = self.activation(clip_pooled_embed)
        gamma = clip_pooled_embed[:, 0:self.channels]
        beta = clip_pooled_embed[:, self.channels:self.channels + 1]
        film_features = torch.add(torch.mul(img_embed, gamma[:, :, None, None]), beta[:, :, None, None])
        return film_features


class ResBlockNoAttention(nn.Module):

    def __init__(self, inp_channel, block_channel, clip_pooled_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(inp_channel, block_channel, (3, 3), padding=1)

        self.gn1 = nn.GroupNorm(min(32, int(abs(block_channel / 4))), int(block_channel))
        self.swish1 = nn.SiLU(True)
        self.conv1 = nn.Conv2d(block_channel, block_channel, (3, 3), padding=1)
        self.gn2 = nn.GroupNorm(min(32, int(abs(block_channel / 4))), int(block_channel))
        self.swish2 = nn.SiLU(True)
        self.conv2 = nn.Conv2d(block_channel, block_channel, (3, 3), padding=1)

        self.film_generator_person_pose = FiLM(clip_pooled_dim, block_channel)
        self.film_generator_garment_pose = FiLM(clip_pooled_dim, block_channel)

    def forward(self, x, clip_embeddings_person, clip_embeddings_garment):
        residual = self.conv0(x)

        x = self.gn1(residual)
        x = self.swish1(x)
        x = self.conv1(x)
        x = self.gn2(x)
        x = self.swish2(x)
        x = self.conv2(x)

        x = self.film_generator_person_pose(clip_embeddings_person, x)
        x = self.film_generator_garment_pose(clip_embeddings_garment, x)

        x += residual

        return x


