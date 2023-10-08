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


class SinusoidalPosEmbed(nn.Module):
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


class AttentionPool1d(nn.Module):

    def __init__(self, pose_embeb_dim, num_heads=1):
        """
        Clip inspired 1D attention pooling, reduce garment and person pose along keypoints
        :param pose_embeb_dim: pose embedding dimensions
        :param num_heads: number of attention heads
        """
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(3, pose_embeb_dim) / pose_embeb_dim ** 0.5)
        self.k_proj = nn.Linear(pose_embeb_dim, pose_embeb_dim)
        self.q_proj = nn.Linear(pose_embeb_dim, pose_embeb_dim)
        self.v_proj = nn.Linear(pose_embeb_dim, pose_embeb_dim)
        self.c_proj = nn.Linear(pose_embeb_dim, pose_embeb_dim)
        self.num_heads = num_heads

        self.pos_embed_time = SinusoidalPosEmbed(pose_embeb_dim)

    def forward(self, x, time_step=None):
        # if x in format NCP
        # N - Batch Dimension, P - Pose Dimension, C - No. of pose(person and garment)
        x = x.permute(1, 0, 2)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (C+1)NP
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (C+1)NP
        batch_size = x.shape[1]
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
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
        x = x.squeeze(0)
        # time step = torch.randint(0, 1000, (batch_size,), device=device).long()
        x += self.pos_embed_time(time_step).squeeze(1)
        return x


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


class SelfAttention(nn.Module):
    pass


class ResBlockNoAttention(nn.Module):

    def __init__(self, block_channel, clip_dim):
        super().__init__()

        self.film_generator_person_pose = FiLM(clip_dim, block_channel)

        self.gn1 = nn.GroupNorm(min(32, int(abs(block_channel / 4))), int(block_channel))
        self.swish1 = nn.SiLU(True)
        self.conv1 = nn.Conv2d(block_channel, block_channel, (3, 3), padding=1)
        self.gn2 = nn.GroupNorm(min(32, int(abs(block_channel / 4))), int(block_channel))
        self.swish2 = nn.SiLU(True)
        self.conv2 = nn.Conv2d(block_channel, block_channel, (3, 3), padding=1)

        self.conv_residual = nn.Conv2d(block_channel, block_channel, (3, 3), padding=1)

    def forward(self, x, clip_embeddings):

        x = self.film_generator_person_pose(clip_embeddings, x)

        h = self.gn1(x)
        h = self.swish1(h)
        h = self.conv1(h)
        h = self.gn2(h)
        h = self.swish2(h)
        h = self.conv2(h)

        h += self.conv_residual(x)

        return h


class UNetBlockNoAttention(nn.Module):

    def __init__(self, inp_channel, block_channel, clip_pooled_dim, sub_blocks_number):
        super().__init__()

        nn.ModuleList()



