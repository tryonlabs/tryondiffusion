import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

import math

from .utils import GaussianSmoothing


class DownSample(nn.Module):

    def __init__(self, dim, dim_out, t_emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(dim * 4, dim_out, (1, 1))

        # positional embedding
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                t_emb_dim,
                dim_out
            ),
        )

    def forward(self, x, t):
        # slicing image into four pieces across h and w and appending pieces to channels
        # new_no_channel = c * 4, conserving the features instead of pooling.
        x = rearrange(x, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        x = self.conv(x)
        # calculate positional embedding from positional encoding t
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpSample(nn.Module):

    def __init__(self, dim, dim_out, t_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim_out, (3, 3), padding=1)

        # positional embedding
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                t_emb_dim,
                dim_out
            ),
        )

    def forward(self, x, t):
        x = self.up(x)
        x = self.conv(x)
        # calculate positional embedding from positional encoding t
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        pos_embed = math.log(10000) / (half_dim - 1)
        pos_embed = torch.exp(torch.arange(half_dim, device=device) * -pos_embed)
        pos_embed = t[:, None] * pos_embed[None, :]
        pos_embed = torch.cat((pos_embed.sin(), pos_embed.cos()), dim=-1)
        return pos_embed


class AttentionPool1d(nn.Module):

    def __init__(self, pose_embeb_dim, device, num_heads=1):
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

        self.pos_encoding = SinusoidalPosEmbed(pose_embeb_dim)

        self.device = device

    def forward(self, x, time_step, noise_level=0.3):
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
        # x = x.squeeze(0)
        x = x.permute(1, 0, 2)

        smoothing = GaussianSmoothing(channels=1,
                                      kernel_size=3,
                                      sigma=noise_level,
                                      conv_dim=1).to(self.device)

        x_noisy = smoothing(F.pad(x, (1, 1), mode='reflect'))

        x = x + x_noisy

        x += self.pos_encoding(time_step)[:, None, :]
        return x.squeeze(1)


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


def l2norm(t):
    return F.normalize(t, dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, feats, dim=-1):
        super().__init__()
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=dim, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=dim, keepdim=True)
        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=1,
            heads=4,
            pose_dim=None,
            scale=1):
        """
        Intialize: att = SelfAttention(1024, 1, pose_dim=16)
        Execute: att(torch.randn(4, 256, 1024), pose_embed=torch.randn(4, 2, 16)).size()
        :param dim:
        :param dim_head:
        :param heads:
        :param pose_dim:
        :param scale:
        """
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        # self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_context = nn.Sequential(nn.LayerNorm(pose_dim), nn.Linear(pose_dim, dim_head * 2))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x, pose_embed=None):
        b, n, device = *x.shape[:2], x.device
        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        # add null key / value for classifier free guidance in prior net
        # nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b=b), self.null_kv.unbind(dim=-2))
        # k = torch.cat((nk, k), dim=-2)
        # v = torch.cat((nv, v), dim=-2)

        # add pose conditioning
        ck, cv = self.to_context(pose_embed).chunk(2, dim=-1)
        k = torch.cat((ck, k), dim=-2)
        v = torch.cat((cv, v), dim=-2)

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(
            self,
            zt_dim,
            ic_dim,
            dim_head=1,
            heads=4,
            scale=1
    ):

        super().__init__()

        try:
            assert zt_dim == ic_dim
        except AssertionError:
            print(
                f"Expecting same dimension for zt and ic, but received {zt_dim} and {ic_dim}, indicating mismatch in "
                f"parallel blocks of both UNets")

        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_zt = LayerNorm(zt_dim)
        self.norm_ic = LayerNorm(ic_dim)

        # self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(zt_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(ic_dim, dim_head * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, zt_dim, bias=False),
            LayerNorm(zt_dim)
        )

    def forward(self, zt, ic):
        # b and n will be same for both zt and ic
        b, n, device = *zt.shape[:2], zt.device

        ic = self.norm_ic(ic)
        zt = self.norm_zt(zt)

        q, k, v = (self.to_q(zt), *self.to_kv(ic).chunk(2, dim=-1))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        # add null key / value for classifier free guidance in prior net
        # nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b=b), self.null_kv.unbind(dim=-2))
        # k = torch.cat((nk, k), dim=-2)
        # v = torch.cat((nv, v), dim=-2)

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ResBlockNoAttention(nn.Module):

    def __init__(self, block_channel, clip_dim, input_channel=None):
        super().__init__()

        self.film = FiLM(clip_dim, block_channel)

        if input_channel is None:
            input_channel = block_channel

        self.gn1 = nn.GroupNorm(min(32, int(abs(input_channel / 4))), int(input_channel))
        self.swish1 = nn.SiLU(True)
        self.conv1 = nn.Conv2d(input_channel, block_channel, (3, 3), padding=1)

        self.gn2 = nn.GroupNorm(min(32, int(abs(block_channel / 4))), int(block_channel))
        self.swish2 = nn.SiLU(True)
        self.conv2 = nn.Conv2d(block_channel, block_channel, (3, 3), padding=1)

        self.conv_residual = nn.Conv2d(input_channel, block_channel, (3, 3), padding=1)

    def forward(self, x, clip_embeddings, unet_residual=None):

        x = self.film(clip_embeddings, x)

        if unet_residual is not None:
            x = torch.concat((x, unet_residual), dim=1)

        h = self.gn1(x)
        h = self.swish1(h)
        h = self.conv1(h)
        h = self.gn2(h)
        h = self.swish2(h)
        h = self.conv2(h)

        h += self.conv_residual(x)

        return h


class ResBlockAttention(nn.Module):

    def __init__(self, block_channel, clip_dim, hw_dim, input_channel=None):
        super().__init__()

        self.block_channel = block_channel
        self.hw_dim = hw_dim

        self.film = FiLM(clip_dim, block_channel)

        if input_channel is None:
            input_channel = block_channel

        self.gn1 = nn.GroupNorm(min(32, int(abs(input_channel / 4))), int(input_channel))
        self.swish1 = nn.SiLU(True)
        self.conv1 = nn.Conv2d(input_channel, block_channel, (3, 3), padding=1)
        self.gn2 = nn.GroupNorm(min(32, int(abs(block_channel / 4))), int(block_channel))
        self.swish2 = nn.SiLU(True)
        self.conv2 = nn.Conv2d(block_channel, block_channel, (3, 3), padding=1)

        self.conv_residual = nn.Conv2d(input_channel, block_channel, (3, 3), padding=1)

        att_dim = (hw_dim // 2) ** 2
        # pose vector length will be same as clip pooled length
        self.self_attention = SelfAttention(dim=att_dim, pose_dim=clip_dim)

        self.cross_attention = CrossAttention(zt_dim=att_dim, ic_dim=att_dim)

    def forward(self, x, clip_embeddings, pose_embed, garment_embed, unet_residual=None):
        x = self.film(clip_embeddings, x)

        if unet_residual is not None:
            x = torch.concat((x, unet_residual), dim=1)

        h = self.gn1(x)
        h = self.swish1(h)
        h = self.conv1(h)
        h = self.gn2(h)
        h = self.swish2(h)
        h = self.conv2(h)

        h += self.conv_residual(x)

        # self attention with pose embed concat with key value
        h = rearrange(h, "b c (h p1) (w p2) -> b (c p1 p2) (h w)", p1=2, p2=2)
        h = self.self_attention(h, pose_embed)

        # cross attention with query from zt and key value from ic
        garment_embed = rearrange(garment_embed, "b c (h p1) (w p2) -> b (c p1 p2) (h w)", p1=2, p2=2)
        h = self.cross_attention(h, garment_embed)

        h = rearrange(h, "b (c p1 p2) (h w) -> b c (h p1) (w p2)",
                      c=self.block_channel, p1=2, p2=2, h=self.hw_dim // 2, w=self.hw_dim // 2)

        return h


class UNetBlockNoAttention(nn.Module):

    def __init__(self, block_channel, clip_dim, res_blocks_number, input_channel=None):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for idx, block in enumerate(range(res_blocks_number)):
            if idx != 0:
                input_channel = None
            self.blocks.append(ResBlockNoAttention(block_channel, clip_dim, input_channel=input_channel))

    def forward(self, x, clip_embeddings, unet_residual=None):
        for idx, block in enumerate(self.blocks):
            if idx != 0:
                unet_residual=None
            x = block(x, clip_embeddings, unet_residual)
        return x


class UNetBlockAttention(nn.Module):

    def __init__(self,
                 block_channel,
                 clip_dim,
                 res_blocks_number,
                 hw_dim,
                 input_channel=None):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for idx, block in enumerate(range(res_blocks_number)):
            if idx != 0:
                input_channel = None
            self.blocks.append(ResBlockAttention(block_channel,
                                                 clip_dim,
                                                 hw_dim,
                                                 input_channel=input_channel))

    def forward(self, x, clip_embeddings, pose_embed, garment_embed, unet_residual=None):
        for idx, block in enumerate(self.blocks):
            if idx != 0:
                unet_residual = None
            x = block(x, clip_embeddings, pose_embed, garment_embed, unet_residual)
        return x


class UNet128(nn.Module):

    def __init__(self, pose_embed_len_dim, device, time_dim=256):
        super().__init__()

        self.pos_encod_layer = SinusoidalPosEmbed(time_dim)

        # process clip embeddings
        self.attn_pool_layer = AttentionPool1d(pose_embed_len_dim, device)

        # =========================person UNet================================
        # initial image embedding size 128x128 and 6 channels(concatenate
        # rgb agnostic image and noise)

        # 3x3 conv layer person
        self.init_conv_person = nn.Conv2d(6, 128, (3, 3), padding=1)

        # 128 unit encoder person
        self.block1_person = UNetBlockNoAttention(block_channel=128,
                                                  clip_dim=pose_embed_len_dim,
                                                  res_blocks_number=3)
        self.downsample1_person = DownSample(dim=128, dim_out=256, t_emb_dim=time_dim)

        # 64 unit encoder person
        self.block2_person = UNetBlockNoAttention(block_channel=256,
                                                  clip_dim=pose_embed_len_dim,
                                                  res_blocks_number=4)
        self.downsample2_person = DownSample(dim=256, dim_out=512, t_emb_dim=time_dim)

        # 32 unit encoder person
        self.block3_person = UNetBlockAttention(block_channel=512,
                                                clip_dim=pose_embed_len_dim,
                                                res_blocks_number=6,
                                                hw_dim=32)
        self.downsample3_person = DownSample(dim=512, dim_out=1024, t_emb_dim=time_dim)

        # 16 unit encoder person
        self.block4_person = UNetBlockAttention(block_channel=1024,
                                                clip_dim=pose_embed_len_dim,
                                                res_blocks_number=7,
                                                hw_dim=16)

        # 16 unit decoder person
        self.block5_person = UNetBlockAttention(block_channel=1024,
                                                clip_dim=pose_embed_len_dim,
                                                res_blocks_number=7,
                                                hw_dim=16,
                                                input_channel=1024*2)
        self.upsample1_person = UpSample(dim=1024, dim_out=512, t_emb_dim=time_dim)

        # 32 unit decoder person
        self.block6_person = UNetBlockAttention(block_channel=512,
                                                clip_dim=pose_embed_len_dim,
                                                res_blocks_number=6,
                                                hw_dim=32,
                                                input_channel=512*2)
        self.upsample2_person = UpSample(dim=512, dim_out=256, t_emb_dim=time_dim)

        # 64 unit decoder person
        self.block7_person = UNetBlockNoAttention(block_channel=256,
                                                  clip_dim=pose_embed_len_dim,
                                                  res_blocks_number=4,
                                                  input_channel=256*2)
        self.upsample3_person = UpSample(dim=256, dim_out=128, t_emb_dim=time_dim)

        # 128 unit decoder person
        self.block8_person = UNetBlockNoAttention(block_channel=128,
                                                  clip_dim=pose_embed_len_dim,
                                                  res_blocks_number=3,
                                                  input_channel=128*2)

        self.final_conv_person = nn.Conv2d(128, 3, (3, 3), padding=1)
        # ==========================person UNet ends==========================

        # ==========================garment UNet==============================
        # initial image embedding size 128x128 and 3 channels
        self.init_conv_garment = nn.Conv2d(3, 128, (3, 3), padding=1)

        # 128 unit encoder garment
        self.block1_garment = UNetBlockNoAttention(block_channel=128,
                                                   clip_dim=pose_embed_len_dim,
                                                   res_blocks_number=3)
        self.downsample1_garment = DownSample(dim=128, dim_out=256, t_emb_dim=time_dim)

        # 64 unit encoder garment
        self.block2_garment = UNetBlockNoAttention(block_channel=256,
                                                   clip_dim=pose_embed_len_dim,
                                                   res_blocks_number=4)
        self.downsample2_garment = DownSample(dim=256, dim_out=512, t_emb_dim=time_dim)

        # 32 unit encoder garment
        self.block3_garment = UNetBlockNoAttention(block_channel=512,
                                                   clip_dim=pose_embed_len_dim,
                                                   res_blocks_number=6)
        self.downsample3_garment = DownSample(dim=512, dim_out=1024, t_emb_dim=time_dim)

        # 16 unit encoder garment
        self.block4_garment = UNetBlockNoAttention(block_channel=1024,
                                                   clip_dim=pose_embed_len_dim,
                                                   res_blocks_number=7)

        # 16 unit decoder garment
        self.block5_garment = UNetBlockNoAttention(block_channel=1024,
                                                   clip_dim=pose_embed_len_dim,
                                                   res_blocks_number=7,
                                                   input_channel=1024*2)
        self.upsample1_garment = UpSample(dim=1024, dim_out=512, t_emb_dim=time_dim)

        # 32 unit decoder garment
        self.block6_garment = UNetBlockNoAttention(block_channel=512,
                                                   clip_dim=pose_embed_len_dim,
                                                   res_blocks_number=6,
                                                   input_channel=512*2)
        # ==========================garment UNet ends==============================

    def forward(self, zt, ic, person_pose_embedding, garment_pose_embedding, time_step, noise_level):
        """

        :param zt: rgb_agnostic and noisy ground truth concatenated across channels
        :param ic: segmented garment
        :param person_pose_embedding: [b, 1, vector_length]
        :param garment_pose_embedding: [b, 1, vector_length]
        :param time_step:
        :param noise_level: sigma sampled from uniform distribution
        :return:
        """

        pos_encoding = self.pos_encod_layer(time_step)

        # concat pose embeddings
        pose_embeddings = torch.concat((person_pose_embedding[:, None, :], garment_pose_embedding[:, None, :]), dim=1)

        # get clip embeddings
        clip_embedding = self.attn_pool_layer(pose_embeddings, time_step, noise_level)

        # passing through initial conv layer
        ic = self.init_conv_garment(ic)
        zt = self.init_conv_person(zt)

        # entering UNet
        # 128 - garment
        ic = self.block1_garment(ic, clip_embedding)

        # 128 - person
        zt_128_1 = self.block1_person(zt, clip_embedding)

        # 64 - garment
        ic = self.downsample1_garment(ic, pos_encoding)
        ic = self.block2_garment(ic, clip_embedding)

        # 64 - person
        zt = self.downsample1_person(zt_128_1, pos_encoding)
        zt_64_1 = self.block2_person(zt, clip_embedding)

        # 32 - garment
        ic = self.downsample2_garment(ic, pos_encoding)
        ic_32_1 = self.block3_garment(ic, clip_embedding)

        # 32 - person
        zt = self.downsample2_person(zt_64_1, pos_encoding)
        zt_32_1 = self.block3_person(zt, clip_embedding, pose_embeddings, ic_32_1)
        # print(f"zt 32 encoder {zt_32_1.size()}")

        # 16 - garment
        ic = self.downsample3_garment(ic_32_1, pos_encoding)
        ic_16_1 = self.block4_garment(ic, clip_embedding)

        # 16 - person
        zt = self.downsample3_person(zt_32_1, pos_encoding)
        zt_16_1 = self.block4_person(zt, clip_embedding, pose_embeddings, ic_16_1)

        # 16 - garment
        ic_16_2 = self.block5_garment(ic_16_1, clip_embedding, ic_16_1)
        # print(f"ic 16 decoder {ic_16_2.size()}")

        # 16 - person
        zt_16_2 = self.block5_person(zt_16_1, clip_embedding, pose_embeddings, ic_16_2, zt_16_1)
        # print(f"zt 16 decoder {zt_16_2.size()}")

        # 32 - garment
        ic = self.upsample1_garment(ic_16_2, pos_encoding)
        ic_32_2 = self.block6_garment(ic, clip_embedding, ic_32_1)
        # print(f"ic 32 decoder {ic_32_2.size()}")

        # 32 - person
        zt = self.upsample1_person(zt_16_2, pos_encoding)
        zt_32_2 = self.block6_person(zt, clip_embedding, pose_embeddings, ic_32_2, zt_32_1)
        # print(f"zt 32 decoder {zt_32_2.size()}")

        # 64 - person
        zt = self.upsample2_person(zt_32_2, pos_encoding)
        zt_64_2 = self.block7_person(zt, clip_embedding, zt_64_1)

        # 128 - person
        zt = self.upsample3_person(zt_64_2, pos_encoding)
        zt_128_2 = self.block8_person(zt, clip_embedding, zt_128_1)

        # final conv layer - person
        zt = self.final_conv_person(zt_128_2)

        return zt


class UNet64(nn.Module):

    def __init__(self, pose_embed_len_dim, device, time_dim=256):
        super().__init__()

        self.pos_encod_layer = SinusoidalPosEmbed(time_dim)

        # process clip embeddings
        self.attn_pool_layer = AttentionPool1d(pose_embed_len_dim, device)

        # =========================person UNet================================
        # initial image embedding size 128x128 and 6 channels(concatenate
        # rgb agnostic image and noise)

        # 3x3 conv layer person
        self.init_conv_person = nn.Conv2d(6, 256, (3, 3), padding=1)

        # 64 unit encoder person
        self.block2_person = UNetBlockNoAttention(block_channel=256,
                                                  clip_dim=pose_embed_len_dim,
                                                  res_blocks_number=4)
        self.downsample2_person = DownSample(dim=256, dim_out=512, t_emb_dim=time_dim)

        # 32 unit encoder person
        self.block3_person = UNetBlockAttention(block_channel=512,
                                                clip_dim=pose_embed_len_dim,
                                                res_blocks_number=6,
                                                hw_dim=32)
        self.downsample3_person = DownSample(dim=512, dim_out=1024, t_emb_dim=time_dim)

        # 16 unit encoder person
        self.block4_person = UNetBlockAttention(block_channel=1024,
                                                clip_dim=pose_embed_len_dim,
                                                res_blocks_number=7,
                                                hw_dim=16)

        # 16 unit decoder person
        self.block5_person = UNetBlockAttention(block_channel=1024,
                                                clip_dim=pose_embed_len_dim,
                                                res_blocks_number=7,
                                                hw_dim=16,
                                                input_channel=1024*2)
        self.upsample1_person = UpSample(dim=1024, dim_out=512, t_emb_dim=time_dim)

        # 32 unit decoder person
        self.block6_person = UNetBlockAttention(block_channel=512,
                                                clip_dim=pose_embed_len_dim,
                                                res_blocks_number=6,
                                                hw_dim=32,
                                                input_channel=512*2)
        self.upsample2_person = UpSample(dim=512, dim_out=256, t_emb_dim=time_dim)

        # 64 unit decoder person
        self.block7_person = UNetBlockNoAttention(block_channel=256,
                                                  clip_dim=pose_embed_len_dim,
                                                  res_blocks_number=4,
                                                  input_channel=256*2)

        self.final_conv_person = nn.Conv2d(256, 3, (3, 3), padding=1)
        # ==========================person UNet ends==========================

        # ==========================garment UNet==============================
        # initial image embedding size 128x128 and 3 channels
        self.init_conv_garment = nn.Conv2d(3, 256, (3, 3), padding=1)

        # 64 unit encoder garment
        self.block2_garment = UNetBlockNoAttention(block_channel=256,
                                                   clip_dim=pose_embed_len_dim,
                                                   res_blocks_number=4)
        self.downsample2_garment = DownSample(dim=256, dim_out=512, t_emb_dim=time_dim)

        # 32 unit encoder garment
        self.block3_garment = UNetBlockNoAttention(block_channel=512,
                                                   clip_dim=pose_embed_len_dim,
                                                   res_blocks_number=6)
        self.downsample3_garment = DownSample(dim=512, dim_out=1024, t_emb_dim=time_dim)

        # 16 unit encoder garment
        self.block4_garment = UNetBlockNoAttention(block_channel=1024,
                                                   clip_dim=pose_embed_len_dim,
                                                   res_blocks_number=7)

        # 16 unit decoder garment
        self.block5_garment = UNetBlockNoAttention(block_channel=1024,
                                                   clip_dim=pose_embed_len_dim,
                                                   res_blocks_number=7,
                                                   input_channel=1024*2)
        self.upsample1_garment = UpSample(dim=1024, dim_out=512, t_emb_dim=time_dim)

        # 32 unit decoder garment
        self.block6_garment = UNetBlockNoAttention(block_channel=512,
                                                   clip_dim=pose_embed_len_dim,
                                                   res_blocks_number=6,
                                                   input_channel=512*2)
        # ==========================garment UNet ends==============================

    def forward(self, zt, ic, person_pose_embedding, garment_pose_embedding, time_step, noise_level):
        """

        :param zt:
        :param ic:
        :param person_pose_embedding: [b, 1, vector_length]
        :param garment_pose_embedding: [b, 1, vector_length]
        :param time_step:
        :return:
        """

        pos_encoding = self.pos_encod_layer(time_step)

        # concat pose embeddings
        pose_embeddings = torch.concat((person_pose_embedding[:, None, :], garment_pose_embedding[:, None, :]), dim=1)

        # get clip embeddings
        clip_embedding = self.attn_pool_layer(pose_embeddings, time_step, noise_level)

        # passing through initial conv layer
        ic = self.init_conv_garment(ic)
        zt = self.init_conv_person(zt)

        # 64 - garment
        # ic = self.downsample1_garment(ic, pos_encoding)
        ic = self.block2_garment(ic, clip_embedding)

        # 64 - person
        # zt = self.downsample1_person(zt, pos_encoding)
        zt_64_1 = self.block2_person(zt, clip_embedding)

        # 32 - garment
        ic = self.downsample2_garment(ic, pos_encoding)
        ic_32_1 = self.block3_garment(ic, clip_embedding)

        # 32 - person
        zt = self.downsample2_person(zt_64_1, pos_encoding)
        zt_32_1 = self.block3_person(zt, clip_embedding, pose_embeddings, ic_32_1)

        # 16 - garment
        ic = self.downsample3_garment(ic_32_1, pos_encoding)
        ic_16_1 = self.block4_garment(ic, clip_embedding)

        # 16 - person
        zt = self.downsample3_person(zt_32_1, pos_encoding)
        zt_16_1 = self.block4_person(zt, clip_embedding, pose_embeddings, ic_16_1)

        # 16 - garment
        ic_16_2 = self.block5_garment(ic_16_1, clip_embedding, ic_16_1)

        # 16 - person
        zt_16_2 = self.block5_person(zt_16_1, clip_embedding, pose_embeddings, ic_16_2, zt_16_1)

        # 32 - garment
        ic = self.upsample1_garment(ic_16_2, pos_encoding)
        ic_32_2 = self.block6_garment(ic, clip_embedding, ic_32_1)
        # print(f"ic 32 decoder {ic_32_2.size()}")

        # 32 - person
        zt = self.upsample1_person(zt_16_2, pos_encoding)
        zt_32_2 = self.block6_person(zt, clip_embedding, pose_embeddings, ic_32_2, zt_32_1)
        # print(f"zt 32 decoder {zt_32_2.size()}")

        # 64 - person
        zt = self.upsample2_person(zt_32_2, pos_encoding)
        zt_64_2 = self.block7_person(zt, clip_embedding, zt_64_1)
        # print(f"zt 64 decoder {zt_64_2.size()}")

        # final conv person
        zt = self.final_conv_person(zt_64_2)

        return zt


if __name__ == "__main__":

    time_step = torch.randint(low=1, high=1000, size=(4,))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net = UNet64(8, device).to(device)
    out = net(torch.randn(4, 6, 64, 64).to(device),
              torch.randn(4, 3, 64, 64).to(device),
              torch.randn(4, 8).to(device),
              torch.randn(4, 8).to(device),
              time_step.to(device),
              0.3)

    print(out.size())

    # att = AttentionPool1d(16)
    # time_step = torch.randint(0, 1000, (4,)).long()
    # out = att(torch.randn(4, 2, 16), time_step, None)
    # print(out.size())
