import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath
import numpy as np
from model import resnet18
from functools import partial

#### ---- DEFORMABLE ATTENTION ---- ####

class DeformableAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_points=4, offset_range=3, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.num_points = num_points
        self.offset_range = offset_range

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.offsets = nn.Linear(dim, num_heads * num_points)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        offsets = self.offsets(x).reshape(B, L, self.num_heads, self.num_points)
        offsets = torch.tanh(offsets) * self.offset_range

        sampled_k = torch.zeros(B, self.num_heads, L, self.num_points, self.head_dim, device=x.device)
        sampled_v = torch.zeros(B, self.num_heads, L, self.num_points, self.head_dim, device=x.device)

        base_positions = torch.arange(L, device=x.device).view(1, 1, L, 1).repeat(B, self.num_heads, 1, self.num_points)
        sampling_positions = base_positions + offsets.permute(0,2,1,3)
        sampling_positions = sampling_positions.clamp(0, L - 1)

        pos_floor = sampling_positions.floor().long()
        pos_ceil = (pos_floor + 1).clamp(max=L-1)
        weight = sampling_positions - pos_floor.float()

        for i in range(self.num_points):
            k_floor = torch.gather(k, 2, pos_floor[:, :, :, i:i+1].expand(-1, -1, -1, self.head_dim))
            k_ceil = torch.gather(k, 2, pos_ceil[:, :, :, i:i+1].expand(-1, -1, -1, self.head_dim))
            v_floor = torch.gather(v, 2, pos_floor[:, :, :, i:i+1].expand(-1, -1, -1, self.head_dim))
            v_ceil = torch.gather(v, 2, pos_ceil[:, :, :, i:i+1].expand(-1, -1, -1, self.head_dim))
            w = weight[:, :, :, i:i+1]

            sampled_k[:, :, :, i, :] = k_floor * (1 - w) + k_ceil * w
            sampled_v[:, :, :, i, :] = v_floor * (1 - w) + v_ceil * w

        q = q.unsqueeze(-2)
        attn = (q * sampled_k).sum(-1) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn.unsqueeze(-1) * sampled_v).sum(-2)
        out = out.permute(0, 2, 1, 3).reshape(B, L, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

#### ---- LAYER SCALE ---- ####

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

#### ---- BLOCK CLASS ---- ####

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_patches,
                 mlp_ratio=4., qkv_bias=False, drop=0.0, attn_drop=0.,
                 init_values=None, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_deformable=True, deform_num_points=4, deform_offset_range=3):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)
        if use_deformable:
            self.attn = DeformableAttention(
                dim, num_heads=num_heads, num_points=deform_num_points,
                offset_range=deform_offset_range,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = Attention(dim, num_patches, num_heads=num_heads,
                                  qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

#### ---- POSITIONAL EMBEDDING AND SUPPORT ---- ####

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

class LayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)

#### ---- VISION TRANSFORMER (MODEL) ---- ####

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, nb_cls=80, img_size=[512, 32],
                 patch_size=[8, 32], embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.layer_norm = LayerNorm()
        self.patch_embed = resnet18.ResNet18(embed_dim)
        self.grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim),
                                      requires_grad=False)
        # use_deformable attention here
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, self.num_patches,
                  mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  use_deformable=True, deform_num_points=4, deform_offset_range=3)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        self.head = torch.nn.Linear(embed_dim, nb_cls)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def generate_span_mask(self, x, mask_ratio, max_span_length):
        N, L, D = x.shape
        mask = torch.ones(N, L, 1).to(x.device)
        span_length = int(L * mask_ratio)
        num_spans = span_length // max_span_length
        for i in range(num_spans):
            idx = torch.randint(L - max_span_length, (1,))
            mask[:, idx:idx + max_span_length, :] = 0
        return mask

    def random_masking(self, x, mask_ratio, max_span_length):
        mask = self.generate_span_mask(x, mask_ratio, max_span_length)
        x_masked = x * mask + (1 - mask) * self.mask_token
        return x_masked

    def forward(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False):
        x = self.layer_norm(x)
        x = self.patch_embed(x)
        b, c, w, h = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        if use_masking:
            x = self.random_masking(x, mask_ratio, max_span_length)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        x = self.layer_norm(x)
        return x

def create_model(nb_cls, img_size, **kwargs):
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=img_size,
                                 patch_size=(4, 64),
                                 embed_dim=768,
                                 depth=4,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 **kwargs)
    return model

