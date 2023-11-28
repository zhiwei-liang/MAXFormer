import math

import torch
import torch.nn as nn
import torch
from timm.models.layers import DropPath
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class External_Attention(nn.Module):
    def __init__(self, dim):
        super(External_Attention, self).__init__()
        self.num_heads = 8
        self.coef = 4
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, x):
        if len(x.shape) == 4:
            B, C, _, _ = x.shape
            x = x.permute(0, 2, 3, 1).view(B, -1, C)

        B, N, C = x.shape
        x = self.query_liner(x)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  #(1, 32, 225, 32)

        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))

        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)
        return x


class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, height, width, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width

        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

    # x2 should be higher-level representation than x1
    def forward(self, x1, x2):
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)

        # Re-arrange into a (Batch, Embedding dim, Tokens)
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = context.transpose(1, 2) @ query  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, self.height, self.width)
        reprojected_value = self.reprojection(aggregated_values).reshape(B, 2 * D, N).permute(0, 2, 1)
        reprojected_value = self.norm(reprojected_value)

        return reprojected_value

class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)

class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out

class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out

class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out

class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """
    def __init__(self, in_dim, key_dim, value_dim, height, width, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.H = height
        self.W = width
        self.channel_attn1 = ChannelAttention(in_dim)
        self.channel_attn2 = ChannelAttention(in_dim)
        self.attn = Cross_Attention(key_dim, value_dim, height, width, head_count=head_count)
        self.norm2 = nn.LayerNorm((in_dim * 2))
        if token_mlp == "mix":
            self.mlp = MixFFN((in_dim * 2), int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip((in_dim * 2), int(in_dim * 4))
        else:
            self.mlp = MLP_FFN((in_dim * 2), int(in_dim * 4))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        norm_1 = self.norm1(x1)
        channel_attn1 = self.channel_attn1(norm_1)
        channel1 = x1+channel_attn1

        norm_2 = self.norm1(x2)
        channel_attn2 = self.channel_attn1(norm_2)
        channel2 = x2 + channel_attn2

        norm_11 = self.norm3(channel1)
        norm_22 = self.norm3(channel2)
        attn = self.attn(norm_11, norm_22)

        residual = torch.cat([channel1, channel2], dim=2)
        tx = residual + attn
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W)
        return mx

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class PatchMerging(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W

class ConvFFN(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, stride,
                 out_channels, act_layer=nn.GELU, drop_out=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.act = act_layer()
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride,
                                kernel_size//2, groups=hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)
        self.drop = nn.Dropout(drop_out)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MyLayer(nn.Module):
    # parallel local-global attention mechanism
    def __init__(self, layer_dim, window_size = 7, dim_head = 32,
        mbconv_expansion_rate = 4, mbconv_shrinkage_rate = 0.25, dropout = 0.1,):
        super().__init__()
        self.conv = MBConv(
                    layer_dim,
                    layer_dim,
                    downsample=False,
                    expansion_rate=mbconv_expansion_rate,
                    shrinkage_rate=mbconv_shrinkage_rate
                )
        self.window_size = window_size
        w = window_size
        self.block_attn = Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)
        self.grid_attn = Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)

        self.proj = nn.Conv2d(2*layer_dim, layer_dim, 1, 1, 0, bias=True)
        self.proj_drop = nn.Dropout(dropout)
        self.drop_path = DropPath(dropout)

        self.ity = nn.Identity()

        self.mlp = ConvFFN(in_channels=layer_dim, hidden_channels=layer_dim*4, kernel_size=5, stride=1, out_channels=layer_dim, drop_out=dropout)


    def forward(self, x: torch.Tensor):
        res = []
        x = self.conv(x)
        w = self.window_size
        global_x = Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)(x)
        global_x = self.grid_attn(global_x)
        global_x = Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)')(global_x)
        res.append(global_x)

        local_x = Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w)(x)
        local_x = self.block_attn(local_x)
        local_x = Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)')(local_x)
        res.append(local_x)

        out = self.proj_drop(self.proj(torch.cat(res, dim=1)))

        out = x + self.drop_path(out)

        out = self.ity(x) + self.drop_path(self.mlp(out))

        return out

class MAXTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """
    def __init__(self, layer_depth, layer_dim, image_size, window_size = 7, dim_head = 32,
        mbconv_expansion_rate = 4, mbconv_shrinkage_rate = 0.25, dropout = 0.1,):
        super().__init__()
        w = window_size
        self.layers = nn.ModuleList([])
        for stage_ind in range(layer_depth):
            block = nn.Sequential(
                # parallel local-global attention mechanism
                MyLayer(layer_dim, dim_head=dim_head, mbconv_shrinkage_rate=mbconv_shrinkage_rate, mbconv_expansion_rate=mbconv_expansion_rate, dropout=dropout),

                # EA Attention
                Rearrange('b d w h -> b (w h) d'),
                PreNormResidual(layer_dim, External_Attention(dim=layer_dim)),
                Rearrange('b (w h) d -> b d w h', w=image_size),
            )
            self.layers.append(block)

    def forward(self, x):
        for stage in self.layers:
            x = stage(x)

        return x

class Refine_Cross_Attention(nn.Module):
    def __init__(self, attention_prosition, img_size, in_dim, dim, mbconv_expansion_rate = 4, mbconv_shrinkage_rate = 0.25, patch_size=3):
        super(Refine_Cross_Attention, self).__init__()
        # in_dim indicates all dims, and dim indicates the dim of the target
        self.attention_prosition = attention_prosition
        self.conv1 = MBConv(dim, dim, downsample=False, expansion_rate=mbconv_expansion_rate, shrinkage_rate=mbconv_shrinkage_rate)
        self.conv2 = MBConv(dim, dim, downsample=False, expansion_rate=mbconv_expansion_rate,
                            shrinkage_rate=mbconv_shrinkage_rate)
        self.conv3 = MBConv(dim, dim, downsample=False, expansion_rate=mbconv_expansion_rate,
                            shrinkage_rate=mbconv_shrinkage_rate)
        self.conv4 = MBConv(dim, dim, downsample=False, expansion_rate=mbconv_expansion_rate,
                            shrinkage_rate=mbconv_shrinkage_rate)

        if attention_prosition == 1:
            # From the bottom up to the first attention block, the input from top to bottom is twice downsampled, unchanged, and twice upsampled, respectively
            self.embedding1 = PatchMerging(img_size=img_size, patch_size=patch_size, stride=2, padding=1, in_ch=in_dim[0], dim=in_dim[1])
            self.embedding2 = PatchExpand(input_resolution=(img_size//16, img_size//16), dim=in_dim[2], dim_scale=2)
            self.linear_2 = nn.Linear(256, 320)

            self.channel_attn = ChannelAttention(3*in_dim[1])
            self.norm = nn.LayerNorm(3*in_dim[1])
            self.linear_downsample = nn.Linear(3*in_dim[1], in_dim[1], bias=False)
        if attention_prosition == 2:
            # For the second attention block from the top up, the input from the top down is unchanged, 2x up sampling, and 4x up sampling
            self.embedding1 = PatchExpand(input_resolution=(img_size//8, img_size//8), dim=in_dim[1], dim_scale=2)
            self.embedding2 = FinalPatchExpand_X4(input_resolution=(img_size//16, img_size//16), dim=in_dim[2], dim_scale=4)
            self.linear_2 = nn.Linear(160, 128)
            self.linear_3 = nn.Linear(512, 128)

            self.channel_attn = ChannelAttention(3*in_dim[0])
            self.norm = nn.LayerNorm(3*in_dim[0])
            self.linear_downsample = nn.Linear(3*in_dim[0], in_dim[0], bias=False)

    def forward(self, x):
        if self.attention_prosition == 1:
            embed1 = x[0]
            embed1, H, W = self.embedding1(embed1)
            embed1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(embed1)

            embed2 = x[1]

            embed3 = x[2]
            embed3 = self.embedding2(embed3)
            embed3 = Rearrange("b d h w -> b (h w) d", h=H, w=W)(embed3)
            embed3 = self.linear_2(embed3)
            embed3 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(embed3)

        if self.attention_prosition == 2:
            embed1 = x[0]
            B, H, W = embed1.shape[0], embed1.shape[2], embed1.shape[3]

            embed2 = x[1]
            embed2 = self.embedding1(embed2)
            embed2 = Rearrange("b d h w -> b (h w) d", h=H, w=W)(embed2)
            embed2 = self.linear_2(embed2)
            embed2 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(embed2)

            embed3 = x[2]
            embed3 = self.embedding2(embed3)
            embed3 = Rearrange("b d h w -> b (h w) d", h=H, w=W)(embed3)
            embed3 = self.linear_3(embed3)
            embed3 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(embed3)

        # MBConv is used as the relative position encoding
        embed1 = self.conv1(embed1)
        embed2 = self.conv2(embed2)
        embed3 = self.conv3(embed3)

        embed = torch.cat([embed1, embed2, embed3], dim=1)
        embed = Rearrange("b d h w -> b (h w) d", h=H, w=W)(embed)
        chan_embed = self.norm(embed)
        chan_embed = self.channel_attn(chan_embed)
        embed = embed + chan_embed

        embed = self.linear_downsample(embed)
        return embed


# Encoder, We used the MAXVit implementation framework, thanks to lucidrains/vit-pytorch.
# However, the building block, MAXTransformerBlock, is what we proposed and implemented.
class MaxViT(nn.Module):
    def __init__(self, image_size, depth, in_dim,
                 dim_head = 32,
                 window_size = 7,
                 mbconv_expansion_rate = 4,
                 mbconv_shrinkage_rate = 0.25,
                 dropout = 0.1):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        image_sizes = [image_size // 4, image_size // 8, image_size // 16, image_size // 32]

        self.patch_embed1 = PatchMerging(
            image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0]
        )
        self.patch_embed2 = PatchMerging(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], in_dim[0], in_dim[1]
        )
        self.patch_embed3 = PatchMerging(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2]
        )

        w = window_size
        # transformer encoder
        self.block1 = MAXTransformerBlock(layer_depth=depth[0],
                                           layer_dim=in_dim[0], image_size= image_sizes[0],
                                           window_size=w, dim_head=dim_head,
                                           mbconv_expansion_rate=mbconv_expansion_rate,
                                           mbconv_shrinkage_rate=mbconv_shrinkage_rate, dropout=dropout)

        self.block2 = MAXTransformerBlock(layer_depth=depth[1],
                                           layer_dim=in_dim[1], image_size= image_sizes[1],
                                           window_size=w, dim_head=dim_head,
                                           mbconv_expansion_rate=mbconv_expansion_rate,
                                           mbconv_shrinkage_rate=mbconv_shrinkage_rate, dropout=dropout)

        self.block3 = MAXTransformerBlock(layer_depth=depth[2],
                                           layer_dim=in_dim[2], image_size= image_sizes[2],
                                           window_size=w, dim_head=dim_head,
                                           mbconv_expansion_rate=mbconv_expansion_rate,
                                           mbconv_shrinkage_rate=mbconv_shrinkage_rate, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        # stage 1
        x, H, W = self.patch_embed1(x)
        x = Rearrange("b (h w) d -> b d h w", h=H, w=W)(x)
        x = self.block1(x)
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        x = Rearrange("b (h w) d -> b d h w", h=H, w=W)(x)
        x = self.block2(x)
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        x = Rearrange("b (h w) d -> b d h w", h=H, w=W)(x)
        x = self.block3(x)
        outs.append(x)

        return outs


# Decoder
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        if len(x.shape) == 4:
            B, C, _, _ = x.shape
            x = x.permute(0, 2, 3, 1).view(B, -1, C)
        x = self.expand(x)

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        B, L, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H * 2, W * 2)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        if len(x.shape) == 4:
            B, C, _, _ = x.shape
            x = x.permute(0, 2, 3, 1).view(B, -1, C)
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale**2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        B, L, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H * 4, W * 4)
        return x


class MyDecoderLayer(nn.Module):
    def __init__(
        self, input_size, in_out_chan, head_count, token_mlp_mode, image_size, layer_depth=2, n_class=9, norm_layer=nn.LayerNorm, is_last=False, is_first=False,
            dim_head = 32, window_size=7, mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, dropout=0.1
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        if not is_last:
            if not is_first:
                self.x1_linear = nn.Linear(x1_dim, out_dim)
                self.cross_attn = CrossAttentionBlock(
                    dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
                )
                self.concat_linear = nn.Linear(2 * dims, out_dim)
                # transformer decoder
                self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
                self.last_layer = None
            else:
                self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2,
                                            norm_layer=norm_layer)
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        if not is_first:
            w = window_size
            self.layer_former = MAXTransformerBlock(layer_depth=layer_depth,
                                               layer_dim=out_dim, image_size=image_size,
                                               window_size=w, dim_head=dim_head,
                                               mbconv_expansion_rate=mbconv_expansion_rate,
                                               mbconv_shrinkage_rate=mbconv_shrinkage_rate, dropout=dropout)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            b, c, h, w = x1.shape
            if len(x1.shape) == 4:
                x1 = Rearrange("b d h w -> b (h w) d", h=h, w=w)(x1)
            x1_expand = self.x1_linear(x1)
            cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2))
            cat_linear_x = Rearrange("b (h w) d -> b d h w", h=h, w=w)(cat_linear_x)
            tran_layer = self.layer_former(cat_linear_x)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer))
            else:
                out = self.layer_up(tran_layer)
        else:
            out = self.layer_up(x1)
        return out


# our network
class MAXFormer(nn.Module):
    def __init__(self, num_classes=9, image_size=224, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        dims, key_dim, value_dim, layers = [[128, 320, 512, 1024], [128, 320, 512], [128, 320, 512], [2, 2, 4, 1]]
        image_sizes = [image_size // 4, image_size // 8, image_size // 16, image_size // 32]
        self.backbone = MaxViT(
            image_size=224,
            in_dim=dims,
            depth=layers
        )
        self.refine_attn_2 = Refine_Cross_Attention(attention_prosition=2, img_size=224, in_dim=dims, dim=128)
        self.refine_attn_1 = Refine_Cross_Attention(attention_prosition=1, img_size=224, in_dim=dims, dim=320)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [64, 128, 128, 128, 160],
            [320, 320, 320, 320, 256],
            [512, 512, 512, 512, 512],
            [1024, 1024, 1024, 1024, 1024],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        self.decoder_2 = MyDecoderLayer(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            layer_depth=layers[2],
            image_size=image_sizes[2],
            n_class=num_classes,
            is_first=True
        )
        self.decoder_1 = MyDecoderLayer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            layer_depth=layers[1],
            image_size=image_sizes[1],
            n_class=num_classes,
        )
        self.decoder_0 = MyDecoderLayer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            layer_depth=layers[0],
            image_size=image_sizes[0],
            n_class=num_classes,
            is_last=True,
        )

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc = self.backbone(x)
        refine_attn_2 = self.refine_attn_2(output_enc)
        refine_attn_1 = self.refine_attn_1(output_enc)

        b, c, _, _ = output_enc[2].shape
        # ---------------Decoder-------------------------
        tmp_2 = self.decoder_2(output_enc[2].permute(0, 2, 3, 1).view(b, -1, c))
        tmp_1 = self.decoder_1(tmp_2, refine_attn_1)
        tmp_0 = self.decoder_0(tmp_1, refine_attn_2)

        return tmp_0
