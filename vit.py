# Inspired by:
# - https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
# - https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# - https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_flash_attn_vit.py
# - https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
# - lightly/models/utils.py


import math
from collections import namedtuple

import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    from einops._torch_specific import allow_ops_in_compiled_graph

    allow_ops_in_compiled_graph()

import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

Config = namedtuple(
    "FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Attend(nn.Module):
    def __init__(self, use_flash=False):
        super().__init__()
        self.use_flash = use_flash
        assert not (
            use_flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = Config(True, False, False)
        else:
            self.cuda_config = Config(False, True, True)

    def flash_attn(self, q, k, v):
        config = self.cuda_config if q.is_cuda else self.cpu_config

        # flash attention - https://arxiv.org/abs/2205.14135

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v)

        return out

    def forward(self, q, k, v):
        n, device, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, use_flash=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = Attend(use_flash=use_flash)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype

        if drop_prob <= 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (batch, *((1,) * (x.ndim - 1)))

        keep_mask = torch.zeros(shape, device=device).float().uniform_(0, 1) < keep_prob
        output = x.div(keep_prob) * keep_mask.float()
        return output

    def __repr__(self):
        return f"DropPath(p={self.drop_prob})"


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, dim, image_size, patch_size, channels, temperature=10000):
        super().__init__()
        self.T = temperature
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, dim))

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b h w (p1 p2 c)", p1=patch_height, p2=patch_width
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def posemb_sincos_2d(self, patches, temperature=10000):
        _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
        omega = 1.0 / (temperature**omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe.type(dtype)

    def forward(self, img, cls_token):
        b, c, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        x = rearrange(x, "b ... d -> b (...) d") + self.posemb_sincos_2d(x)
        cls_tokens = repeat(cls_token + self.cls_pos_embedding, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        return x


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, dim, image_size, patch_size, channels):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embedding.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embedding
        class_pos_embed = self.pos_embedding[:, 0]
        patch_pos_embed = self.pos_embedding[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, img, cls_token):
        b, nc, w, h = img.shape
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, drop_path_rate, use_flash):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, use_flash=use_flash)
        self.ff = FeedForward(dim, mlp_dim)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        x = self.drop_path(self.attn(x)) + x
        x = self.drop_path(self.ff(x)) + x
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, drop_path_rate, use_flash):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    drop_path_rate=layer_dpr,
                    use_flash=use_flash,
                )
                for layer_dpr in dpr
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        num_classes=0,
        mlp_ratio=4,
        channels=3,
        drop_path_rate=0.0,
        use_flash=True,
        pos_emb="learned",
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        dim_head = dim // heads
        mlp_dim = int(dim * mlp_ratio)

        self.num_features = dim

        assert pos_emb in (
            "cosine",
            "learned",
        ), "Choose pos_emb as 'cosine' or 'learned'."
        assert dim % heads == 0, f"{dim=} not divisible by {heads=}."
        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        if pos_emb == "cosine":
            self.prepare_tokens = CosinePositionalEmbedding(
                dim, image_size, patch_size, channels
            )
        else:
            self.prepare_tokens = LearnedPositionalEmbedding(
                dim, image_size, patch_size, channels
            )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, drop_path_rate, use_flash
        )

        self.to_latent = nn.Identity()
        self.linear_head = nn.Identity()

        if num_classes > 0:
            self.linear_head = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, num_classes)
            )

    @staticmethod
    def get_at_index(tokens, index):
        dim = tokens.shape[-1]
        index = index.unsqueeze(-1).expand(-1, -1, dim)
        return torch.gather(tokens, 1, index)

    def forward(self, img, idx_keep=None):
        x = self.prepare_tokens(img, self.cls_token)
        if idx_keep is not None:
            x = self.get_at_index(x, idx_keep)
        x = self.transformer(x)
        x = x[:, 0]

        x = self.to_latent(x)
        return self.linear_head(x)


def get_vit(
    arch: str,
    image_size: int = 224,
    patch_size: int = 16,
    drop_path_rate: float = 0.0,
    pos_emb: str = "learned",
):
    if arch == "tiny":
        dim = 192
        heads = 3
        depth = 12

    elif arch == "small":
        dim = 384
        heads = 6
        depth = 12

    elif arch == "base":
        dim = 768
        heads = 12
        depth = 12

    elif arch == "large":
        dim = 1024
        heads = 16
        depth = 24

    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        drop_path_rate=drop_path_rate,
        pos_emb=pos_emb,
    )
