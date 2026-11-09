import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q = qkv[:, 0].transpose(-1, -2)
        k = qkv[:, 1].transpose(-1, -2)
        v = qkv[:, 2].transpose(-1, -2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 8)
    ):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        c = [base_channels * m for m in channel_mult]

        # Down
        self.down0 = ResnetBlock(c[0], c[0], time_dim)
        self.downsample0 = nn.Conv2d(c[0], c[0], 3, stride=2, padding=1)

        self.down1 = ResnetBlock(c[0], c[1], time_dim)
        self.downsample1 = nn.Conv2d(c[1], c[1], 3, stride=2, padding=1)

        self.down2 = ResnetBlock(c[1], c[2], time_dim)
        self.attn2 = Attention(c[2])
        self.downsample2 = nn.Conv2d(c[2], c[2], 3, stride=2, padding=1)

        self.down3 = ResnetBlock(c[2], c[3], time_dim)

        # Mid
        self.mid = ResnetBlock(c[3], c[3], time_dim)

        # Up
        self.up3 = ResnetBlock(c[3] + c[3], c[2], time_dim)
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c[2], c[2], 3, padding=1)
        )

        self.attn_up2 = Attention(c[2])
        self.up2 = ResnetBlock(c[2] + c[2], c[1], time_dim)
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c[1], c[1], 3, padding=1)
        )

        self.up1 = ResnetBlock(c[1] + c[1], c[0], time_dim)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c[0], c[0], 3, padding=1)
        )

        self.up0 = ResnetBlock(c[0] + c[0], c[0], time_dim)

        self.norm_out = nn.GroupNorm(8, c[0])
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(c[0], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = self.conv_in(x)

        h0 = self.down0(x, t_emb)
        d0 = self.downsample0(h0)

        h1 = self.down1(d0, t_emb)
        d1 = self.downsample1(h1)

        h2 = self.attn2(self.down2(d1, t_emb))
        d2 = self.downsample2(h2)

        h3 = self.down3(d2, t_emb)

        m = self.mid(h3, t_emb)

        u3 = self.up3(torch.cat([m, h3], dim=1), t_emb)
        u3 = self.upsample3(u3)

        u2 = self.attn_up2(u3)
        u2 = self.up2(torch.cat([u2, h2], dim=1), t_emb)
        u2 = self.upsample2(u2)

        u1 = self.up1(torch.cat([u2, h1], dim=1), t_emb)
        u1 = self.upsample1(u1)

        u0 = self.up0(torch.cat([u1, h0], dim=1), t_emb)

        return self.conv_out(self.act_out(self.norm_out(u0)))


if __name__ == "__main__":
    model = UNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.2f}M ({total_params:,})")

    assert 10_000_000 <= total_params <= 18_000_000

    batch_size = 4
    x_test = torch.randn(batch_size, 3, 64, 64)
    t_test = torch.randint(0, 1000, (batch_size,))

    out = model(x_test, t_test)
    print(f"Input: {x_test.shape}, Output: {out.shape}")

    assert out.shape == (batch_size, 3, 64, 64)
    print("U-Net validada exitosamente.")
