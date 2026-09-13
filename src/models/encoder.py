import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Small CNN encoder: 84x84x3 → z ∈ R^256
    GroupNorm + SiLU throughout (no BatchNorm — incompatible with EMA targets
    and small eval batch sizes).
    No global average pooling — spatial layout encodes lane geometry.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim

        self.conv = nn.Sequential(
            # 84x84 → 42x42
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),

            # 42x42 → 42x42
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),

            # 42x42 → 21x21
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),

            # 21x21 → 21x21
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),

            # 21x21 → 11x11
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(),

            # 11x11 → 6x6
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(),
        )

        # 128 * 6 * 6 = 4608
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4608, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 84, 84) float, values in [0, 255] or [0, 1]
        returns: (B, 256)
        """
        return self.proj(self.conv(x))


def build_encoder(out_dim: int = 256) -> CNNEncoder:
    return CNNEncoder(out_dim=out_dim)


if __name__ == "__main__":
    enc = CNNEncoder()
    x   = torch.randint(0, 255, (4, 3, 84, 84)).float()
    z   = enc(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(z.shape)}")
    assert z.shape == (4, 256), f"wrong shape: {z.shape}"
    n = sum(p.numel() for p in enc.parameters())
    print(f"params: {n:,}")
    print("CNN encoder OK")