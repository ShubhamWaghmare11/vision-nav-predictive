import torch
import torch.nn as nn
from encoder import CNNEncoder
from transformer import TransformerTrunk


class ChunkHead(nn.Module):
    """
    Maps h_t → action chunk of shape (H, 2)
    H = chunk length, 2 = [steer, accel]
    """
    def __init__(self, d_model: int = 256, chunk_len: int = 8):
        super().__init__()
        self.chunk_len = chunk_len
        self.net = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, chunk_len * 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) → (B, H, 2)"""
        return torch.tanh(self.net(h)).reshape(-1, self.chunk_len, 2)


class VisualPolicy(nn.Module):
    """
    Full policy: image + history + proprio + command → action chunk

    Also exposes encode() for use by auxiliary heads and analysis.
    """

    def __init__(
        self,
        chunk_len:   int = 8,
        K:           int = 8,
        d_model:     int = 256,
        n_layers:    int = 4,
        n_heads:     int = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.chunk_len = chunk_len

        self.encoder = CNNEncoder(out_dim=d_model)
        self.trunk    = TransformerTrunk(
            d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, dropout=dropout, K=K,
        )
        self.chunk_head = ChunkHead(d_model=d_model, chunk_len=chunk_len)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, 3, 84, 84)  — arbitrary T frames
        returns: (B, T, d_model)   — per-frame latents
        """
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W).float() / 255.0
        z = self.encoder(flat)
        return z.view(B, T, -1)

    def forward(
        self,
        frames:  torch.Tensor,   # (B, K, 3, 84, 84)  — history of K frames
        proprio: torch.Tensor,   # (B, 5)
        command: torch.Tensor,   # (B,) long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            actions: (B, H, 2)  — predicted action chunk
            h_t:     (B, d)     — trunk readout (for auxiliary heads)
        """
        z = self.encode(frames)              # (B, K, 256)
        h = self.trunk(z, proprio, command)  # (B, 256)
        actions = self.chunk_head(h)         # (B, H, 2)
        return actions, h


if __name__ == "__main__":
    policy = VisualPolicy()

    B, K, H = 4, 8, 8
    frames  = torch.randint(0, 255, (B, K, 3, 84, 84), dtype=torch.uint8)
    proprio = torch.randn(B, 5)
    command = torch.randint(0, 3, (B,))

    actions, h = policy(frames, proprio, command)

    print(f"frames:  {tuple(frames.shape)}")
    print(f"actions: {tuple(actions.shape)}")
    print(f"h_t:     {tuple(h.shape)}")
    assert actions.shape == (B, H, 2), f"wrong actions shape: {actions.shape}"
    assert h.shape == (B, 256),        f"wrong h shape: {h.shape}"

    n = sum(p.numel() for p in policy.parameters())
    print(f"total params: {n:,}")
    print("VisualPolicy OK")