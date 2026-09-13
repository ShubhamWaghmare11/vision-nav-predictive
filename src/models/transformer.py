import torch
import torch.nn as nn
import math


class TransformerTrunk(nn.Module):
    """
    Causal-free encoder-only transformer over a token sequence:
      [READ] + [CMD] + [PROP] + z_{t-K+1} ... z_t
    Total sequence length = 1 + 1 + 1 + K = K + 3

    Output: the READ token's representation h_t ∈ R^d_model
    """

    def __init__(
        self,
        d_model:   int = 256,
        n_layers:  int = 4,
        n_heads:   int = 4,
        mlp_ratio: int = 4,
        dropout:   float = 0.1,
        K:         int = 8,       # history length (number of past latents)
        n_commands:int = 3,       # STRAIGHT / LEFT / RIGHT
        proprio_dim: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.seq_len = K + 3     # READ + CMD + PROP + K latents

        # input projections
        self.cmd_emb   = nn.Linear(n_commands, d_model)
        self.prop_emb  = nn.Linear(proprio_dim, d_model)
        self.lat_emb   = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # learned positional embeddings — one per slot
        self.pos_emb = nn.Embedding(self.seq_len, d_model)

        # learned READ token
        self.read_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.read_token, std=0.02)

        # transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        latents:  torch.Tensor,   # (B, K, 256)  — history of visual latents
        proprio:  torch.Tensor,   # (B, 5)
        command:  torch.Tensor,   # (B,) long  — class index
    ) -> torch.Tensor:
        """Returns h_t = READ token output, shape (B, d_model)"""
        B = latents.size(0)

        # one-hot command
        cmd_oh = torch.zeros(B, 3, device=latents.device)
        cmd_oh.scatter_(1, command.unsqueeze(1), 1.0)

        # project each input to d_model
        cmd_tok  = self.cmd_emb(cmd_oh).unsqueeze(1)           # (B,1,d)
        prop_tok = self.prop_emb(proprio).unsqueeze(1)         # (B,1,d)
        lat_toks = self.lat_emb(latents)                       # (B,K,d)
        read_tok = self.read_token.expand(B, -1, -1)           # (B,1,d)

        # assemble: [READ, CMD, PROP, z_{t-K+1}, ..., z_t]
        tokens = torch.cat([read_tok, cmd_tok, prop_tok, lat_toks], dim=1)  # (B, K+3, d)

        # add positional embeddings
        pos = torch.arange(self.seq_len, device=latents.device)
        tokens = tokens + self.pos_emb(pos).unsqueeze(0)

        # transformer
        out = self.transformer(tokens)   # (B, K+3, d)
        out = self.norm(out)

        return out[:, 0, :]   # READ token → h_t, shape (B, d)


if __name__ == "__main__":
    trunk = TransformerTrunk()
    B = 4
    latents = torch.randn(B, 8, 256)
    proprio = torch.randn(B, 5)
    command = torch.randint(0, 3, (B,))

    h = trunk(latents, proprio, command)
    print(f"latents: {tuple(latents.shape)}")
    print(f"proprio: {tuple(proprio.shape)}")
    print(f"command: {tuple(command.shape)}")
    print(f"output h_t: {tuple(h.shape)}")
    assert h.shape == (B, 256), f"wrong shape: {h.shape}"
    n = sum(p.numel() for p in trunk.parameters())
    print(f"params: {n:,}")
    print("Transformer trunk OK")