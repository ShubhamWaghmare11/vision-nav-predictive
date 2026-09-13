import torch
import torch.nn as nn
import copy


# ── EMA target encoder ────────────────────────────────────────────────────────

class EMAEncoder(nn.Module):
    """
    Exponential moving average copy of the online encoder.
    Parameters updated as: ξ ← τ·ξ + (1-τ)·θ
    Never receives gradients.
    """

    def __init__(self, online_encoder: nn.Module, tau: float = 0.99):
        super().__init__()
        self.tau = tau
        self.encoder = copy.deepcopy(online_encoder)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: nn.Module):
        for ema_p, online_p in zip(
            self.encoder.parameters(), online_encoder.parameters()
        ):
            ema_p.data.mul_(self.tau).add_(online_p.data, alpha=1 - self.tau)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 3, 84, 84) → (B, T, 256)"""
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W).float() / 255.0
        z = self.encoder(flat)
        return z.view(B, T, -1)


# ── shared projector (online + EMA) ──────────────────────────────────────────

def make_projector(in_dim: int = 256, out_dim: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.LayerNorm(512),
        nn.SiLU(),
        nn.Linear(512, out_dim),
    )


# ── A1: action-conditioned future latent prediction ───────────────────────────

class A1Head(nn.Module):
    """
    Predicts future latents z_{t+k} for k in {2, 4, 8}
    conditioned on h_t and the expert action chunk.

    Loss: negative cosine similarity against EMA target latents (BYOL-style).
    The EMA encoder + stop-gradient on targets prevent collapse.
    """

    HORIZONS = [2, 4, 8]

    def __init__(self, d_model: int = 256, action_dim: int = 16, proj_dim: int = 128):
        super().__init__()
        # action chunk embedder: (H*2,) → action_dim
        self.action_emb = nn.Linear(8 * 2, action_dim)  # 8-step chunk × 2 dims

        # predictor: [h_t || action_emb] → K_pred × proj_dim
        self.predictor = nn.Sequential(
            nn.Linear(d_model + action_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, len(self.HORIZONS) * proj_dim),
        )

        self.online_proj  = make_projector(d_model, proj_dim)
        self.proj_dim = proj_dim

    def forward(
        self,
        h_t:          torch.Tensor,   # (B, d_model)  trunk readout
        action_chunk: torch.Tensor,   # (B, 8, 2)     expert actions
    ) -> torch.Tensor:
        """Returns predictions: (B, n_horizons, proj_dim)"""
        a = self.action_emb(action_chunk.flatten(1))        # (B, action_dim)
        inp = torch.cat([h_t, a], dim=-1)                   # (B, d+action_dim)
        preds = self.predictor(inp)                         # (B, n_horizons*proj_dim)
        return preds.reshape(-1, len(self.HORIZONS), self.proj_dim)

    def loss(
        self,
        h_t:           torch.Tensor,   # (B, d_model)
        action_chunk:  torch.Tensor,   # (B, 8, 2)
        future_frames: torch.Tensor,   # (B, max_k, 3, 84, 84)
        ema_encoder:   EMAEncoder,
        online_proj:   nn.Module = None,
        ema_proj:      nn.Module = None,
    ) -> torch.Tensor:
        """
        Computes mean negative-cosine loss over prediction horizons.
        Targets are stop-gradient EMA projections of future frames.
        """
        preds = self.forward(h_t, action_chunk)   # (B, 3, proj_dim)

        total = 0.0
        for i, k in enumerate(self.HORIZONS):
            # future frame at horizon k (0-indexed: k=2 → index 1)
            fut = future_frames[:, k - 1:k, :, :, :]    # (B,1,3,84,84)
            with torch.no_grad():
                z_target = ema_encoder(fut).squeeze(1)   # (B, 256)
                if ema_proj is not None:
                    y = ema_proj(z_target)               # (B, proj_dim)
                else:
                    y = z_target[:, :self.proj_dim]

            p = preds[:, i, :]                           # (B, proj_dim)
            # negative cosine similarity
            loss_k = 1.0 - torch.nn.functional.cosine_similarity(p, y, dim=-1)
            total = total + loss_k.mean()

        return total / len(self.HORIZONS)


# ── A2: same-timestep cross-view invariance (non-predictive control) ──────────

class A2Head(nn.Module):
    """
    Predicts the representation of the current frame under a different
    augmentation. No temporal or action information.
    Matched capacity to A1: same projector + predictor architecture,
    constant zero vector in place of the action embedding.
    """

    def __init__(self, d_model: int = 256, action_dim: int = 16, proj_dim: int = 128):
        super().__init__()
        # constant input replacing action (matched capacity)
        self.const_input = nn.Parameter(torch.zeros(1, action_dim))

        self.predictor = nn.Sequential(
            nn.Linear(d_model + action_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 3 * proj_dim),   # same output size as A1
        )

        self.online_proj = make_projector(d_model, proj_dim)
        self.proj_dim = proj_dim

    def loss(
        self,
        h_t:          torch.Tensor,    # (B, d_model)  from the FIRST augmented view
        z_t_2nd_view: torch.Tensor,    # (B, 256)  EMA encoder on 2nd augmented view
        ema_proj:     nn.Module = None,
    ) -> torch.Tensor:
        """
        Loss: predict the EMA projection of the second view
        from the trunk output of the first view.
        """
        B = h_t.size(0)
        const = self.const_input.expand(B, -1)
        inp   = torch.cat([h_t, const], dim=-1)
        preds = self.predictor(inp).reshape(B, 3, self.proj_dim)

        with torch.no_grad():
            if ema_proj is not None:
                y = ema_proj(z_t_2nd_view)
            else:
                y = z_t_2nd_view[:, :self.proj_dim]

        # use mean over the 3 output slots (matched capacity)
        loss = 0.0
        for i in range(3):
            p = preds[:, i, :]
            loss = loss + (1.0 - torch.nn.functional.cosine_similarity(p, y, dim=-1)).mean()
        return loss / 3.0


# ── A3: action-unconditioned future latent prediction ─────────────────────────

class A3Head(nn.Module):
    """
    Identical to A1 but the action chunk is replaced by a zero tensor.
    Isolates whether action-conditioning specifically matters (A1 vs A3).
    """

    def __init__(self, d_model: int = 256, action_dim: int = 16, proj_dim: int = 128):
        super().__init__()
        self.action_emb = nn.Linear(8 * 2, action_dim)   # same arch as A1

        self.predictor = nn.Sequential(
            nn.Linear(d_model + action_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 3 * proj_dim),
        )
        self.proj_dim = proj_dim
        self.HORIZONS = [2, 4, 8]

    def loss(
        self,
        h_t:           torch.Tensor,
        future_frames: torch.Tensor,
        ema_encoder:   EMAEncoder,
        ema_proj:      nn.Module = None,
    ) -> torch.Tensor:
        B = h_t.size(0)
        # zero action — same embedder, zero input
        zero_action = torch.zeros(B, 8 * 2, device=h_t.device)
        a   = self.action_emb(zero_action)
        inp = torch.cat([h_t, a], dim=-1)
        preds = self.predictor(inp).reshape(B, 3, self.proj_dim)

        total = 0.0
        for i, k in enumerate(self.HORIZONS):
            fut = future_frames[:, k - 1:k, :, :, :]
            with torch.no_grad():
                z_target = ema_encoder(fut).squeeze(1)
                if ema_proj is not None:
                    y = ema_proj(z_target)
                else:
                    y = z_target[:, :self.proj_dim]
            p = preds[:, i, :]
            total = total + (1.0 - torch.nn.functional.cosine_similarity(p, y, dim=-1)).mean()
        return total / 3.0


# ── shape test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.models.encoder import CNNEncoder
    B = 4
    enc    = CNNEncoder()
    ema    = EMAEncoder(enc)
    a1     = A1Head()
    a2     = A2Head()
    a3     = A3Head()

    h_t          = torch.randn(B, 256)
    action_chunk = torch.randn(B, 8, 2)
    future_frames = torch.randint(0, 255, (B, 8, 3, 84, 84), dtype=torch.uint8)
    z_2nd        = torch.randn(B, 256)

    proj = make_projector()

    l_a1 = a1.loss(h_t, action_chunk, future_frames, ema, proj, proj)
    l_a2 = a2.loss(h_t, z_2nd, proj)
    l_a3 = a3.loss(h_t, future_frames, ema, proj)

    print(f"A1 loss: {l_a1.item():.4f}")
    print(f"A2 loss: {l_a2.item():.4f}")
    print(f"A3 loss: {l_a3.item():.4f}")

    # verify EMA update
    ema.update(enc)
    print("EMA update OK")

    n_a1 = sum(p.numel() for p in a1.parameters())
    n_a2 = sum(p.numel() for p in a2.parameters())
    n_a3 = sum(p.numel() for p in a3.parameters())
    print(f"A1 params: {n_a1:,}  A2 params: {n_a2:,}  A3 params: {n_a3:,}")
    print("All auxiliary heads OK")


