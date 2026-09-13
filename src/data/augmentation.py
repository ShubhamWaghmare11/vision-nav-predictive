import torch
import torch.nn as nn
import torch.nn.functional as F


class GPUAugment(nn.Module):
    """
    Batched, GPU-side replacement for the old per-sample CPU augmentation.

    Applies (per-sample, shared across all frames within a sample so a
    history/future stack stays visually consistent):
      - random crop (pad by `pad`, crop back to `crop_size`)
      - brightness/contrast jitter
      - random grayscale

    Hue jitter is dropped entirely — it was the dominant cost of the old
    CPU path and contributes little signal for this task.
    """

    def __init__(
        self,
        pad:        int   = 4,
        crop_size:  int   = 84,
        brightness: float = 0.2,
        contrast:   float = 0.2,
        color_p:    float = 0.8,
        gray_p:     float = 0.1,
    ):
        super().__init__()
        self.pad        = pad
        self.crop_size  = crop_size
        self.brightness = brightness
        self.contrast   = contrast
        self.color_p    = color_p
        self.gray_p     = gray_p

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W) uint8, already on GPU.
        returns: (B, T, C, crop_size, crop_size) uint8, same device.
        """
        B, T, C, H, W = x.shape
        device = x.device
        pad = self.pad

        x = x.reshape(B * T, C, H, W).float()
        x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        Wp = W + 2 * pad

        # ── random crop: one offset per sample, shared across its T frames ──
        offset_i = torch.randint(0, 2 * pad + 1, (B,), device=device).repeat_interleave(T)
        offset_j = torch.randint(0, 2 * pad + 1, (B,), device=device).repeat_interleave(T)

        arange_c = torch.arange(self.crop_size, device=device)
        rows = offset_i.view(-1, 1) + arange_c.view(1, -1)   # (B*T, crop)
        cols = offset_j.view(-1, 1) + arange_c.view(1, -1)   # (B*T, crop)

        idx_rows = rows.view(B * T, 1, self.crop_size, 1).expand(B * T, C, self.crop_size, Wp)
        x = torch.gather(x, 2, idx_rows)
        idx_cols = cols.view(B * T, 1, 1, self.crop_size).expand(B * T, C, self.crop_size, self.crop_size)
        x = torch.gather(x, 3, idx_cols)

        # ── brightness / contrast jitter, one draw per sample ──
        apply_color = (torch.rand(B, device=device) < self.color_p).repeat_interleave(T).view(-1, 1, 1, 1)
        b_factor = (1.0 + (torch.rand(B, device=device) * 2 - 1) * self.brightness).repeat_interleave(T).view(-1, 1, 1, 1)
        c_factor = (1.0 + (torch.rand(B, device=device) * 2 - 1) * self.contrast).repeat_interleave(T).view(-1, 1, 1, 1)

        x_jit  = x * b_factor
        mean   = x_jit.mean(dim=(1, 2, 3), keepdim=True)
        x_jit  = (x_jit - mean) * c_factor + mean
        x = torch.where(apply_color, x_jit, x)

        # ── random grayscale, one draw per sample ──
        apply_gray = (torch.rand(B, device=device) < self.gray_p).repeat_interleave(T).view(-1, 1, 1, 1)
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand(-1, C, -1, -1)
        x = torch.where(apply_gray, gray, x)

        x = x.clamp(0, 255).byte()
        return x.view(B, T, C, self.crop_size, self.crop_size)
