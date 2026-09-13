import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from pathlib import Path

from src.models.policy import VisualPolicy
from src.models.auxiliary import EMAEncoder, A1Head, A2Head, A3Head, make_projector
from src.data.dataset import DrivingDataset


class Trainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Device: {self.device}")

        # ── model ──────────────────────────────────────────────────────────
        self.policy = VisualPolicy(
            chunk_len=config.get("chunk_len", 8),
            K=config.get("K", 8),
            d_model=config.get("d_model", 256),
            n_layers=config.get("n_layers", 4),
            n_heads=config.get("n_heads", 4),
            dropout=config.get("dropout", 0.1),
        ).to(self.device)

        # ── auxiliary components ────────────────────────────────────────────
        self.arm = config.get("arm", "a0")
        self.lam = config.get("lambda_aux", 0.1)

        self.ema_encoder = None
        self.aux_head    = None
        self.online_proj = None
        self.ema_proj    = None

        if self.arm != "a0":
            self.ema_encoder = EMAEncoder(
                self.policy.encoder, tau=config.get("ema_tau", 0.99)
            ).to(self.device)
            self.online_proj = make_projector().to(self.device)
            self.ema_proj    = make_projector().to(self.device)
            self.ema_proj.load_state_dict(self.online_proj.state_dict())
            for p in self.ema_proj.parameters():
                p.requires_grad_(False)

            if self.arm == "a1":
                self.aux_head = A1Head().to(self.device)
            elif self.arm == "a2":
                self.aux_head = A2Head().to(self.device)
            elif self.arm == "a3":
                self.aux_head = A3Head().to(self.device)

        # ── optimizer ──────────────────────────────────────────────────────
        params = list(self.policy.parameters())
        if self.aux_head    is not None: params += list(self.aux_head.parameters())
        if self.online_proj is not None: params += list(self.online_proj.parameters())

        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.get("lr", 3e-4),
            betas=(0.9, 0.95),
            weight_decay=config.get("weight_decay", 0.05),
        )

        # ── scheduler ──────────────────────────────────────────────────────
        total_steps  = config.get("total_steps", 60_000)
        warmup_steps = config.get("warmup_steps", 2_000)
        self.total_steps = total_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        # ── data ───────────────────────────────────────────────────────────
        print("Loading training dataset...")
        self.train_ds = DrivingDataset(
            config["train_index"],
            K=config.get("K", 8),
            H=config.get("chunk_len", 8),
            augment=True,
        )
        print("Loading validation dataset...")
        self.val_ds = DrivingDataset(
            config["val_index"],
            K=config.get("K", 8),
            H=config.get("chunk_len", 8),
            augment=False,
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=config.get("batch_size", 256),
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )

        # ── logging ────────────────────────────────────────────────────────
        self.out_dir  = Path(config["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "metrics.jsonl"
        self.step     = 0

        # save config
        with open(self.out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    # ── single training step ───────────────────────────────────────────────

    def _step(self, batch):
        frames        = batch["frames"].to(self.device)
        future_frames = batch["future_frames"].to(self.device)
        proprio       = batch["proprio"].to(self.device)
        command       = batch["command"].to(self.device)
        action_chunk  = batch["action_chunk"].to(self.device)

        pred_actions, h_t = self.policy(frames, proprio, command)
        bc_loss = F.smooth_l1_loss(pred_actions, action_chunk, beta=0.1)

        aux_loss = torch.tensor(0.0, device=self.device)
        if self.arm == "a1":
            aux_loss = self.aux_head.loss(
                h_t, action_chunk, future_frames,
                self.ema_encoder, self.online_proj, self.ema_proj,
            )
        elif self.arm == "a2":
            cur_frame = frames[:, -1:, :, :, :]
            with torch.no_grad():
                z_2nd = self.ema_encoder(cur_frame).squeeze(1)
            aux_loss = self.aux_head.loss(h_t, z_2nd, self.ema_proj)
        elif self.arm == "a3":
            aux_loss = self.aux_head.loss(
                h_t, future_frames, self.ema_encoder, self.ema_proj,
            )

        total_loss = bc_loss + self.lam * aux_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        if self.ema_encoder is not None:
            self.ema_encoder.update(self.policy.encoder)
            tau = 0.99
            for ep, op in zip(
                self.ema_proj.parameters(), self.online_proj.parameters()
            ):
                ep.data.mul_(tau).add_(op.data, alpha=1 - tau)

        return {
            "bc_loss":    bc_loss.item(),
            "aux_loss":   aux_loss.item(),
            "total_loss": total_loss.item(),
            "lr":         self.scheduler.get_last_lr()[0],
        }

    # ── validation ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self):
        self.policy.eval()
        total_bc = 0.0
        n = 0
        for batch in self.val_loader:
            frames       = batch["frames"].to(self.device)
            proprio      = batch["proprio"].to(self.device)
            command      = batch["command"].to(self.device)
            action_chunk = batch["action_chunk"].to(self.device)
            pred, _      = self.policy(frames, proprio, command)
            total_bc    += F.smooth_l1_loss(pred, action_chunk, beta=0.1).item()
            n           += 1
        self.policy.train()
        return total_bc / max(1, n)

    # ── checkpoint ─────────────────────────────────────────────────────────

    def _save(self, tag: str = ""):
        name = f"ckpt_{self.step:06d}{tag}.pt"
        torch.save({
            "step":      self.step,
            "policy":    self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config":    self.cfg,
        }, self.out_dir / name)

    def _log(self, metrics: dict):
        metrics["step"] = self.step
        metrics["time"] = time.time()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    # ── main training loop ─────────────────────────────────────────────────

    def train(self):
        print(f"Training arm={self.arm}, steps={self.total_steps}")
        self.policy.train()

        log_every  = 100
        val_every  = 1_000
        save_every = 5_000

        data_iter = iter(self.train_loader)
        t0 = time.time()

        while self.step < self.total_steps:
            # get next batch, restart loader if exhausted
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            metrics = self._step(batch)
            self.step += 1

            if self.step % log_every == 0:
                elapsed = time.time() - t0
                metrics["steps_per_sec"] = log_every / elapsed
                self._log(metrics)
                print(
                    f"step {self.step:06d} | "
                    f"bc={metrics['bc_loss']:.4f} "
                    f"aux={metrics['aux_loss']:.4f} "
                    f"lr={metrics['lr']:.2e} "
                    f"sps={metrics['steps_per_sec']:.1f}",
                    flush=True,
                )
                t0 = time.time()

            if self.step % val_every == 0:
                val_loss = self._validate()
                self._log({"val_bc_loss": val_loss})
                print(f"  → val bc_loss: {val_loss:.4f}")

            if self.step % save_every == 0:
                self._save()
                print(f"  → checkpoint saved at step {self.step}")

        # final save
        self._save(tag="_final")
        print(f"Training complete. Final checkpoint saved.")