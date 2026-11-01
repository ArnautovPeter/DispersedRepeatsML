"""NT-v2 token representations, content-token pooling, and contrastive head."""

from contextlib import nullcontext
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.data import revcomp


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep CPU BLAS from oversubscribing small evaluation workloads.
    torch.set_num_threads(4)


def masked_mean(hidden, attention, special):
    mask = attention.bool() & ~special.bool()
    if not mask.any(dim=1).all():
        raise ValueError("Sequence contains no content tokens")

    weights = mask.unsqueeze(-1).to(hidden.dtype)

    # Accumulate in float32 under mixed precision.
    return (hidden.float() * weights).sum(1) / weights.sum(1).float()


def supcon_loss(z, labels, temperature=0.1):
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    labels = torch.as_tensor(labels, device=z.device)
    diag = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diag
    if not positive.any(1).all():
        raise ValueError("Every anchor needs an independent positive copy")

    logits = (
        F.normalize(z.float(), dim=-1) @ F.normalize(z.float(), dim=-1).T / temperature
    )
    denominator = torch.logsumexp(logits.masked_fill(diag, -torch.inf), dim=1)
    log_prob = logits - denominator[:, None]

    return -(log_prob.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()


class RepeatEncoder(nn.Module):
    def __init__(self, model_dir="data/model", device=None):
        super().__init__()
        self.model_dir = str(Path(model_dir).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True, local_files_only=True
        )

        # The published custom config exposes AutoModelForMaskedLM, not AutoModel.
        # Discard the MLM head and call .esm directly; never compute vocabulary logits.
        mlm = AutoModelForMaskedLM.from_pretrained(
            self.model_dir, trust_remote_code=True, local_files_only=True
        )
        self.backbone = mlm.esm
        self.head = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device_name)
        self.set_stage("projection")

    @property
    def device(self):
        return next(self.parameters()).device

    def autocast(self):
        return (
            torch.autocast("cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )

    def set_stage(self, stage):
        if stage not in ("projection", "finetune"):
            raise ValueError(stage)

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if stage == "finetune":
            layers = self.backbone.encoder.layer
            if len(layers) < 2:
                raise ValueError("Backbone must have at least two blocks")
            for block in layers[-2:]:
                for p in block.parameters():
                    p.requires_grad_(True)

        for p in self.head.parameters():
            p.requires_grad_(True)

        self.stage = stage

    def tokenize(self, sequences):
        if any(not s or set(s.upper()) - set("ACGTN") for s in sequences):
            raise ValueError("Expected nonempty DNA strings (ACGTN)")

        batch = self.tokenizer(
            [s.upper() for s in sequences],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        # EsmTokenizer.get_special_tokens_mask assumes EOS even when this NT
        # tokenizer has no EOS, producing a mask one token too long. Derive it
        # from actual IDs so CLS/PAD/UNK/MASK are always excluded correctly.
        batch["special_tokens_mask"] = torch.zeros_like(
            batch["input_ids"], dtype=torch.bool
        )
        for token_id in self.tokenizer.all_special_ids:
            batch["special_tokens_mask"] |= batch["input_ids"].eq(token_id)

        lengths = batch["attention_mask"].sum(1)
        limit = min(
            self.tokenizer.model_max_length,
            self.backbone.config.max_position_embeddings - 2,
        )
        if (lengths > limit).any():
            raise ValueError(
                f"Token length {int(lengths.max())} exceeds {limit}; truncation is forbidden"
            )

        return {
            k: v.to(self.device)
            for k, v in batch.items()
            if k in ("input_ids", "attention_mask", "special_tokens_mask")
        }

    def forward(self, sequences, project=True):
        batch = self.tokenize(sequences)
        special = batch.pop("special_tokens_mask")

        with self.autocast():
            hidden = self.backbone(**batch, return_dict=True).last_hidden_state
            pooled = masked_mean(hidden, batch["attention_mask"], special)
            z = self.head(pooled) if project else pooled

        return F.normalize(z.float(), dim=-1)

    @torch.no_grad()
    def embed(self, sequences, batch_size=8, project=True, rc_average=True):
        self.eval()
        vectors = []

        for i in range(0, len(sequences), batch_size):
            ss = sequences[i : i + batch_size]
            z = self(ss, project=project)
            if rc_average:
                z = F.normalize(
                    z + self([revcomp(s) for s in ss], project=project), dim=-1
                )
            vectors.append(z.cpu().numpy())

        return (
            np.concatenate(vectors)
            if vectors
            else np.zeros((0, 128 if project else self.backbone.config.hidden_size))
        )

    def save(self, path, metadata):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "metadata": metadata,
                "model_source": json.loads(
                    (Path(self.model_dir) / "source.json").read_text()
                ),
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        current = json.loads((Path(self.model_dir) / "source.json").read_text())
        if current != ckpt["model_source"]:
            raise ValueError("Checkpoint/model source mismatch")

        self.load_state_dict(ckpt["state_dict"])
        return ckpt["metadata"]
