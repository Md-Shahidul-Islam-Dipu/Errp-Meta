"""Label-aware subject encoder + FiLM conditioning (+ NoFiLM variant).

Auto-ported verbatim from the legacy notebooks by _build_pkg.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import Config
from .encoders import EEGEncoder, SimpleTaskHead, MetaEEGClassifier

class LabelAwareSubjectEncoder(nn.Module):
    """
    Compute subject embedding z_s from support set using label-aware prototypes.

    Input : support features (K, input_dim) + support labels (K,)
    Output: subject embedding z_s (embed_dim,)

    Three prototype vectors are concatenated as input:
        [proto_error, proto_correct, diff]  →  MLP  →  z_s
    This gives the encoder an explicit discriminative signal (diff) in addition
    to the class-conditional mean embeddings.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 embed_dim: int = Config.SUBJECT_EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, support_features: torch.Tensor,
                support_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support_features : (K, input_dim)
            support_labels   : (K,) integer labels 0/1
        Returns:
            z_s : (embed_dim,)
        """
        err_mask  = support_labels == 1
        corr_mask = support_labels == 0

        proto_err  = (support_features[err_mask].mean(0)
                      if err_mask.sum() > 0 else support_features.mean(0))
        proto_corr = (support_features[corr_mask].mean(0)
                      if corr_mask.sum() > 0 else support_features.mean(0))
        diff = proto_err - proto_corr

        pooled = torch.cat([proto_err, proto_corr, diff], dim=-1)   # (3 * input_dim,)
        return self.encoder(pooled)

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation: h' = LayerNorm(h) * gamma(z_s) + beta(z_s)

    LayerNorm applied BEFORE subject-specific scaling stabilizes activations.
    embed_dim is always Config.SUBJECT_EMBED_DIM = 32 (consistent everywhere).
    """
    def __init__(self, hidden_dim: int, embed_dim: int = Config.SUBJECT_EMBED_DIM):
        super().__init__()
        self.norm     = nn.LayerNorm(hidden_dim)
        self.gamma_fc = nn.Linear(embed_dim, hidden_dim)
        self.beta_fc  = nn.Linear(embed_dim, hidden_dim)

    def forward(self, h: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        return self.gamma_fc(z_s) * self.norm(h) + self.beta_fc(z_s)

class ConditionedEEGEncoder(nn.Module):
    """
    EEG encoder with FiLM conditioning from subject embedding z_s.

    Architecture:
        fc1(input → hidden) → FiLM(z_s) → GELU → Dropout
        fc2(hidden → hidden) → FiLM(z_s) → GELU → Dropout
        fc3(hidden → output)   [no activation]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 output_dim: int = 64,
                 embed_dim: int = Config.SUBJECT_EMBED_DIM):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.film1 = FiLMLayer(hidden_dim, embed_dim)
        self.dp1   = nn.Dropout(0.2)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.film2 = FiLMLayer(hidden_dim, embed_dim)
        self.dp2   = nn.Dropout(0.2)
        self.fc3   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        h = self.dp1(F.gelu(self.film1(self.fc1(x), z_s)))
        h = self.dp2(F.gelu(self.film2(self.fc2(h), z_s)))
        return self.fc3(h)

class TaskHead(nn.Module):
    """Linear classifier head — adapted per subject in inner loop."""
    def __init__(self, input_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)

class UnconditionedEEGEncoder(nn.Module):
    """
    Ablation encoder for Ablation 1: FiLM conditioning removed.

    Derived directly from ConditionedEEGEncoder (cell 25) by replacing
    each FiLMLayer with a plain LayerNorm. This is the minimal possible change:

    FiLMLayer computes:  LayerNorm(h) * gamma(z_s) + beta(z_s)
    Without conditioning: LayerNorm(h) * 1 + 0  =  LayerNorm(h)

    This isolates exactly the subject-conditioned scale/shift.
    Depth, width, GELU, Dropout, and LayerNorm are all preserved.

    Original ConditionedEEGEncoder forward:
        h = dp1(gelu(film1(fc1(x), z_s)))   # FiLM1: LN(h)*gamma(z_s)+beta(z_s)
        h = dp2(gelu(film2(fc2(h), z_s)))   # FiLM2: LN(h)*gamma(z_s)+beta(z_s)
        return fc3(h)

    This encoder forward:
        h = dp1(gelu(ln1(fc1(x))))           # plain LayerNorm, no z_s
        h = dp2(gelu(ln2(fc2(h))))           # plain LayerNorm, no z_s
        return fc3(h)

    Does NOT take z_s — call as encoder(x), not encoder(x, z_s).
    """
    def __init__(self, input_dim: int,
                 hidden_dim: int = Config.ENCODER_HIDDEN2,
                 output_dim: int = Config.ENCODER_OUTPUT,
                 embed_dim: int = Config.SUBJECT_EMBED_DIM):  # kept for signature parity
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)               # replaces FiLMLayer1
        self.dp1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)               # replaces FiLMLayer2
        self.dp2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dp1(F.gelu(self.ln1(self.fc1(x))))
        h = self.dp2(F.gelu(self.ln2(self.fc2(h))))
        return self.fc3(h)
