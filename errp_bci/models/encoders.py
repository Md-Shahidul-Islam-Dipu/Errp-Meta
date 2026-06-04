"""Base EEG encoder architecture (ported verbatim).

Auto-ported verbatim from the legacy notebooks by _build_pkg.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGEncoder(nn.Module):
    """
    Meta-learned EEG feature encoder.

    Architecture: input → 256 → 128 → 64 (representation)
    Each hidden layer: Linear → LayerNorm → GELU → Dropout(0.3)
    Final layer:       Linear  [NO activation, NO dropout — unconstrained embedding]

    Rationale:
    - LayerNorm: makes activations amplitude-invariant across subjects
    - GELU: better gradient flow than ReLU for meta-learning outer loops
    - No final activation: unconstrained embedding (ReLU would discard ~50% of space)
    - Size 256→128→64: capacity without over-parameterization at 32-dim PCA input
    """
    def __init__(self, input_dim: int,
                 h1: int = 256, h2: int = 128, out_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.ln1 = nn.LayerNorm(h1)
        self.dp1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(h1, h2)
        self.ln2 = nn.LayerNorm(h2)
        self.dp2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(h2, out_dim)    # representation layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dp1(F.gelu(self.ln1(self.fc1(x))))
        x = self.dp2(F.gelu(self.ln2(self.fc2(x))))
        return self.fc3(x)

class SimpleTaskHead(nn.Module):
    """Linear classifier on encoder representation. Adapted per-subject in inner loop."""
    def __init__(self, input_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class MetaEEGClassifier(nn.Module):
    """Full model: EEGEncoder + SimpleTaskHead. Used by MAML, ANIL, Reptile."""
    def __init__(self, input_dim: int,
                 h1: int = 256, h2: int = 128, enc_out: int = 64,
                 dropout: float = 0.3, num_classes: int = 2):
        super().__init__()
        self.encoder   = EEGEncoder(input_dim, h1, h2, enc_out, dropout)
        self.task_head = SimpleTaskHead(enc_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.task_head(self.encoder(x))

    def get_repr(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder representation without classification."""
        return self.encoder(x)
