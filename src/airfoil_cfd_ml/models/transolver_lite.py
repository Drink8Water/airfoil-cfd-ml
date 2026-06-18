"""Transolver-lite: physics-aware transformer for PDE surrogate modelling.

Inspired by Wu et al., "Transolver: A Fast and Accurate Transformer-Based
PDE Solver", 2024.  **This is NOT an official reproduction.**  It is a
lightweight variant that uses:
  - Flattened grid points with spatial + physical features.
  - K learnable physics tokens.
  - Point↔Token cross-attention + Token self-attention.
  - Point-wise FFN for decoding.

Memory-aware: K=64 by default to keep cross-attention matrices manageable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


# ---------------------------------------------------------------------------
# Attention utilities
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer MLP with GELU and optional dropout."""

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """Cross-attention: Q attends to K-V pairs.  No causal mask."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention: Q attends to K-V.

        Args:
            query: (B, Nq, dim)
            key:   (B, Nk, dim)
            value: (B, Nk, dim)

        Returns:
            (B, Nq, dim)
        """
        B, Nq, _ = query.shape
        Nk = key.shape[1]

        q = self.wq(query).view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(key).view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(value).view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nk)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, H, Nq, head_dim)
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return self.wo(out)


class SelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = CrossAttention(dim, n_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cross_attn(x, x, x)


class TransformerBlock(nn.Module):
    """Self-attention + FFN with pre-norm and residual connections."""

    def __init__(self, dim: int, n_heads: int = 4, ff_expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, ff_expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# TransolverLite
# ---------------------------------------------------------------------------

@register_model("transolver_lite")
class TransolverLite(nn.Module):
    """Lightweight Transolver-inspired model for airfoil flow prediction.

    Pixels are treated as "points" with physical + spatial features.
    K learnable physics tokens aggregate information from points via
    cross-attention, interact via self-attention, then distribute
    information back to points via cross-attention.

    Args:
        in_channels: total input channels (physical + geometry).
        d_model: internal embedding dimension.
        K: number of physics tokens (default 64).
        n_heads: attention heads.
        n_layers: number of Transolver layers.
        ff_expansion: expansion factor for point-wise FFN.
        dropout: dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 128,
        K: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.K = K
        self.d_model = d_model

        # Point embedding: project input channels → d_model
        self.point_embed = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable physics tokens
        self.tokens = nn.Parameter(torch.randn(1, K, d_model) * 0.02)

        # Transolver layers
        self.layers = nn.ModuleList([
            _TransolverLayer(d_model, n_heads, ff_expansion, dropout)
            for _ in range(n_layers)
        ])

        # Output projection: d_model → 3
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W).

        Returns:
            (B, 3, H, W).
        """
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dimensions: (B, C, H, W) → (B, N, C)
        points = x.permute(0, 2, 3, 1).reshape(B, N, C)

        # Embed points
        point_feat = self.point_embed(points)  # (B, N, d_model)

        # Expand tokens to batch
        tokens = self.tokens.expand(B, -1, -1)  # (B, K, d_model)

        # Transolver layers
        for layer in self.layers:
            tokens, point_feat = layer(tokens, point_feat)

        # Decode points to 3 output channels
        out = self.head(point_feat)  # (B, N, 3)

        # Reshape back to spatial: (B, N, 3) → (B, 3, H, W)
        out = out.reshape(B, H, W, 3).permute(0, 3, 1, 2)
        return out


class _TransolverLayer(nn.Module):
    """One Transolver layer:

       1. Point → Token: cross-attention (tokens attend to points).
       2. Token self-attention.
       3. Token → Point: cross-attention (points attend to tokens).
       4. Point FFN.
    """

    def __init__(self, dim: int, n_heads: int, ff_expansion: int, dropout: float):
        super().__init__()

        # Point → Token cross-attention
        self.p2t_norm1 = nn.LayerNorm(dim)
        self.p2t_norm2 = nn.LayerNorm(dim)
        self.p2t_attn = CrossAttention(dim, n_heads, dropout)

        # Token self-attention
        self.token_block = TransformerBlock(dim, n_heads, ff_expansion, dropout)

        # Token → Point cross-attention
        self.t2p_norm1 = nn.LayerNorm(dim)
        self.t2p_norm2 = nn.LayerNorm(dim)
        self.t2p_attn = CrossAttention(dim, n_heads, dropout)

        # Point FFN
        self.point_norm = nn.LayerNorm(dim)
        self.point_ffn = MLP(dim, ff_expansion, dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Point → Token
        tokens = tokens + self.p2t_attn(
            self.p2t_norm1(tokens), self.p2t_norm2(points), points
        )

        # 2. Token self-attention
        tokens = self.token_block(tokens)

        # 3. Token → Point
        points = points + self.t2p_attn(
            self.t2p_norm1(points), self.t2p_norm2(tokens), tokens
        )

        # 4. Point FFN
        points = points + self.point_ffn(self.point_norm(points))

        return tokens, points
