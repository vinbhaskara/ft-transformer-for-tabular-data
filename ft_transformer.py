"""
FT-Transformer — PyTorch implementation for tabular data
=========================================================

Paper
-----
"Revisiting Deep Learning Models for Tabular Data"
Gorishniy, Rubachev, Khrulkov, Babenko — NeurIPS 2021
https://arxiv.org/abs/2106.11959
Official code: https://github.com/yandex-research/tabular-dl-revisiting-models

What is FT-Transformer?
-----------------------
FT-Transformer (Feature Tokenizer + Transformer) adapts the standard
Transformer architecture to tabular data.  The key idea is to embed every
scalar feature into its own d-dimensional token, then run a stack of
Transformer layers over those N feature tokens plus one special [CLS] token.
The final [CLS] representation is passed through a small MLP head to produce
the prediction.

Step-by-step:
  1. FeatureTokenizer  — each feature j gets its own weight vector W_j ∈ R^d
                         and bias b_j ∈ R^d.  The token is:
                             T_j = b_j + x_j * W_j        (Section 3.3)
  2. [CLS] token       — a learnable vector prepended to the N feature tokens.
  3. Transformer       — L layers of Multi-Head Self-Attention + FFN, PreNorm.
                         First layer omits the pre-MHSA LayerNorm (Appendix E.1).
  4. Prediction head   — Linear(ReLU(LayerNorm( T_L^[CLS] )))

Tokenization design — what lives in the d dimensions
-----------------------------------------------------
Each feature j produces a single d-dimensional token:

    T_j = b_j  +  x_j * W_j

  • b_j ∈ R^d  is the per-feature bias.  It depends only on the column
    index j, not on the observed value.  It acts as a learned "column
    identity" embedding — every sample, regardless of its value for
    feature j, starts from this same base vector.

  • x_j * W_j  shifts that base vector along a per-feature direction W_j
    by an amount proportional to the scalar value x_j.  This is the
    value-dependent part of the token.

  • The d dimensions are fully shared between column identity and value
    information.  They are superimposed (added), not concatenated.  There
    is no fixed partition such as "first k dims for identity, last d-k
    for value."  The network learns to use all d dims jointly.

  • There are NO positional embeddings.  In NLP, position matters because
    "bank" at position 3 vs 7 is different.  Tabular features have no
    meaningful order — feature j is always feature j.  Column identity is
    already baked into b_j and W_j which are indexed by j, not by sequence
    position.  Adding positional embeddings for tabular data is therefore
    not meaningful by default, and the paper does not use them.
    An experimental option (use_positional_embeddings=True) is provided
    for ablation purposes but is off by default.

  • The sequence length is always fixed at N+1 (N feature tokens + 1 CLS
    token), regardless of how many features are missing in a given sample.
    Missing features still occupy a token slot, but their embedding is the
    learned per-feature missing vector m_j rather than b_j + x_j*W_j.
    This means the model always processes a sequence of identical length,
    with no padding, bucketing, or variable-length handling required.

Differences from the paper in this implementation
--------------------------------------------------
1. Activation: the paper uses ReGLU (a gated activation) inside the FFN.
   This implementation uses GELU, which requires no changes to the FFN
   structure and is simpler to implement via nn.TransformerEncoderLayer.
   The paper itself notes: "we did not observe strong difference between
   ReGLU and ReLU in preliminary experiments."

2. Missing values: the paper does not address missing data — all benchmark
   datasets are fully observed.  This implementation adds optional support
   through a learned-missing-token strategy (see "Missing-value handling"
   below).  When no mask is supplied the model behaves exactly as described
   in the paper.

3. Categorical features: the paper embeds categorical features via lookup
   tables.  This implementation accepts only numerical tensors.  Encode
   categoricals (e.g. ordinal or learned embeddings) before passing them in.

4. Positional embeddings: the paper does not use them (see above).  This
   implementation exposes use_positional_embeddings=False as an opt-in
   experimental flag.

Missing-value handling
----------------------
Convention
~~~~~~~~~~
    input_missing_masks[b, j] == 1  →  feature j is MISSING for sample b
    input_missing_masks[b, j] == 0  →  feature j is OBSERVED for sample b

    The tensor may be bool or float.
    The value in input_features at a missing position is entirely ignored;
    pass any dummy (e.g. 0.0) there.

Strategy — per-feature learned missing token
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each feature j has a dedicated trainable vector m_j ∈ R^d.  When feature j
is missing for sample b, the normally computed token T[b, j] is replaced by
m_j in a single vectorised operation:

    token[b, j] = (1 - mask[b,j]) * (b_j + x[b,j] * W_j)
                +      mask[b,j]  * m_j

Rationale:
  • The sequence length stays fixed at N+1 — no padding or bucketing needed.
  • The [CLS] token attends to ALL N feature tokens, including missing ones.
    This lets the model use the *pattern* of which features are absent as a
    predictive signal, which is often informative in practice (e.g. in
    medical data, a missing lab test is not random).
  • Missingness is distinguishable from a zero-valued observation because
    m_j is a free parameter, independent of W_j and b_j.
  • Unlike hard attention masking (setting attention logits for missing
    positions to -inf), this approach lets the model decide how much
    to weight absent features rather than forcing it to ignore them.

Usage
-----
    from ft_transformer import FTTransformer
    import torch

    # --- recommended config for 32 features ---
    # Large model (~675K params) with stronger regularisation.
    # Go deeper rather than wider: short sequences (33 tokens) benefit more
    # from additional layers than from a larger embedding dimension.
    # Pair with AdamW(lr=1e-4, weight_decay=1e-4) during training.
    model = FTTransformer(
        n_features        = 32,
        d_token           = 128,   # embedding dim for every feature token
        n_layers          = 6,     # depth; each layer refines feature interactions
        n_heads           = 8,     # head_dim = 128/8 = 16 per attention head
        ffn_factor        = 4/3,   # FFN hidden = int(128 * 4/3) = 170
        attention_dropout = 0.3,   # drop attention edges; prevents co-adaptation
        ffn_dropout       = 0.2,   # drop inside FFN; main regularisation lever
        output_size       = 1,     # single logit; wrap in sigmoid for binary clf
    )

    B = 64   # batch size
    x       = torch.randn(B, 32)          # (B, N) feature values, pre-normalised
    mask    = torch.zeros(B, 32)          # (B, N) 0=observed, 1=missing
    mask[:, 5]  = 1                       # feature 5 is missing for all samples
    mask[0, 12] = 1                       # feature 12 missing only for sample 0

    # All features observed — mask is optional
    logits = model(x)                     # (B, 1)

    # With missing-value mask
    logits = model(x, mask)               # (B, 1)

    # Recommended optimizer (weight_decay omitted for tokenizer/norm/bias
    # layers ideally, but a flat value is a reasonable starting point)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

Requirements: torch >= 1.11  (for norm_first in TransformerEncoderLayer)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal: first Transformer layer (no pre-MHSA LayerNorm)
# ---------------------------------------------------------------------------

class _FirstTransformerLayer(nn.TransformerEncoderLayer):
    """
    Identical to nn.TransformerEncoderLayer(norm_first=True) except the
    LayerNorm before MHSA is omitted.

    From the paper (Appendix E.1):
      "we also found it to be necessary to remove the first normalization
       from the first Transformer layer to achieve good performance."
    """
    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        x = src
        x = x + self._sa_block(x, src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Feature Tokenizer
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """
    Converts a batch of scalar features into per-feature embedding vectors.

    For an observed feature j:
        T_j = b_j + x_j * W_j              (paper Section 3.3)

    For a missing feature j  (when missing_mask[b, j] == 1):
        T_j = m_j                           (learned missing token)

    Parameters
    ----------
    n_features : int
        Number of input features N.
    d_token : int
        Embedding dimension d.
    """
    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight        = nn.Parameter(torch.empty(n_features, d_token))
        self.bias          = nn.Parameter(torch.empty(n_features, d_token))
        self.missing_token = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight,        a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias,          a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.missing_token, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,                             # (B, N)
        missing_mask: Optional[torch.Tensor] = None, # (B, N) — 1 = missing
    ) -> torch.Tensor:                               # (B, N, d_token)
        # (B, N, 1) * (1, N, d) + (1, N, d)  →  (B, N, d)
        T = self.bias.unsqueeze(0) + x.unsqueeze(-1) * self.weight.unsqueeze(0)
        if missing_mask is not None:
            mask = missing_mask.unsqueeze(-1).to(dtype=T.dtype)       # (B, N, 1)
            T = T * (1.0 - mask) + self.missing_token.unsqueeze(0) * mask
        return T


# ---------------------------------------------------------------------------
# FT-Transformer
# ---------------------------------------------------------------------------

class FTTransformer(nn.Module):
    """
    FT-Transformer (Feature Tokenizer + Transformer) for tabular data.

    See module docstring for full background, paper reference, and notes on
    differences from the original paper.

    Parameters
    ----------
    n_features : int
        Number of input features.  All are treated as numerical.
    d_token : int, default 192
        Embedding / model dimension.  Must be divisible by n_heads.
    n_layers : int, default 3
        Number of Transformer layers.
    n_heads : int, default 8
        Number of self-attention heads.
    ffn_factor : float, default 4/3
        Ratio of the FFN hidden dimension to d_token.
        ffn_hidden = int(d_token * ffn_factor).
    attention_dropout : float, default 0.2
        Dropout probability applied inside MultiheadAttention.
    ffn_dropout : float, default 0.1
        Dropout probability applied inside the FFN and on residual branches.
    output_size : int, default 1
        Dimensionality of the output.  Use 1 for binary classification or
        regression; use C for C-class classification (then apply softmax /
        cross-entropy externally).
    missing_indication_value : int, default 1
        Which value in input_missing_masks encodes "this feature is missing".
        Must be 0 or 1.
            1 (default) — the mask contains 1 where a feature is absent.
                          e.g. mask = (df.isna().astype(int))
            0           — the mask contains 0 where a feature is absent,
                          i.e. it is an *observation* mask / validity mask.
                          e.g. mask = (~df.isna()).astype(int)
        Internally the mask is always normalised to the convention
        "1 = missing" before use, so either format is handled correctly.
    use_positional_embeddings : bool, default False
        If True, adds a learnable positional embedding to the full token
        sequence (all N feature tokens + the CLS token) immediately after
        the CLS token is prepended.  Each of the N+1 positions gets its
        own d-dimensional vector, initialised from N(0, 0.02).
        This is OFF by default and is not recommended for typical tabular
        use: feature order is arbitrary, and column identity is already
        encoded by the per-feature parameters b_j and W_j in the
        FeatureTokenizer.  Provided as an experimental ablation option.

    Inputs
    ------
    input_features : Tensor, shape (B, N)
        Raw (preprocessed) feature values.  Recommended to apply quantile
        normalisation or standardisation before passing in, as done in the
        paper (Section 4.3).  Values at missing positions are ignored.
    input_missing_masks : Tensor, shape (B, N), optional
        Binary mask in the format declared by missing_indication_value.
        May be bool or float dtype.  When omitted, all features are treated
        as observed.

    Returns
    -------
    Tensor, shape (B, output_size)
        Raw output logits or regression values.  No activation is applied;
        wrap in sigmoid / softmax externally as appropriate.
    """

    def __init__(
        self,
        n_features:               int,
        d_token:                  int   = 192,
        n_layers:                 int   = 3,
        n_heads:                  int   = 8,
        ffn_factor:               float = 4.0 / 3.0,
        attention_dropout:        float = 0.2,
        ffn_dropout:              float = 0.1,
        output_size:              int   = 1,
        missing_indication_value: int   = 1,
        use_positional_embeddings: bool = False,
    ) -> None:
        assert missing_indication_value in (0, 1), \
            "missing_indication_value must be 0 or 1"
        super().__init__()

        self.missing_indication_value  = missing_indication_value
        self.use_positional_embeddings = use_positional_embeddings
        self.tokenizer = FeatureTokenizer(n_features, d_token)

        # Optional learned positional embeddings — one vector per sequence
        # position (N feature slots + 1 CLS slot).  Not used by default.
        if use_positional_embeddings:
            self.pos_embedding = nn.Parameter(
                torch.empty(1, n_features + 1, d_token)
            )
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))

        layer_kwargs = dict(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=int(d_token * ffn_factor),
            dropout=ffn_dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        # First layer omits the pre-MHSA LayerNorm (paper Appendix E.1).
        layers = [_FirstTransformerLayer(**layer_kwargs)]
        layers += [nn.TransformerEncoderLayer(**layer_kwargs) for _ in range(n_layers - 1)]

        # attention_dropout lives as a plain float on nn.MultiheadAttention
        # and is read during its forward pass; set it after construction.
        for layer in layers:
            layer.self_attn.dropout = attention_dropout

        self.layers    = nn.ModuleList(layers)
        self.head_norm = nn.LayerNorm(d_token)
        self.head      = nn.Linear(d_token, output_size)

        self._print_summary(n_features, d_token, n_layers, n_heads,
                            ffn_factor, attention_dropout, ffn_dropout,
                            output_size, missing_indication_value,
                            use_positional_embeddings)

    # ------------------------------------------------------------------

    def _print_summary(self, n_features, d_token, n_layers, n_heads,
                       ffn_factor, attention_dropout, ffn_dropout,
                       output_size, missing_indication_value,
                       use_positional_embeddings):
        total = sum(p.numel() for p in self.parameters())
        miss_hint = "1=missing" if missing_indication_value == 1 else "0=missing"
        print("=" * 57)
        print("FT-Transformer  (arXiv:2106.11959)")
        print("=" * 57)
        print(f"  n_features               : {n_features}")
        print(f"  d_token                  : {d_token}  "
              f"(seq len = {n_features + 1}: {n_features} features + 1 CLS)")
        print(f"  n_layers                 : {n_layers}")
        print(f"  n_heads                  : {n_heads}  "
              f"(head_dim = {d_token // n_heads})")
        print(f"  ffn_factor               : {ffn_factor:.4g}  "
              f"(ffn_hidden = {int(d_token * ffn_factor)})")
        print(f"  attention_dropout        : {attention_dropout}")
        print(f"  ffn_dropout              : {ffn_dropout}")
        print(f"  output_size              : {output_size}")
        print(f"  missing_indication_value : {missing_indication_value}  ({miss_hint})")
        print(f"  positional_embeddings    : {use_positional_embeddings}")
        print("-" * 57)
        print(f"  Learnable params         : {total:,}")
        print("=" * 57)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_features:      torch.Tensor,                   # (B, N)
        input_missing_masks: Optional[torch.Tensor] = None,  # (B, N)
    ) -> torch.Tensor:                                       # (B, output_size)
        B = input_features.size(0)

        # Normalise mask to internal convention: 1 = missing.
        # If the user declared missing_indication_value=0 (observation mask),
        # flip it so that absent features map to 1 before the tokenizer sees it.
        mask = input_missing_masks
        if mask is not None and self.missing_indication_value == 0:
            mask = 1.0 - mask.float()

        # Tokenise: (B, N, d).  Missing positions receive learned token m_j.
        tokens = self.tokenizer(input_features, mask)

        # Prepend [CLS]: (B, N+1, d)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)

        # Optional positional embeddings (off by default — see module docstring).
        if self.use_positional_embeddings:
            tokens = tokens + self.pos_embedding

        # [CLS] attends freely to all N feature tokens, including missing ones.
        # Missing tokens carry their learned m_j, so the model can use the
        # pattern of absent features as a predictive signal.
        for layer in self.layers:
            tokens = layer(tokens)

        # Prediction head:  Linear(ReLU(LayerNorm( cls_repr )))
        return self.head(F.relu(self.head_norm(tokens[:, 0])))
