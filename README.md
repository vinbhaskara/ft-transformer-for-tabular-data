# FT-Transformer

A clean, single-file PyTorch implementation of the **FT-Transformer** (Feature Tokenizer + Transformer) for tabular data, with added support for missing values.

Based on the NeurIPS 2021 paper:

> **Revisiting Deep Learning Models for Tabular Data**  
> Gorishniy, Rubachev, Khrulkov, Babenko — NeurIPS 2021  
> [https://arxiv.org/abs/2106.11959](https://arxiv.org/abs/2106.11959)  
> Official code: [yandex-research/tabular-dl-revisiting-models](https://github.com/yandex-research/tabular-dl-revisiting-models)

---

## What is FT-Transformer?

FT-Transformer adapts the standard Transformer architecture to tabular data. Instead of treating a row of features as a flat vector (as an MLP would), it embeds each feature into its own token and runs self-attention over those tokens. This allows the model to explicitly learn interactions between every pair of features at every layer.

**Step-by-step forward pass:**

1. **Feature Tokenizer** — each scalar feature `j` is mapped to a `d`-dimensional token:
   ```
   T_j = b_j + x_j * W_j
   ```
   where `b_j` is a per-feature bias (column identity embedding) and `W_j` is a per-feature weight vector (value direction). Both are learned. The `d` dimensions are fully shared — there is no partition between "identity dims" and "value dims".

2. **[CLS] token** — a learnable vector is prepended to the `N` feature tokens, giving a sequence of length `N+1`.

3. **Transformer layers** — `L` layers of Multi-Head Self-Attention + Feed-Forward Network (PreNorm variant). The first layer omits the pre-attention LayerNorm, as the paper recommends.

4. **Prediction head** — the final [CLS] representation is passed through:
   ```
   ŷ = Linear(ReLU(LayerNorm(T_L^[CLS])))
   ```

---

## Key design notes

### No positional embeddings
Unlike NLP Transformers, FT-Transformer uses **no positional embeddings**. Tabular features have no meaningful order — feature `j` is always feature `j`. Its identity is already encoded by the per-feature parameters `b_j` and `W_j`, which are indexed by column, not by sequence position. An experimental `use_positional_embeddings=True` flag is available for ablation.

### Fixed sequence length
The sequence is always exactly `N+1` tokens long, regardless of how many features are missing. Missing features still occupy a token slot — just with a different learned embedding. This means no padding, no bucketing, and no variable-length handling is needed.

### Missing values (extension beyond the paper)
The paper does not address missing data. This implementation adds support via a **per-feature learned missing token** `m_j`. When feature `j` is missing for a sample, its token is replaced by `m_j` instead of `b_j + x_j * W_j`. The [CLS] token attends to all tokens including missing ones, letting the model use the *pattern* of missingness as a predictive signal.

---

## Differences from the paper

| Aspect | Paper | This implementation |
|---|---|---|
| FFN activation | ReGLU | GELU |
| Missing values | Not addressed | Learned per-feature missing token |
| Categorical features | Lookup table embeddings | Not supported; encode externally |
| Positional embeddings | Not used | Not used by default; opt-in flag |

On activation: the paper itself notes *"we did not observe strong difference between ReGLU and ReLU in preliminary experiments."* GELU is used here for simplicity via `nn.TransformerEncoderLayer`.

---

## Requirements

```
torch >= 1.11
```

Install PyTorch for your platform from [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).

```bash
pip install -r requirements.txt
```

---

## Quick start

```python
from ft_transformer import FTTransformer
import torch

# Large model (~675K params) with strong regularisation — good starting point
# for 32 features. Go deeper rather than wider: the sequence is short (33
# tokens), so depth matters more than embedding width.
model = FTTransformer(
    n_features        = 32,
    d_token           = 128,   # embedding dim per feature token
    n_layers          = 6,     # transformer depth
    n_heads           = 8,     # head_dim = 128 / 8 = 16
    ffn_factor        = 4/3,   # FFN hidden = int(128 * 4/3) = 170
    attention_dropout = 0.3,   # regularise attention edges
    ffn_dropout       = 0.2,   # regularise FFN
    output_size       = 1,     # single logit for binary clf / regression
)
# Prints architecture summary and total parameter count on init.

B = 64
x    = torch.randn(B, 32)   # (B, N) — pre-normalise features before passing in
mask = torch.zeros(B, 32)   # (B, N) — 0 = observed, 1 = missing
mask[:, 5]  = 1              # feature 5 missing for all samples
mask[0, 12] = 1              # feature 12 missing only for sample 0

logits = model(x)            # (B, 1) — no missing values
logits = model(x, mask)      # (B, 1) — with missing-value mask

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

---

## API reference

### `FTTransformer(n_features, ...)`

| Argument | Type | Default | Description |
|---|---|---|---|
| `n_features` | `int` | — | Number of input features. All treated as numerical. |
| `d_token` | `int` | `192` | Embedding dimension. Must be divisible by `n_heads`. |
| `n_layers` | `int` | `3` | Number of Transformer layers. |
| `n_heads` | `int` | `8` | Number of attention heads. `head_dim = d_token / n_heads`. |
| `ffn_factor` | `float` | `4/3` | FFN hidden size ratio. `ffn_hidden = int(d_token * ffn_factor)`. |
| `attention_dropout` | `float` | `0.2` | Dropout inside MultiheadAttention. |
| `ffn_dropout` | `float` | `0.1` | Dropout inside FFN and on residual branches. |
| `output_size` | `int` | `1` | Output dimensionality. Use `C` for C-class classification. |
| `missing_indication_value` | `int` | `1` | Whether `1` or `0` encodes missingness in the mask (see below). |
| `use_positional_embeddings` | `bool` | `False` | Add learned positional embeddings. Not recommended for tabular data. |

### `forward(input_features, input_missing_masks=None)`

| Argument | Shape | Description |
|---|---|---|
| `input_features` | `(B, N)` | Float tensor of feature values. Quantile-normalise or standardise beforehand. Values at missing positions are ignored. |
| `input_missing_masks` | `(B, N)` | Optional binary mask. Convention set by `missing_indication_value`. |

Returns `(B, output_size)` — raw logits with no activation applied.

---

## Missing-value mask convention

The `missing_indication_value` argument controls which value means "missing":

```python
# Default: 1 = missing  (matches df.isna())
model = FTTransformer(n_features=32, missing_indication_value=1)
mask = torch.tensor(df.isna().values, dtype=torch.float32)

# Alternative: 0 = missing  (observation / validity mask)
model = FTTransformer(n_features=32, missing_indication_value=0)
mask = torch.tensor((~df.isna()).values, dtype=torch.float32)
```

Internally the mask is always normalised to `1 = missing` before use.

---

## Recommended training setup

```python
# AdamW with moderate weight decay.
# Ideally exclude tokenizer weights, LayerNorm params, and biases from
# weight decay (as the paper does), but a flat value is a reasonable start.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Binary classification
loss = torch.nn.BCEWithLogitsLoss()(logits.squeeze(1), targets.float())

# Regression
loss = torch.nn.MSELoss()(logits.squeeze(1), targets.float())

# Multi-class (output_size = C)
loss = torch.nn.CrossEntropyLoss()(logits, targets.long())
```

Pre-process numerical features before passing them in. The paper uses **quantile normalisation** (via `sklearn.preprocessing.QuantileTransformer`) by default, with standardisation for some datasets. Raw features with very different scales will hurt performance.

---

## Parameter count guide

For reference, approximate parameter counts for common configurations with `n_features=32`:

| `d_token` | `n_layers` | `n_heads` | ~Params |
|---|---|---|---|
| 32 | 2 | 4 | 17K |
| 48 | 1 | 4 | 21K |
| 128 | 3 | 8 | 344K |
| 128 | 6 | 8 | 675K |
| 192 | 3 | 8 | 762K |

The dominant cost is the Transformer layers (each scales as `O(d²)`). The FeatureTokenizer cost is small and scales as `O(N × d)`.

---

## Citation

```bibtex
@inproceedings{gorishniy2021revisiting,
  title     = {Revisiting Deep Learning Models for Tabular Data},
  author    = {Gorishniy, Yury and Rubachev, Ivan and Khrulkov, Valentin and Babenko, Artem},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.11959}
}
```
