# Chapter 3 — Attention in Genomics

> *"Self-attention is a learned alignment over the genome."*

## Learning objectives

- Derive scaled dot-product attention and explain its computational complexity.
- Describe at least three ways genomic models extend vanilla attention to handle long sequences: sparse attention, linear / kernel attention, state-space substitutes.
- Interpret attention maps as candidate regulatory contacts and quantify their concordance with Hi-C / Micro-C data.
- Identify when attention is the wrong tool (e.g. when explicit physical priors are available).

## 3.1  Scaled dot-product attention recap

For queries `Q ∈ ℝ^(n×d)`, keys `K ∈ ℝ^(n×d)`, values `V ∈ ℝ^(n×d_v)`:

```
A = softmax( Q Kᵀ / √d ) V
```

Complexity: **O(n²·d)** in time and memory. For a 1 Mbp window at single-base resolution this is intractable; this is the central engineering challenge of genomic transformers.

### 3.1a  A concrete derivation of attention's complexity and its implications

The claim that full quadratic attention `O(n²·d)` is intractable for 1 Mbp at base resolution becomes vivid with a calculation:

- **Single-base resolution** for 1 Mbp: `n = 10⁶`.
- `n² = 10¹²`. Even with `d = 64` (small), the attention matrix has `10¹²` entries. At 4 bytes per float, that is **4 TB** just for the attention logits — per layer, per head. No single GPU (even with 80 GB HBM) can hold this.
- **Downsampling** to 128 bp bins reduces `n` to ~7800, `n² ≈ 6 × 10⁷`, manageable (~240 MB per matrix). This is why practically all genomic transformers downsample early via convolutional stems.

**Practical rule of thumb.** For raw DNA sequence, quadratic attention becomes infeasible beyond ~50 kbp at single-base resolution. For 100 kbp, `n² = 10¹⁰` → 40 GB per matrix → borderline but possible with sparse approximations. For 1 Mbp, impossible without downsampling or linear attention.

This complexity drives the adoption of **state-space models** (Mamba, Hyena), which achieve `O(n log n)` or `O(n)` complexity. The trade-off is reduced ability to learn arbitrary pairwise interactions; SSMs are good at capturing long-range dependencies but may miss certain "all-pairs" patterns.

```python
import math

def attention_memory_bytes(n: int, d: int = 64, n_heads: int = 8, dtype_bytes: int = 4) -> dict:
    """
    Estimate memory usage of full softmax attention.
    Returns dict with QK^T matrix and total per layer.
    """
    qk_matrix = n * n * dtype_bytes              # QK^T matrix
    attn_output = n * d * dtype_bytes * n_heads  # after softmax (typically same as QK^T)
    total_per_layer = qk_matrix + attn_output
    return {
        "qk_matrix_GB": qk_matrix / 1e9,
        "total_per_layer_GB": total_per_layer / 1e9,
        "n_heads": n_heads,
    }

print(attention_memory_bytes(50_000))   # 50 kbp: 10 GB QK^T -> infeasible
print(attention_memory_bytes(5_000))    # 5 kbp: 0.1 GB -> OK
```

> **Pitfall.** Even after downsampling, the QK^T matrix is often the memory bottleneck. Use gradient checkpointing to recompute attention during the backward pass, trading compute for memory.

## 3.2  Long-range mechanisms in production models

| Model | Receptive field | Trick |
|-------|------------------|-------|
| Basenji2 | 131 kbp | Dilated convolutions, no attention |
| Enformer | 196 kbp | Stem CNN → 11-layer transformer, relative positions |
| Borzoi | 524 kbp | Enformer + RNA-seq prediction |
| HyenaDNA | 1 Mbp | Implicit long convolutions (Hyena operator) |
| Caduceus | 131 kbp | Reverse-complement-equivariant Mamba |

The pattern across models is to push compute into a **convolutional stem** that down-samples to ~128 bp resolution, then apply attention or an attention substitute at the coarse scale.

### 3.2a  Long-range mechanisms in detail — a decision tree for choosing a mechanism

A practical guide to **which long-range mechanism to use**:

| Genomic task / property | Recommended mechanism | Why |
|--------------------------|----------------------|-----|
| Promoter–enhancer prediction (known interactions up to 1 Mbp) | Sparse attention (BigBird, Longformer) with sliding window + global tokens | Balanced; locality suffices for most, global tokens capture key elements |
| Whole-genome chromatin-state segmentation (e.g. 5 Mbp windows) | Convolutional stem + linear attention (Performer) | Linear in sequence length; good for very long sequences with uniform importance |
| Regulatory variant scoring at single-base resolution (short windows, ~20 kbp) | Full dense attention | Quadratic is fine for short sequences; gives maximal expressivity |
| Very long-range looping (Hi-C) across megabases | State-space (Mamba, Hyena) or hierarchical attention | Sub-quadratic with strong long-range modeling |
| Multispecies alignment (many sequences of moderate length) | Cross-attention between sequences (set-transformer style) | Need interactions *between* sequences, not just within |

**Implementation tip for sparse attention.** Use a fixed sliding window of size 512, plus a set of global tokens (e.g. every 1 kbp). This keeps memory `O(n·w + n_global²)` rather than `O(n²)`.

```python
# Example using Hugging Face Longformer config
from transformers import LongformerConfig, LongformerModel

config = LongformerConfig(
    attention_window=512,           # local window size
    attention_dilation=1,           # no dilation in this example
    num_attention_heads=12,
    hidden_size=768,
    max_position_embeddings=4096,   # window * 8 approx
)
model = LongformerModel(config)
```

## 3.3  What attention maps reveal

Attention heads in genomic models frequently learn:

- **Promoter–enhancer contacts** (heads 3, 7 of Enformer layer 6 typically light up between known regulatory pairs).
- **CTCF anchor proximity** (head specialization observed in DNABERT-2).
- **Splice-site coupling** within transcripts (SpliceBERT).

Quantitative protocol: compute the per-head attention matrix `A_h`, threshold to the top 1 %, and report enrichment for ChIA-PET loops. Random-network baselines are essential.

### 3.3a  Deep dive: what attention maps actually tell us — a cautionary guide

Attention heads can learn promoter–enhancer contacts, but over-interpretation is rampant in the literature. Here is a rigorous protocol for **validating attention as a biological contact predictor**.

**Gold-standard validation steps:**

1. **Baseline: random network.** Train the same architecture with randomly initialized weights (or freeze it after random initialization). Compute attention–contact enrichment for this baseline. If the real model does not significantly exceed random, attention is not capturing biology.
2. **Control: shuffled input.** Permute the input sequence (preserving k-mer frequencies but breaking order). Compute attention. If attention still aligns with Hi-C contacts, the model is exploiting non-sequential biases (e.g. GC content correlates with contact maps).
3. **Comparison with perturbation-based importance.** Use in-silico mutagenesis (Chapter 7) or integrated gradients to compute per-base importance for the same task. Compute overlap between top-attention positions and top-importance positions. Typically <30 % overlap — attention is not a faithful explanation.
4. **Causal test: mutate a high-attention site** in a held-out validation set and observe whether predicted contact strength changes as expected. This is expensive but the only true validation.

**Recommended figure for papers.** A side-by-side heatmap of (a) the attention-weight matrix from a specific head, (b) the Hi-C contact matrix, and (c) the perturbation-importance matrix. Report Pearson correlation after smoothing.

```python
import numpy as np

def compare_attention_to_hic(attn_matrix: np.ndarray, hic_matrix: np.ndarray,
                             symmetric: bool = True) -> float:
    """
    attn_matrix: (L, L) float, e.g. from a single head after softmax
    hic_matrix:  (L, L) float, contact frequencies, typically log1p normalized
    Returns Pearson correlation after flattening the upper triangle.
    """
    if symmetric:
        triu_indices = np.triu_indices_from(attn_matrix, k=1)
        attn_triu = attn_matrix[triu_indices]
        hic_triu = hic_matrix[triu_indices]
    else:
        attn_triu = attn_matrix.flatten()
        hic_triu = hic_matrix.flatten()
    return np.corrcoef(attn_triu, hic_triu)[0, 1]
```

> **Pitfall.** Attention weights are normalized across keys for each query. A position may have high attention to many others simply because the query is "broadly attending" — not because of a specific interaction. Compare the attention **difference** between two conditions (e.g. with and without a transcription-factor binding) to isolate specific contacts.

## 3.4  Worked example — attention as contact prediction

```python
import torch
import torch.nn.functional as F

def head_attention(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Return (n, n) attention weights for one head."""
    d = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d**0.5
    return F.softmax(scores, dim=-1)

# Symmetrize and threshold to obtain candidate long-range contacts
def candidate_contacts(A: torch.Tensor, top_frac: float = 0.01) -> torch.Tensor:
    sym = 0.5 * (A + A.T)
    k = int(top_frac * sym.numel())
    thresh = torch.topk(sym.flatten(), k).values.min()
    return (sym >= thresh).int()
```

Combine with a Hi-C matrix and compute an odds ratio of overlap.

### 3.4a  Worked example extension: attention as contact prediction with real data

Extend the minimal `head_attention` function to a full pipeline using a pre-trained Enformer or Borzoi model.

**Steps:**

1. Load a pre-trained model (e.g. from `enformer-pytorch` or `borzoi`).
2. Input a 200 kbp sequence centered on a known locus with enhancer–promoter contacts (e.g. the *Sox2* locus or the *β-globin* locus).
3. Extract attention weights from a chosen layer and head (layer 6, head 3 has been reported to show enhancer–promoter patterns).
4. Compute the candidate contact map using the `candidate_contacts` function from §3.4.
5. Overlay with ChIA-PET or Hi-C data from the same cell type (e.g. from ENCODE or 4DN).
6. Compute an enrichment odds ratio: (# attention pairs overlapping Hi-C contacts) / (total attention pairs), divided by the genomic background overlap.

```python
def contact_enrichment(attn_pairs: set, hic_pairs: set,
                       total_possible_pairs: int) -> float:
    """Compute odds ratio: (a / A) / ((N - a) / (N - A)) approximated."""
    a = len(attn_pairs.intersection(hic_pairs))
    A = len(hic_pairs)
    N = total_possible_pairs
    if a == 0 or A == N:
        return 1.0
    return (a / A) / ((N - a) / (N - A))
```

**Expected outcome.** For a good model (e.g. Enformer), the enrichment odds ratio is >5 for the top 1 % of attention pairs. For a random baseline, the odds ratio ≈ 1.

**Extension exercise (see §3.6e).** Repeat the analysis on a locus where you expect no contacts (e.g. a gene desert). Report the false-positive rate of attention-based contact prediction.

## 3.5  Common pitfalls

- **Position-encoding leakage.** Absolute positional encodings memorize chromosome coordinates; prefer relative or rotary positions.
- **Attention as "explanation".** Attention weight is not causal attribution. Use integrated gradients or attention rollout with care.
- **Distribution shift across species.** Models trained on human do not transfer naively to plant genomes (different repeat landscape, polyploidy).

### 3.5a  Common pitfalls extended — positional-encoding leakage

The pitfall above notes that absolute positional encodings can memorize chromosome coordinates. This is a serious issue for genomics models that operate on fixed-length windows from different genomic locations.

**Why it happens.** Absolute positional encodings add a unique vector per position (sine/cosine or learned). If the model always sees the same absolute position associated with the same label in training, it can memorize that association without learning sequence content. For example, a model might learn that positions 10,000,001–10,005,000 (a specific window on chr1) are always in a certain chromatin state, simply because that window contains a known CTCF site. But when the window moves to a different location with the same sequence pattern, absolute positions break.

**Solution.** Use **relative positional encodings** (e.g. Rotary Position Embedding, RoPE) or **ALiBi** (attention with linear biases). RoPE encodes relative distance via rotation matrices and is standard in many modern transformers (e.g. GPT-NeoX, LLaMA). For DNA, relative position works well because biology depends on the *distance* between elements, not on absolute coordinate.

**Implementation check.** After training, take a sequence and shift it by 1 base (or, better, take a different genomic window with the same k-mer distribution). Compute the prediction. If the prediction changes dramatically despite identical sequence content (up to boundaries), the model is relying on absolute positions — a failure.

```python
import numpy as np

def test_positional_robustness(model, seq, shift=10):
    """Predict on seq and on seq shifted by 'shift' bases (with padding)."""
    # assume model takes fixed-length input
    seq_shifted = '_' * shift + seq[:-shift]  # simplistic; real use needs a proper shift with tokenization
    pred_orig = model(seq)
    pred_shift = model(seq_shifted)
    delta = np.abs(pred_orig - pred_shift).mean()
    return delta
```

> **Pitfall.** Some relative encodings still have a "preferred" range (e.g. they work best for distances up to some limit). Ensure your training data covers the distances relevant to your task.

## 3.6  Exercises

1. **Replicate the Enformer ablation.** Remove the transformer stack from a pre-trained Enformer (keep only the CNN trunk). Re-evaluate on CAGE prediction. Quantify the loss.
2. **Sparse attention budget.** Implement BigBird-style block-sparse attention. What density preserves >95 % of dense-attention predictive performance on a CAGE benchmark?
3. **Attention ↔ Hi-C.** Pick a 1 Mbp locus. Plot the layer-6 attention matrix alongside the corresponding Hi-C contact map. Compute Pearson correlation after log-normalization.
4. **Equivariance test.** Verify that your transformer's predictions on `seq` and `reverse_complement(seq)` differ by less than 1 % in L1 norm.
5. **3.6e — Attention vs. integrated gradients.** For the same locus and task, compute (i) attention scores (averaged across heads and layers) and (ii) integrated-gradients (IG) attribution for the same output. Plot a scatter of IG importance vs. attention weight for each base pair and compute the Spearman correlation. Typically correlation <0.3; discuss why.
6. **3.6f — Memory benchmark.** Implement three attention variants on a fixed sequence length (e.g. 100 kbp): dense (full quadratic), sliding window (window = 1024), and linear attention (Performer). Measure peak GPU memory and forward-pass time, and plot memory vs. L.
7. **3.6g — Attention as a motif-discovery tool.** For a head you suspect detects a specific transcription-factor motif (e.g. CTCF), extract the subsequences that receive highest attention from a fixed reference position. Use a motif-discovery tool (MEME) to find enriched motifs and compare to known motifs from JASPAR.

## 3.7  Further reading

- Vaswani, A. *Attention is all you need.* NeurIPS (2017).
- Avsec, Ž. *Effective gene-expression prediction from sequence by integrating long-range interactions.* Nat. Methods (2021) — Enformer.
- Gu, A.; Dao, T. *Mamba.* arXiv (2023).
- Linder, J. *Predicting RNA-seq from DNA with Borzoi.* bioRxiv (2023).
- Jain, S. et al. *Attention is not Explanation in Genomic Models.* Bioinformatics (2023) — a critical empirical study showing low faithfulness.
- Vig, J. et al. *BERTology Meets Biology: Interpreting Attention in Protein Language Models.* ICLR (2021) — methods for attention interpretation adapted to biology.
- Dao, T. et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS (2022) — makes dense attention more practical by avoiding materializing QK^T.
- Nguyen, E. et al. *HyenaDNA: Long-Range Genomic Foundation Model.* ICML (2024) — includes detailed complexity comparisons.

## See also

- [Chapter 2 — DNA Language Models](chapter_02_dna_language_models.md)
- [Chapter 7 — Genomics & Gene Regulation](chapter_07_genomics.md)
- [Genomics API](../api/genomics.md)


---
<sub>Support DaScient, Inc. (a non-profit promoting accessible intelligence and community learning) via [Donations](https://cash.app/dascient/).</sub>
