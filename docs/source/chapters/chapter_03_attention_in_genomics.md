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

## 3.2  Long-range mechanisms in production models

| Model | Receptive field | Trick |
|-------|------------------|-------|
| Basenji2 | 131 kbp | Dilated convolutions, no attention |
| Enformer | 196 kbp | Stem CNN → 11-layer transformer, relative positions |
| Borzoi | 524 kbp | Enformer + RNA-seq prediction |
| HyenaDNA | 1 Mbp | Implicit long convolutions (Hyena operator) |
| Caduceus | 131 kbp | Reverse-complement-equivariant Mamba |

The pattern across models is to push compute into a **convolutional stem** that down-samples to ~128 bp resolution, then apply attention or an attention substitute at the coarse scale.

## 3.3  What attention maps reveal

Attention heads in genomic models frequently learn:

- **Promoter–enhancer contacts** (heads 3, 7 of Enformer layer 6 typically light up between known regulatory pairs).
- **CTCF anchor proximity** (head specialization observed in DNABERT-2).
- **Splice-site coupling** within transcripts (SpliceBERT).

Quantitative protocol: compute the per-head attention matrix `A_h`, threshold to the top 1 %, and report enrichment for ChIA-PET loops. Random-network baselines are essential.

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

## 3.5  Common pitfalls

- **Position-encoding leakage.** Absolute positional encodings memorize chromosome coordinates; prefer relative or rotary positions.
- **Attention as "explanation".** Attention weight is not causal attribution. Use integrated gradients or attention rollout with care.
- **Distribution shift across species.** Models trained on human do not transfer naively to plant genomes (different repeat landscape, polyploidy).

## 3.6  Exercises

1. **Replicate the Enformer ablation.** Remove the transformer stack from a pre-trained Enformer (keep only the CNN trunk). Re-evaluate on CAGE prediction. Quantify the loss.
2. **Sparse attention budget.** Implement BigBird-style block-sparse attention. What density preserves >95 % of dense-attention predictive performance on a CAGE benchmark?
3. **Attention ↔ Hi-C.** Pick a 1 Mbp locus. Plot the layer-6 attention matrix alongside the corresponding Hi-C contact map. Compute Pearson correlation after log-normalization.
4. **Equivariance test.** Verify that your transformer's predictions on `seq` and `reverse_complement(seq)` differ by less than 1 % in L1 norm.

## 3.7  Further reading

- Vaswani, A. *Attention is all you need.* NeurIPS (2017).
- Avsec, Ž. *Effective gene-expression prediction from sequence by integrating long-range interactions.* Nat. Methods (2021) — Enformer.
- Gu, A.; Dao, T. *Mamba.* arXiv (2023).
- Linder, J. *Predicting RNA-seq from DNA with Borzoi.* bioRxiv (2023).

## See also

- [Chapter 2 — DNA Language Models](chapter_02_dna_language_models.md)
- [Chapter 7 — Genomics & Gene Regulation](chapter_07_genomics.md)
- [Genomics API](../api/genomics.md)
