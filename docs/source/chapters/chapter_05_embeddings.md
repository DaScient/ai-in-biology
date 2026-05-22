# Chapter 5 — Representation Learning for Life

> *"A good embedding is a hypothesis about which differences between biological objects matter."*

## Learning objectives

- Define and contrast self-supervised, contrastive, and generative representation-learning objectives.
- Implement masked-modeling and contrastive pre-training for biological sequences and cells.
- Diagnose embedding quality with intrinsic (alignment / uniformity) and extrinsic (probing) metrics.
- Choose a foundation model appropriate to the downstream task and dataset size.

## 5.1  Why representation learning?

Raw biological data — base-pair strings, gene-count vectors, microscopy images — are *high-dimensional and structured*. Hand-engineered features (k-mers, BLAST hits, hand-drawn ROIs) capture only what the engineer already knew to look for. Representation learning offloads feature design to the data.

The empirical claim of the last five years: **a single self-supervised model trained on millions of biological objects rivals or beats task-specific supervised models trained on thousands of labels**, especially under distribution shift.

## 5.2  Three families of objectives

| Family | Objective | Examples |
|--------|-----------|----------|
| Masked / autoregressive | Predict held-out tokens or pixels | DNABERT, ESM-2, scGPT, BulkRNABert |
| Contrastive | Pull positive pairs together, push negatives apart | scCLIP, OpenProtein–CLIP, BioCLIP |
| Generative (latent) | Maximize ELBO of `p(x)` | scVI, totalVI, geneVAE |

In practice, current state-of-the-art models often combine objectives (e.g. ESM-3 mixes masked and discrete-diffusion losses).

## 5.3  Intrinsic metrics

For an encoder `f: 𝒳 → ℝ^d`:

- **Alignment**:  `𝔼[‖f(x) − f(x⁺)‖²]` for positive pairs — lower is better.
- **Uniformity**: `log 𝔼[exp(−2 ‖f(x) − f(y)‖²)]` for random pairs — lower (more uniform) is generally better.
- **Effective rank**: `exp(H(eigenvalues))` — guards against representational collapse.

## 5.4  Worked example — contrastive cell embeddings

```python
import torch
import torch.nn.functional as F

def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Symmetric InfoNCE loss for batched positive pairs."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / tau           # (B, B)
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
```

Positive pairs for single cells can be constructed by:

- two *augmentations* of the same cell's expression vector (random gene dropout, Poisson resampling), or
- *paired* RNA + protein measurements from CITE-seq (cross-modal contrastive).

## 5.5  Probing — how good is the embedding really?

A useful, cheap protocol:

1. Freeze the encoder.
2. Fit a logistic-regression *linear probe* on the embedding for each downstream label.
3. Report macro-F1 and compare against (a) raw features, (b) a fully supervised end-to-end model.

If the linear probe approaches end-to-end performance, the representation is "good enough" and frees compute for adaptation rather than retraining.

## 5.6  Pitfalls

- **Batch effects encoded as biology.** Sequencer, lab, donor, and date can dominate a learned embedding. Always test stratified by batch.
- **Negatives that aren't.** In contrastive learning, sampling another cell of the same type as a "negative" hurts. Use hard-negative mining or supervised contrastive when labels are partly available.
- **Foundation-model lock-in.** Pre-trained checkpoints may bake in proprietary tokenizers; document and pin them.

## 5.7  Exercises

1. **Alignment / uniformity.** Train two encoders for scRNA: (a) a simple PCA, (b) a 2-layer MLP with `info_nce`. Plot both metrics across training.
2. **Probe vs. fine-tune.** On the Tabula Sapiens labels, compare linear-probe and full fine-tune macro-F1. At what dataset size do they converge?
3. **Cross-modal contrast.** Implement CITE-seq RNA↔ADT contrastive pre-training. Show that protein-side queries retrieve the correct RNA cells better than chance.
4. **Effective rank.** Estimate the effective rank of an ESM-2 650 M embedding over a held-out Pfam family. Compare to that of a randomly initialized model.

## 5.8  Further reading

- Devlin, J. *BERT.* NAACL (2019).
- Chen, T. *SimCLR.* ICML (2020).
- Lin, Z. *Evolutionary-scale prediction of atomic-level protein structure.* Science (2023) — ESM-2.
- Cui, H. *scGPT.* Nat. Methods (2024).

## See also

- [Chapter 6 — Modeling Living Systems](chapter_06_modeling.md)
- [Chapter 8 — Protein Structure & Design](chapter_08_protein.md)
- [Chapter 9 — Single-Cell Intelligence](chapter_09_single_cell.md)
