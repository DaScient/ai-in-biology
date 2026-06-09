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

### 5.1a  Why representation learning? — an extended motivation

The claim above is abstract. Three case studies from recent literature make it concrete:

1. **Protein language models (ESM-2, 650 M).** Trained on 65 million protein sequences via masked language modeling. For predicting thermostability (ΔΔG), a linear probe on *frozen* ESM-2 embeddings achieves Spearman ρ ≈ 0.65, while a supervised CNN trained from scratch on 50k labeled variants achieves ρ ≈ 0.55. The representation already captures stability-relevant features.

2. **DNA language models (Nucleotide Transformer, 2.5 B).** Pre-trained on the human reference genome plus 3,000 other genomes. For promoter classification, a frozen embedding + logistic regression achieves AUROC 0.94 — only 0.02 below fine-tuning the whole model. The pre-training generalized across species.

3. **Single-cell foundation models (scGPT, 100 M).** Pre-trained on 10 million cells from multiple tissues. For cell-type annotation in a new pancreas dataset, zero-shot (no fine-tuning) reaches 78% accuracy; a supervised model trained on 5,000 labeled cells from that dataset alone reaches 72% — the foundation model already knows more about cell types than task-specific training.

**Key insight.** Self-supervised pre-training learns a **generalized biological feature extractor** that transfers to many downstream tasks. The cost of pre-training (millions of GPU hours) is amortized across thousands of users. For most labs, fine-tuning or probing a foundation model is the correct starting point.

**Pitfall.** Foundation models may fail on tasks requiring **compositional generalization** (e.g. predicting the effect of combinations of mutations that never co-occurred during pre-training). Always test on held-out combinations.

## 5.2  Three families of objectives

| Family | Objective | Examples |
|--------|-----------|----------|
| Masked / autoregressive | Predict held-out tokens or pixels | DNABERT, ESM-2, scGPT, BulkRNABert |
| Contrastive | Pull positive pairs together, push negatives apart | scCLIP, OpenProtein–CLIP, BioCLIP |
| Generative (latent) | Maximize ELBO of `p(x)` | scVI, totalVI, geneVAE |

In practice, current state-of-the-art models often combine objectives (e.g. ESM-3 mixes masked and discrete-diffusion losses).

### 5.2a  Contrastive learning: hard-negative mining

InfoNCE pulls positives together and pushes negatives apart, but not all negatives are equally informative. A **hard negative** is a sample that is similar to the positive anchor in representation space yet belongs to a different class (or is not a true positive pair). In single-cell data, two cells of different but related types (e.g. CD4⁺ T-cell vs. CD8⁺ T-cell) are hard negatives. Training only with *easy* negatives (T-cell vs. neuron) yields a representation that fails to distinguish subtle differences — a serious problem for biology, where classes are often hierarchical and imbalanced.

**Implementation strategies:**

1. **Semi-supervised hard-negative mining.** Use a small amount of labeled data to identify class boundaries; treat same-class pairs as positives and different-class pairs as negatives.
2. **Online hard-negative mining** (SimCLR with large batch). Within a batch, for each anchor treat the most similar negative (highest cosine similarity) as the hardest and up-weight it in the loss.
3. **Hard-negative sampling from a queue** (MoCo-style). Maintain a queue of past embeddings and sample negatives that are close to the anchor but not positives.

```python
import torch
import torch.nn.functional as F

def info_nce_hard(h1, h2, tau=0.1, hard_frac=0.5):
    """InfoNCE with hard-negative emphasis.
    For each anchor, the top `hard_frac` fraction of negatives are up-weighted.
    """
    h1 = F.normalize(h1, dim=-1)
    h2 = F.normalize(h2, dim=-1)
    logits = h1 @ h2.T / tau   # (B, B)

    # Compute weights: for each row, higher weight for negatives with high similarity
    with torch.no_grad():
        eye = torch.eye(len(h1), device=h1.device).bool()
        neg_logits = logits.masked_fill(eye, -float('inf'))  # mask positive pairs
        # Rank negatives by logit value (higher = harder)
        neg_ranks = neg_logits.argsort(dim=-1, descending=True)
        neg_weights = torch.zeros_like(neg_logits)
        k = max(1, int((len(h1) - 1) * hard_frac))
        for i in range(len(h1)):
            hard_idx = neg_ranks[i, :k]
            neg_weights[i, hard_idx] = 1.0 / k  # uniform over the hard negatives
        neg_weights = neg_weights / neg_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Weighted cross entropy over the hard negatives
    loss = 0.0
    for i in range(len(h1)):
        pos = logits[i, i]
        neg = (neg_weights[i] * torch.exp(logits[i])).sum()
        loss += -pos + torch.log(neg + torch.exp(pos))
    return loss / len(h1)
```

## 5.3  Intrinsic metrics

For an encoder `f: 𝒳 → ℝ^d`:

- **Alignment**:  `𝔼[‖f(x) − f(x⁺)‖²]` for positive pairs — lower is better.
- **Uniformity**: `log 𝔼[exp(−2 ‖f(x) − f(y)‖²)]` for random pairs — lower (more uniform) is generally better.
- **Effective rank**: `exp(H(eigenvalues))` — guards against representational collapse.

### 5.3a  Alignment & uniformity: practical interpretation

The definitions are easy to misread without reference values. Some intuition and diagnostic thresholds:

- **Alignment** (expected squared distance between positive pairs). Lower is better.
  - Perfect alignment (identical embeddings): 0.0
  - Typical well-trained contrastive model: 0.1–0.5 (depending on augmentation strength)
  - Random untrained encoder: ≈ 2 × variance of the data

- **Uniformity** (log of average exponentiated negative squared distance). More negative is better (more uniform); range roughly [-10, 0].
  - Perfect uniform distribution on the hypersphere: lower bound depends on dimension. For d = 128, theoretical minimum ≈ −log(128) ≈ −4.85.
  - Good contrastive model: −3 to −5.
  - Collapsed model (all embeddings identical): ≈ 0.0 (distances are zero, so exp(−0) = 1 and log(1) = 0).

**Detecting collapse.** If alignment < 0.01 *and* uniformity > −1, the encoder may be outputting near-constant vectors (collapse). This often follows from too-strong augmentation or too-high a learning rate.

```python
import torch
import torch.nn.functional as F

def compute_alignment_uniformity(z, positive_pairs):
    """
    z: (N, d) embeddings
    positive_pairs: list of (i, j) indices where i and j form a positive pair
    """
    z_norm = F.normalize(z, p=2, dim=-1)

    # Alignment
    align = 0.0
    for i, j in positive_pairs:
        align += torch.norm(z_norm[i] - z_norm[j], p=2).pow(2)
    align = align / len(positive_pairs)

    # Uniformity: average over all pairs (or a subsample)
    # For N > 1e4, subsample rows/pairs first — this matrix is O(N^2) in time and memory.
    all_pairs = torch.cdist(z_norm, z_norm, p=2).pow(2)  # squared distances
    mask = ~torch.eye(len(z_norm), dtype=bool, device=z.device)  # exclude diagonal
    uniform = torch.log((torch.exp(-2 * all_pairs[mask])).mean())
    return align.item(), uniform.item()
```

**Pitfall.** Computing uniformity on large datasets (> 10⁴ samples) is O(N²). Use subsampling (e.g. 2048 random pairs) for an approximation.

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

### 5.4a  Worked example extension: cross-modal contrastive embeddings with CITE-seq

The `info_nce` loss above is only the core; a full pipeline needs encoders, batching, and an optimizer. CITE-seq provides paired RNA-seq and protein antibody-derived tags (ADT) for the same cell, making it ideal for **cross-modal** contrastive learning.

**Goal.** Learn a joint embedding space where RNA expression and protein abundance of the *same* cell are aligned (positive pairs), while RNA of cell A and protein of cell B are negative pairs.

**Data shape.** `X_rna` is `(n_cells, n_genes)` and `X_adt` is `(n_cells, n_proteins)`; both are log-normalized.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class CITE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def train_cross_modal(rna_data, adt_data, batch_size=256, epochs=50, lr=1e-3, tau=0.1):
    rna_enc = CITE_Encoder(rna_data.shape[1])
    adt_enc = CITE_Encoder(adt_data.shape[1])
    optimizer = torch.optim.Adam(
        list(rna_enc.parameters()) + list(adt_enc.parameters()), lr=lr
    )

    dataset = TensorDataset(
        torch.tensor(rna_data, dtype=torch.float32),
        torch.tensor(adt_data, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for rna_batch, adt_batch in loader:
            z_rna = rna_enc(rna_batch)
            z_adt = adt_enc(adt_batch)
            loss = info_nce(z_rna, z_adt, tau)  # reuse the symmetric InfoNCE above
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: loss {total_loss / len(loader):.4f}")
    return rna_enc, adt_enc

# Usage:
# rna_enc, adt_enc = train_cross_modal(X_rna, X_adt)
# combined = torch.cat([rna_enc(X_rna), adt_enc(X_adt)], dim=1)  # or average
```

**Evaluation.** For a held-out set of cells, compute retrieval accuracy: given an RNA query, retrieve the most similar ADT embedding (cosine similarity) — the correct cell should rank first more often than random. Baseline: PCA on RNA + nearest neighbor in RNA space.

## 5.5  Probing — how good is the embedding really?

A useful, cheap protocol:

1. Freeze the encoder.
2. Fit a logistic-regression *linear probe* on the embedding for each downstream label.
3. Report macro-F1 and compare against (a) raw features, (b) a fully supervised end-to-end model.

If the linear probe approaches end-to-end performance, the representation is "good enough" and frees compute for adaptation rather than retraining.

### 5.5a  Probing: when and how to go beyond linear

A linear probe is a cheap diagnostic, but sometimes it is too weak. Use this decision table to choose a probe:

| Downstream task | Recommended probe | Rationale |
|-----------------|-------------------|-----------|
| Simple classification (cell type, disease status) | Logistic regression | Often sufficient; interpretable coefficients |
| Regression with mild nonlinearity (gene-expression prediction) | Linear + polynomial features (degree 2) | Adds interaction terms without overfitting |
| Multi-task (predicting multiple clinical variables) | Multi-head linear probe | Shared representation, task-specific heads |
| Complex relationship (protein fitness from sequence embedding) | Small MLP (1–2 hidden layers, dropout) | Nonlinear but still sample-efficient |
| Very small labeled set (< 100 samples) | Linear probe with strong regularization | Avoid overfitting; nonlinear is too flexible |

**If a linear probe performs poorly but an end-to-end fine-tuned model does well** → the representation is missing task-relevant features. Consider:

- Pre-training with a different objective (e.g. contrastive vs. masked).
- Using a larger foundation model.
- Domain-specific pre-training (e.g. on related species or tissues).

**If a linear probe matches fine-tuning** → the representation is excellent. Use the linear probe for production (faster, smaller).

**Pitfall.** Linear probes can be misleading if the task is inherently nonlinear but only a small fraction of samples lie near the decision boundary. Always visualize the embedding space (UMAP colored by label) before committing to linear.

## 5.6  Pitfalls

- **Batch effects encoded as biology.** Sequencer, lab, donor, and date can dominate a learned embedding. Always test stratified by batch.
- **Negatives that aren't.** In contrastive learning, sampling another cell of the same type as a "negative" hurts. Use hard-negative mining or supervised contrastive when labels are partly available.
- **Foundation-model lock-in.** Pre-trained checkpoints may bake in proprietary tokenizers; document and pin them.

### 5.6 (extended)  Additional traps in representation learning

- **5.6e — Batch effects as shortcuts (quantitative check).** Beyond stratifying by batch, train a model to predict batch ID *from the learned embedding*. If AUROC > 0.8, the representation is heavily confounded. Remediate with adversarial batch correction or domain-adversarial training.
- **5.6f — Representational collapse in VAE-based models** (e.g. scVI). Monitor the KL-divergence term. If it drops to near zero, the latent space is not being used — the decoder ignores the latents and merely reconstructs from the mean. Solution: KL annealing or β-VAE with β > 1.
- **5.6g — Tokenizer dependency in sequence models.** Different tokenizations (6-mers vs. BPE vs. characters) produce embeddings that are not directly comparable. When reporting results, state the tokenizer and vocabulary size. For transfer learning between models, realign tokens via projection layers.

## 5.7  Exercises

1. **Alignment / uniformity.** Train two encoders for scRNA: (a) a simple PCA, (b) a 2-layer MLP with `info_nce`. Plot both metrics across training.
2. **Probe vs. fine-tune.** On the Tabula Sapiens labels, compare linear-probe and full fine-tune macro-F1. At what dataset size do they converge?
3. **Cross-modal contrast.** Implement CITE-seq RNA↔ADT contrastive pre-training. Show that protein-side queries retrieve the correct RNA cells better than chance.
4. **Effective rank.** Estimate the effective rank of an ESM-2 650 M embedding over a held-out Pfam family. Compare to that of a randomly initialized model.

**Extended exercises:**

5. **(5.7d) Hard-negative mining on hierarchical labels.** On a single-cell dataset with known hierarchical labels (e.g. PBMC with cell types and subtypes), compare standard InfoNCE vs. hard-negative mining. Which yields better separation of subtypes in UMAP? Quantify with adjusted mutual information (AMI) against ground truth.
6. **(5.7e) Three-modality contrast.** Extend the CITE-seq example with a third modality — spatial location `(x, y)` — using a small MLP encoder. Implement joint contrastive learning across all three modalities. Report how much retrieval accuracy improves over two modalities.
7. **(5.7f) Probing with limited labels.** Take a pre-trained protein language model (ESM-2) and a protein-function classification task (e.g. EC-number prediction). Vary the number of labeled examples from 10 to 10,000 and plot the performance of: a linear probe, a 2-layer MLP probe (hidden 256), and full fine-tuning (all parameters). Report which method wins at each sample size.
8. **(5.7g) Alignment/uniformity ablation.** Train a contrastive model with three augmentation strategies for single-cell data: (a) weak (Gaussian noise only), (b) moderate (random gene dropout + scaling), (c) strong (permute 10% of genes). Compute alignment and uniformity after each epoch. Which augmentation yields the best downstream classification, and why?
9. **(5.7h) Cross-species transfer.** Pre-train a DNA language model on *E. coli* genomes. Probe for transcription-start-site (TSS) prediction on a held-out bacterial species (e.g. *B. subtilis*). Compare to a model pre-trained on *B. subtilis* directly (same amount of compute). Report the transfer gap.

## 5.8  Further reading

- Devlin, J. *BERT.* NAACL (2019).
- Chen, T. *SimCLR.* ICML (2020).
- Lin, Z. *Evolutionary-scale prediction of atomic-level protein structure.* Science (2023) — ESM-2.
- Cui, H. *scGPT.* Nat. Methods (2024).
- Wang, T. & Isola, P. *Understanding contrastive representation learning through alignment and uniformity on the hypersphere.* ICML (2020) — theoretical foundation for these metrics.
- Kolesnikov, A. et al. *Big Transfer (BiT): General visual representation learning.* ECCV (2020) — practical advice on linear probing vs. fine-tuning, applicable to biology.
- Bao, H. et al. *BEiT: BERT pre-training of image transformers.* ICLR (2022) — masked image modeling, analogous to masked DNA modeling.
- Lopez, R. et al. *A deep generative model for single-cell multi-omics data with missing modalities.* Nat. Biotechnol. (2023) — multi-modal contrastive learning.

## See also

- [Chapter 6 — Modeling Living Systems](chapter_06_modeling.md)
- [Chapter 8 — Protein Structure & Design](chapter_08_protein.md)
- [Chapter 9 — Single-Cell Intelligence](chapter_09_single_cell.md)


---
<sub>Support DaScient, Inc. (a non-profit promoting accessible intelligence and community learning) via [Donations](https://cash.app/dascient/).</sub>
