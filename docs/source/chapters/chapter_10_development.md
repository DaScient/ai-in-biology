# Chapter 10 — Development & Morphogenesis

> *"Embryogenesis is the most reliable program ever written; we are only beginning to read its source."*

## Learning objectives

- Describe the major developmental modalities (lineage tracing, spatial transcriptomics, 4-D imaging) and which questions each can answer.
- Implement an optimal-transport based mapping between two single-cell snapshots in time.
- Use a graph-neural network on spatial data to predict cell-cell communication.
- Reason about reaction–diffusion (Turing) patterns and where AI complements them.

## 10.1  Reading developmental time

| Modality | What it measures | Example tool |
|----------|------------------|--------------|
| Scar-based lineage | Inheritable barcodes | GESTALT, scGESTALT |
| CRISPR-based lineage | Editable sites | Cas9-LINEAGE, ScarTrace |
| Live imaging | Cell positions over time | Mastodon, TrackMate |
| Spatial transcriptomics | Gene × location | Visium, MERFISH, Slide-seqV2 |
| Single-cell time courses | gene × cell at multiple t | scRNA-seq series |

A modern study often combines several — e.g. CRISPR scars + scRNA at three time points + Visium of the same embryo.

## 10.2  Optimal transport for trajectory inference

Given two snapshots at times `t₀` and `t₁` with empirical distributions `μ`, `ν`, OT finds a coupling `π(x, y)` minimizing transport cost `c(x, y)` such that marginals match. This couples *unmatched* cells across time without needing lineage barcodes.

```python
import ot
import numpy as np

# x: (n0, d) expression at t0; y: (n1, d) expression at t1
def waddington_ot(x: np.ndarray, y: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """Entropic OT (Sinkhorn) coupling between two snapshots."""
    a = np.ones(len(x)) / len(x)
    b = np.ones(len(y)) / len(y)
    M = ot.dist(x, y, metric="sqeuclidean")
    M /= M.max()
    return ot.sinkhorn(a, b, M, reg=eps)
```

This is the core of Waddington-OT and many downstream developmental-trajectory packages.

## 10.3  Graphs for spatial transcriptomics

Treat each spot / cell as a node, neighbors within `r` µm as edges, expression as features. A graph-attention network learns:

- **Domain assignment** (cortical layer, organoid sub-region).
- **Cell-cell communication** (decoded by ligand–receptor attention weights).
- **Imputation** of missing genes from neighbors.

`STAGATE`, `GraphST`, and `COMMOT` are widely used reference implementations.

## 10.4  Reaction–diffusion meets ML

Classic Turing patterns:

```
∂u/∂t = D_u ∇²u + f(u, v)
∂v/∂t = D_v ∇²v + g(u, v)
```

Today, **physics-informed neural networks (PINNs)** fit `f, g, D_u, D_v` directly from imaging data, blending mechanism and learning. This is how recent work has identified candidate morphogen pairs from limb-bud time-lapses without prior assumptions.

## 10.5  Worked example — pseudotime with diffusion maps

```python
import scanpy as sc

sc.tl.diffmap(adata, n_comps=15)
adata.uns["iroot"] = int(np.argmax(adata.obs["expr_stemness"]))
sc.tl.dpt(adata, n_dcs=15)

sc.pl.embedding(adata, basis="umap", color=["dpt_pseudotime", "leiden"])
```

Validate by checking that known early markers (`POU5F1`, `NANOG` for ESC) peak near pseudotime 0 and lineage markers peak late.

## 10.6  Pitfalls

- **OT identifiability.** Many couplings have similar cost; report uncertainty (entropic OT yields a smoothed map for free).
- **Lineage barcode dropout.** Half of scarred cells may lose their barcode by sequencing; quantify capture.
- **Pseudotime ≠ real time.** Two cells at the same pseudotime can be at very different chronological times.

## 10.7  Exercises

1. **OT vs. RNA velocity.** On the pancreatic endocrinogenesis data (Bastidas-Ponce et al., 2019), compare OT trajectories with `scVelo` velocity. Identify the regions where they disagree.
2. **Spatial domains.** Run `STAGATE` on the DLPFC Visium data. Compute ARI against manual layer annotation. Compare to non-spatial Leiden clustering.
3. **PINN morphogen.** Fit a 2-component PINN to a synthetic stripe pattern. Recover the diffusion constants within 10 % of truth.
4. **Lineage capture.** Simulate a scar-based lineage experiment with 10 % barcode dropout. Quantify how dropout biases inferred clone sizes.

## 10.8  Further reading

- Schiebinger, G. *Optimal transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming.* Cell (2019).
- Wagner, D. E. *Single-cell mapping of gene expression landscapes.* Science (2018).
- Dong, K., Zhang, S. *Deciphering spatial domains from spatial transcriptomics with STAGATE.* Nat Commun (2022).
- Karniadakis, G. *Physics-informed machine learning.* Nat Rev Phys (2021).

## See also

- [Chapter 6 — Modeling Living Systems](chapter_06_modeling.md)
- [Chapter 9 — Single-Cell Intelligence](chapter_09_single_cell.md)
- [Chapter 11 — Neuroscience](chapter_11_neuroscience.md)
