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

### 10.1a  A modality decision matrix

The list above tells you *what* each method measures; the matrix below helps you choose *which combination* fits a given developmental question.

| Question | Best method(s) | Spatial resolution | Temporal resolution | Scale | Cost |
|----------|----------------|--------------------|---------------------|-------|------|
| Which cells give rise to which? (fate mapping) | Lineage tracing (CRISPR scars, barcodes) | Single-cell | Snapshot (terminal) | Thousands of cells | Moderate |
| When do cells commit to a lineage? | Time-series scRNA-seq + RNA velocity | Single-cell | Minutes (inferred) | Tens of thousands | Low–moderate |
| Where do cells move during gastrulation? | Live imaging (light-sheet) + cell tracking | Subcellular | Seconds–minutes | Hundreds of cells | High (microscopy) |
| What genes are active in each spatial domain? | Spatial transcriptomics (Visium, MERFISH) | 10–100 µm | Snapshot (fixed) | Thousands of spots | Moderate–high |
| How do signalling gradients shape tissue? | Spatial + optimal transport / PINNs | 10–100 µm | Snapshot + inference | Tissue-scale | High |
| What is the complete lineage tree of an organism? | Whole-organism lineage tracing (zebrafish, *C. elegans*) | Single-cell | Fixed timepoints | Whole organism | Very high |

**Rule of thumb:** Start with time-series scRNA-seq (lowest cost, highest throughput). Use spatial methods to validate and localize candidate trajectories. Use lineage tracing to resolve branch points with single-cell resolution.

**Pitfall:** Lineage barcodes (e.g. expressed CRISPR scars) can be lost during differentiation if the barcode is silenced. Always include a constitutive promoter control.

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

### 10.2a  Unbalanced OT when cells are born and die

Balanced (entropic) OT assumes the total mass — the number of cells — is conserved between snapshots. Real development violates this: cells proliferate, die, or migrate in and out of view. **Unbalanced optimal transport (UOT)** relaxes the marginal constraints so total mass can vary.

- **Use balanced OT** when the cell count is conserved, or two close time points of the same section where birth/death is negligible.
- **Use unbalanced OT** when cell numbers differ (e.g. day 1 vs. day 5 of differentiation) or you expect significant death/proliferation.

```python
import ot
import numpy as np

def waddington_ot_unbalanced(x: np.ndarray, y: np.ndarray,
                             reg_mass: float = 1.0, eps: float = 0.05) -> np.ndarray:
    """Unbalanced OT coupling allowing mass creation/destruction.

    x: source distribution (n0, d); y: target distribution (n1, d)
    reg_mass: penalty for mass deviation (higher = more balanced)
    """
    a = np.ones(len(x)) / len(x)
    b = np.ones(len(y)) / len(y)
    M = ot.dist(x, y, metric="sqeuclidean")
    M /= M.max()
    return ot.unbalanced.sinkhorn_unbalanced(
        a, b, M, reg=eps, reg_m=reg_mass, numItermax=1000
    )

# coupling is (n_day1, n_day5)
coupling = waddington_ot_unbalanced(day1_expr, day5_expr, reg_mass=0.5)
```

**Interpretation:** For a source cell `i`, `coupling[i, :]` gives its probabilistic *descendants* in the target snapshot; the row sum is `< 1` if the cell likely died. For a target cell `j`, `coupling[:, j]` sums to `< 1` if it arose from division or was not present before.

**Pitfall:** UOT is sensitive to `reg_mass`. Too high → forces conservation (like balanced OT); too low → allows arbitrary mass changes, losing biological plausibility. Cross-validate by comparing predicted cell numbers to experimental growth curves.

## 10.3  Graphs for spatial transcriptomics

Treat each spot / cell as a node, neighbors within `r` µm as edges, expression as features. A graph-attention network learns:

- **Domain assignment** (cortical layer, organoid sub-region).
- **Cell-cell communication** (decoded by ligand–receptor attention weights).
- **Imputation** of missing genes from neighbors.

`STAGATE`, `GraphST`, and `COMMOT` are widely used reference implementations.

### 10.3a  Beyond neighbors — graphs with biological priors

A purely Euclidean graph assumes connectivity is isotropic, but cells communicate along basal lamina, blood vessels, or synaptic gaps. Prior-guided edges encode this:

1. **Physical proximity** — Euclidean distance `< R` µm. Default, but misses long-range signalling via secreted factors.
2. **Ligand–receptor compatibility** — connect spot A to spot B if A expresses ligand L and B expresses receptor R (and vice versa). Requires a curated database (CellChat, CellPhoneDB).
3. **Tissue compartment** — epithelial cells connect only to other epithelial cells or to basement membrane, not to lumen.
4. **Manual annotation** — draw regions of interest (e.g. crypt vs. villus in intestine).

**Hybrid approach:** use a weighted graph where edge weight `= exp(-distance² / σ²) · (1 + α · ligand–receptor score)`, combining physical proximity with signalling potential.

```python
import numpy as np
import scanpy as sc
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix

def build_spatial_graph(adata, spatial_key="spatial", max_distance=100,
                        lr_matrix=None, alpha=0.5):
    """Build a weighted spatial graph.

    lr_matrix: (n_spots, n_spots) pre-computed ligand–receptor interaction score.
    alpha: weight of the ligand–receptor term relative to physical proximity.
    """
    coords = adata.obsm[spatial_key]          # (n, 2)
    dist = euclidean_distances(coords)
    adj_physical = np.exp(-dist**2 / (2 * max_distance**2))   # Gaussian kernel
    adj_physical[dist > max_distance] = 0
    np.fill_diagonal(adj_physical, 0)

    if lr_matrix is not None:
        lr_norm = (lr_matrix - lr_matrix.min()) / (lr_matrix.max() - lr_matrix.min() + 1e-8)
        adj = adj_physical + alpha * lr_norm
    else:
        adj = adj_physical

    return csr_matrix(adj)

adata.obsp["spatial_connectivity"] = build_spatial_graph(adata, lr_matrix=lr_scores)
sc.pp.neighbors(adata, use_rep="spatial_connectivity", key_added="spatial")
```

**Exercise extension (10.7e):** Use CellChat to compute ligand–receptor interaction scores from a spatial transcriptomics dataset (e.g. mouse brain). Build the hybrid graph and compare domain segmentation (e.g. cortical layers) against the Euclidean-only graph. Which better recovers known anatomical boundaries?

## 10.4  Reaction–diffusion meets ML

Classic Turing patterns:

```
∂u/∂t = D_u ∇²u + f(u, v)
∂v/∂t = D_v ∇²v + g(u, v)
```

Today, **physics-informed neural networks (PINNs)** fit `f, g, D_u, D_v` directly from imaging data, blending mechanism and learning. This is how recent work has identified candidate morphogen pairs from limb-bud time-lapses without prior assumptions.

### 10.4a  A complete PINN for a 1-D morphogen gradient

Consider a single morphogen obeying `∂u/∂t = D ∂²u/∂x² − k u + source(x)`, observed as fluorescence intensity `u(x, t)` at sparse points along a tissue. A PINN represents `u_θ(x, t)` as a neural network and enforces the PDE as a soft constraint: the loss combines data MSE, the PDE residual, and boundary/initial conditions.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MorphogenPINN(nn.Module):
    def __init__(self, layers=(2, 64, 64, 64, 1)):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f"linear{i}", nn.Linear(layers[i], layers[i + 1]))
            self.net.add_module(f"tanh{i}", nn.Tanh())
        self.net.add_module("linear_out", nn.Linear(layers[-2], layers[-1]))

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

def pde_residual(model, x, t, D, k, source_term):
    """Residual: du/dt - D * d²u/dx² + k*u - source."""
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    return u_t - D * u_xx + k * u - source_term(x)

def train_pinn(x_obs, t_obs, u_obs, source_term, x_max, t_max,
               epochs=1000, D=1.0, k=0.1):
    model = MorphogenPINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    x_coll = torch.rand(5000, 1, requires_grad=True) * x_max
    t_coll = torch.rand(5000, 1, requires_grad=True) * t_max

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_data = nn.MSELoss()(model(x_obs, t_obs), u_obs)
        residual = pde_residual(model, x_coll, t_coll, D, k, source_term)
        loss_pde = torch.mean(residual**2)
        loss = loss_data + 0.01 * loss_pde
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"epoch {epoch}: data={loss_data.item():.4f} pde={loss_pde.item():.4f}")
    return model
```

**What this enables:** promote `D` and `k` to learnable parameters and the same loop yields data-driven estimates of diffusion and degradation rates — no numerical PDE solve required.

**Pitfall:** PINNs are sensitive to collocation sampling and loss weighting. Use adaptive loss weighting (learning-rate annealing per term) or a hard-constraint formulation that bakes boundary conditions into the architecture.

## 10.5  Worked example — pseudotime with diffusion maps

```python
import scanpy as sc

sc.tl.diffmap(adata, n_comps=15)
adata.uns["iroot"] = int(np.argmax(adata.obs["expr_stemness"]))
sc.tl.dpt(adata, n_dcs=15)

sc.pl.embedding(adata, basis="umap", color=["dpt_pseudotime", "leiden"])
```

Validate by checking that known early markers (`POU5F1`, `NANOG` for ESC) peak near pseudotime 0 and lineage markers peak late.

### 10.5a  Detecting branches with Palantir

DPT orders cells from a single root but does not resolve where trajectories split. **Palantir** (or PAGA) adds branch detection, computing pseudotime from a start cell, per-cell branch probabilities (which terminal fate a cell will adopt), and terminal-state entropy (low entropy = committed).

```python
import palantir

# adata has a diffusion-map embedding (X_diffmap); pick a start cell
start_cell = adata.obs_names[adata.obs["expr_stemness"].argmax()]

pr_res = palantir.utils.run_palantir(adata, start_cell, num_waypoints=500)

adata.obs["palantir_pseudotime"] = pr_res.pseudotime
branch_probs = pr_res.branch_probs
adata.obsm["branch_probs"] = branch_probs.values
for branch in branch_probs.columns:
    adata.obs[f"prob_{branch}"] = branch_probs[branch]

sc.pl.scatter(adata, basis="umap", color="palantir_pseudotime", title="Pseudotime")
sc.pl.scatter(adata, basis="umap", color="prob_branch_1", title="Branch 1 probability")
```

**Expected outcome:** on a pancreatic endocrinogenesis dataset (Bastidas-Ponce et al., 2019), Palantir should identify Ngn3⁺ endocrine progenitors as a branch point towards alpha, beta, delta, and PP cells.

**Pitfall:** Palantir assumes a tree-like trajectory (no cycles). For cyclic processes (e.g. the cell cycle), use Cyclone or ReCAT instead.

## 10.6  Pitfalls

- **OT identifiability.** Many couplings have similar cost; report uncertainty (entropic OT yields a smoothed map for free).
- **Lineage barcode dropout.** Half of scarred cells may lose their barcode by sequencing; quantify capture.
- **Pseudotime ≠ real time.** Two cells at the same pseudotime can be at very different chronological times. To recover real time you need labelled time points (EdU incorporation, metabolic labelling) or known differentiation rates from the literature for calibration.

**Quantifying OT uncertainty.** Because many couplings have near-equal cost, bootstrap the Sinkhorn solve (resample cells with replacement, recompute the coupling) and report variability, or sweep `reg_mass` in unbalanced OT to see how mass deviation changes results.

```python
import numpy as np
import ot

def ot_uncertainty(x, y, n_bootstrap=100, reg=0.05):
    couplings = []
    for _ in range(n_bootstrap):
        idx_x = np.random.choice(len(x), len(x), replace=True)
        idx_y = np.random.choice(len(y), len(y), replace=True)
        M = ot.dist(x[idx_x], y[idx_y])
        a = np.ones(len(idx_x)) / len(idx_x)
        b = np.ones(len(idx_y)) / len(idx_y)
        couplings.append(ot.sinkhorn(a, b, M, reg=reg))
    # Coefficient of variation per cell pair
    return np.std(couplings, axis=0) / (np.mean(couplings, axis=0) + 1e-8)
```

## 10.7  Exercises

1. **OT vs. RNA velocity.** On the pancreatic endocrinogenesis data (Bastidas-Ponce et al., 2019), compare OT trajectories with `scVelo` velocity. Identify the regions where they disagree.
2. **Spatial domains.** Run `STAGATE` on the DLPFC Visium data. Compute ARI against manual layer annotation. Compare to non-spatial Leiden clustering.
3. **PINN morphogen.** Fit a 2-component PINN to a synthetic stripe pattern. Recover the diffusion constants within 10 % of truth.
4. **Lineage capture.** Simulate a scar-based lineage experiment with 10 % barcode dropout. Quantify how dropout biases inferred clone sizes.
5. **Spatial graph with biological priors (10.7e).** Use CellChat to score ligand–receptor interactions on a spatial dataset (e.g. mouse brain), build the hybrid graph from §10.3a, and compare domain segmentation against the Euclidean-only graph. Which better recovers known anatomical boundaries?
6. **Unbalanced OT on proliferation data (10.7f).** Generate a trajectory where cells double between `t0` and `t1` (e.g. 500 → 1000 cells). Run balanced OT and unbalanced OT; compare coupling row sums. Which method correctly shows source cells have `> 1` descendant on average?
7. **Spatial graph with ligand–receptor priors (10.7g).** Using SpatialDB or CellChat on a Visium mouse olfactory bulb dataset, compute pairwise ligand–receptor scores, build the hybrid graph, run Leiden clustering, and compare clusters to known anatomical layers.
8. **PINN for Turing pattern discovery (10.7h).** Generate synthetic data from a 2-component activator–inhibitor reaction–diffusion system on a 1-D domain. Train a PINN observing only the activator at sparse space–time points. Can it recover the diffusion constants and reaction parameters?
9. **Pseudotime alignment across conditions (10.7i).** Take two time-series scRNA-seq datasets (healthy vs. diseased, e.g. control vs. diabetic pancreas). Compute Palantir pseudotime separately, then use OT to align the pseudotime axes. Plot a key marker (e.g. Ngn3) against aligned pseudotime — does disease delay or accelerate differentiation?

## 10.8  Further reading

- Schiebinger, G. *Optimal transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming.* Cell (2019).
- Wagner, D. E. *Single-cell mapping of gene expression landscapes.* Science (2018).
- Dong, K., Zhang, S. *Deciphering spatial domains from spatial transcriptomics with STAGATE.* Nat Commun (2022).
- Karniadakis, G. *Physics-informed machine learning.* Nat Rev Phys (2021).
- Schiebinger, G. *Optimal transport for single-cell genomics.* Annual Review of Biomedical Data Science (2021) — review with code examples.
- Tong, A. et al. *Geometric deep learning for spatial transcriptomics.* Nat Methods (2023) — GNN architectures for spatial domains.
- Raissi, M. et al. *Physics-infused deep neural networks for predicting morphogen gradients.* PNAS (2019) — PINNs in developmental biology.
- Setty, M. et al. *Characterization of cell fate probabilities in single-cell data with Palantir.* Nat Biotechnol (2019) — detailed methods.

## See also

- [Chapter 6 — Modeling Living Systems](chapter_06_modeling.md)
- [Chapter 9 — Single-Cell Intelligence](chapter_09_single_cell.md)
- [Chapter 11 — Neuroscience](chapter_11_neuroscience.md)
