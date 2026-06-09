# Chapter 13 — Evolutionary Dynamics

> *"Evolution is the original gradient descent — noisy, parallel, and unforgiving."*

## Learning objectives

- Relate population-genetic theory (Wright–Fisher, coalescent, selection) to machine-learning models trained on genomic variation.
- Use deep models to infer demographic history, detect selection, and call recombination breakpoints from sequence alone.
- Apply phylogenetic neural networks to large molecular trees.
- Connect evolutionary search to in-silico protein and genome design (directed evolution, MLDE).

## 13.1  Population genetics in 200 words

Allele frequency `p` of a neutral variant under Wright–Fisher drift behaves as

```
Δp ∼ 𝒩(0, p(1 − p) / (2N_e))
```

Selection adds a deterministic term `s · p(1 − p)`. The *coalescent* runs this backwards: lineages merge at rate `1/(2N_e)` per generation. From a sample of `n` chromosomes, the coalescent tree summarizes everything we can ever learn about demography from neutral variation.

Modern ML inference (`dadi`, `momi2`, `dinf`) trains neural networks on simulated coalescent samples and infers demographic parameters from a single empirical SFS in milliseconds.

### 13.1a  Population genetics for ML practitioners — key equations and their ML analogues

To make the connection concrete for readers with a machine-learning background, the table below pairs each core population-genetic concept with its closest ML analogue.

| Population genetics concept | Mathematical form | ML analogue | Why it matters for AI |
|----------------------------|------------------|-------------|----------------------|
| **Wright–Fisher drift** | `p_{t+1} ∼ (1 / 2N_e) · Binomial(2N_e, p_t)` | Stochastic gradient descent with mini-batch noise | Drift adds noise to allele frequencies; models must account for sampling variance |
| **Selection** | `p_{t+1} = p_t(1 + s) / (1 + s·p_t)` (haploid) | Gradient step with learning rate `s` | Selection is a deterministic force; ML models can learn to separate it from drift |
| **Recombination** | Breakpoints occur with rate `r` per bp | Cross-over in genetic algorithms | Recombination creates new haplotypes; long-range dependencies in genomes arise from recombination |
| **Coalescent** | `T_k ∼ Exp(k(k − 1) / 4N_e)` | Bayesian prior over tree topologies and branch lengths | Most demographic-inference methods rely on coalescent simulations; deep learning can approximate the likelihood |
| **Site frequency spectrum (SFS)** | Histogram of derived allele counts in a sample | Summary statistic (like a histogram of predictions) | SFS is the workhorse for inference; neural networks can map SFS to demography without specifying a model |

**Key insight for AI.** The coalescent generates *exact* training data (simulated genomes) under known parameters. This enables supervised learning for inference problems where the likelihood is intractable — a classic *simulation-based inference* setup.

```python
# Simulate coalescent trees and sequences with msprime
import msprime
import numpy as np

def simulate_coalescent(Ne=10000, n_samples=100, seq_len=10000,
                        mutation_rate=1e-8, recombination_rate=1e-8):
    ts = msprime.sim_ancestry(
        samples=n_samples,
        population_size=Ne,
        recombination_rate=recombination_rate,
        sequence_length=seq_len,
        random_seed=42,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=42)
    # Extract genotype matrix (n_variants x n_samples)
    genotypes = ts.genotype_matrix()  # 0/1 (haploid sample nodes)
    # Folded site frequency spectrum (for unknown ancestral state)
    sfs = np.bincount(genotypes.sum(axis=1))[1:-1]  # exclude monomorphic
    return genotypes, sfs
```

**Pitfall.** Simulators assume an infinite-sites model, but real genomes have recurrent mutations and multi-nucleotide variants. Use more realistic simulators (e.g., `SLiM`, `stdpopsim`) when needed.

## 13.2  Detecting selection

| Signal | Statistic | ML augmentation |
|--------|-----------|------------------|
| Reduced diversity | π, θ_W | CNN on haplotype matrices (`diploS/HIC`) |
| Long haplotypes | iHS, XP-EHH | RNN over windowed scans |
| Allele-frequency time series | s vs. drift | LSTM, Transformer |
| Co-evolution | DCA, ESM contacts | Transformer (`evoformer`) |

A useful guard: *always* match neutral background simulations to the empirical recombination map and demographic history.

### 13.2a  Deep learning for sweeps — diploSHIC

Classical methods (Tajima's D, iHS, nSL) rely on summary statistics that may not capture every signature of selection. **diploSHIC** (diploid Sweep Heterozygosity and Identity by descent) is a convolutional neural network that detects sweeps directly from haplotype data, learning the full spatial pattern of heterozygosity and haplotype structure.

**diploSHIC workflow.**

1. Simulate neutral and selective-sweep data under your species' demography (using `msprime` or `SLiM`).
2. For each simulated region (e.g., 100 kb), encode the haplotype matrix (`n_haplotypes × n_sites`, with 0/1 for ancestral/derived) as an image (or feature tensor).
3. Train a CNN to classify *sweep* vs. *neutral* (binary), or to predict sweep parameters (hard/soft, age).
4. Apply to real data: slide a window across the genome, classify each window, and compute a sweep probability.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_diploshic_cnn(seq_len, n_haps=100):
    """CNN for sweep detection from a haplotype matrix."""
    input_shape = (n_haps, seq_len)  # haplotypes as rows, sites as columns
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((n_haps, seq_len, 1)),  # add channel dim
        layers.Conv2D(16, kernel_size=(5, 10), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(5, 10), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),  # probability of sweep
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# X_train: (n_samples, n_haps, seq_len) of 0/1 haplotypes
# y_train: 0 (neutral) or 1 (sweep)
model = build_diploshic_cnn(seq_len=20000, n_haps=100)
model.fit(X_train, y_train, epochs=20, validation_split=0.2)
```

**Post-processing.** For each genomic window, apply the model. Smooth sweep probabilities with a moving average (e.g., a 10-window average) and flag candidate sweep regions where probability > 0.9.

**Pitfall.** The model is only as good as the simulations. If the demographic model is wrong (e.g., an incorrect bottleneck size), the classifier will be mis-calibrated. Always simulate under a null demographic model fitted to your data (e.g., using `dadi` or `momi2`).

## 13.3  Worked example — phylogenetic GNN

```python
import torch
from torch_geometric.nn import GCNConv

class PhyloGNN(torch.nn.Module):
    """GCN over a phylogenetic tree's node graph (binary, rooted)."""

    def __init__(self, in_dim, hidden=64, out_dim=1):
        super().__init__()
        self.g1 = GCNConv(in_dim, hidden)
        self.g2 = GCNConv(hidden, hidden)
        self.out = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        h = self.g1(x, edge_index).relu()
        h = self.g2(h, edge_index).relu()
        return self.out(h)
```

Used for ancestral-state reconstruction, trait evolution rate inference, and even reroot-the-tree tasks.

### 13.3a  Handling missing data in a phylogenetic GNN

Real trees often have **missing data** — some species lack a trait value. Suppose you have a phylogenetic tree with known leaf traits (e.g., body mass) for some species and want to predict traits for internal nodes and for missing leaves. The solution is a GNN that masks missing values and only backpropagates from observed nodes.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class PhyloGNNMissing(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, mask):
        # x: (n_nodes, in_dim), mask: boolean for observed nodes
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        pred = self.out(h).squeeze(-1)  # (n_nodes,)
        return pred, mask  # masked nodes are ignored in the loss

def train_phylo_gnn(data, observed_mask, epochs=200):
    model = PhyloGNNMissing(in_dim=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss(reduction='none')
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred, mask = model(data.x, data.edge_index, observed_mask)
        loss = loss_fn(pred[mask], data.y[mask]).mean()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: loss {loss.item():.4f}")
    return model

# Build the tree graph: adjacency from parent->child edges (undirected).
# data.x: features (one-hot clade membership, or placeholder ones).
# data.y: observed trait values (NaN for missing).
# observed_mask = ~torch.isnan(data.y); fill missing y with arbitrary values.
```

**Interpretation.** After training, `pred[observed_mask]` approximates the observed traits and `pred[~observed_mask]` gives predicted traits for missing leaves or internal nodes — a phylogenetically imputed value, with uncertainty if you enable dropout at inference.

**Pitfall.** The GNN treats the tree as undirected and may not respect the direction of evolution (root → leaves). Add direction-aware message passing (e.g., a DAG neural network) if the root is known.

## 13.4  Directed evolution with ML

Machine-learning-guided directed evolution (MLDE) iterates:

1. *Library design* — diverse mutations sampled from a PLM prior.
2. *Wet-lab screen* — fluorescence / activity readout.
3. *Surrogate fit* — GP or PLM-head on the labeled subset.
4. *Acquisition* — propose next round (UCB, Thompson, Bayesian optimization).

Three rounds typically beat any single round of random mutagenesis at ~10× fewer wet-lab variants.

### 13.4a  Active learning with Gaussian processes and Thompson sampling

A Gaussian process (GP) surrogate provides uncertainty estimates (posterior variance) that are essential for acquisition functions such as UCB or expected improvement. A detailed active-learning loop for a protein-engineering campaign looks like:

1. **Initial library** — 96 variants (e.g., single mutants around a hotspot).
2. **Wet-lab assay** — measure fitness (e.g., fluorescence, catalytic rate).
3. **Encode sequences** using ESM-2 embeddings (mean-pooled over residues).
4. **Fit a GP** with a Matérn kernel on the embedding distance.
5. **Select the next batch** of 96 variants via Thompson sampling (sample from the GP posterior, pick the highest predicted fitness).
6. **Validate** top candidates; update the GP; iterate.

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def gp_active_loop(sequences, embeddings, fitness, n_rounds=3, batch_size=96):
    """
    sequences:  list of strings (already tested)
    embeddings: numpy array (n_tested, d)
    fitness:    numpy array (n_tested,)
    """
    for round in range(n_rounds):
        # Fit GP
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp.fit(embeddings, fitness)

        # Candidate library (e.g., all single mutants from a parent)
        candidates, cand_emb = generate_candidate_library(parent_seq)  # (n_cand, d)
        # Thompson sampling: draw one sample from the posterior at candidates
        f_sample = gp.sample_y(cand_emb, n_samples=1, random_state=round).flatten()
        best_idx = np.argsort(f_sample)[-batch_size:]
        selected = [candidates[i] for i in best_idx]

        # Wet-lab step (mock: oracle returns noisy fitness)
        new_fitness = oracle(selected)

        # Update training set
        sequences.extend(selected)
        embeddings = np.vstack([embeddings, cand_emb[best_idx]])
        fitness = np.append(fitness, new_fitness)

        print(f"Round {round + 1}: best fitness {fitness.max():.3f}")
    return sequences, fitness
```

**Expected outcome.** Within 3 rounds (≈300 variants) you should reach fitness levels that would otherwise require screening thousands of random variants.

**Pitfall.** A GP scales as `O(n³)` in the number of tested variants. For >10,000 variants, use a sparse GP or switch to a deep ensemble (more scalable, and also provides uncertainty).

## 13.5  Pitfalls

- **Demography ↔ selection confounding.** A bottleneck mimics positive selection. Always condition selection scans on a fitted demographic model.
- **Phylogeny error.** Single-tree analyses ignore inference uncertainty; use posterior summaries.
- **PLM bias.** Protein language models prefer the natural training distribution; do not interpret PLM scores as fitness without calibration.

### 13.5a  Diagnostic — empirical p-values for candidate sweeps

To separate selection from demography quantitatively, compute an **empirical p-value** for a candidate sweep against a null demographic model:

1. Fit a demographic model to neutral variants (e.g., using `dadi` or `momi2`).
2. Simulate 10,000 neutral datasets under the fitted model (coalescent simulations with the same sample size and recombination map).
3. For each simulation, compute the selection statistic (e.g., Tajima's D, iHS, or your CNN sweep probability).
4. The empirical p-value is the fraction of simulations whose statistic exceeds the observed value.
5. A region is *significant* if `p < 0.05` after multiple-testing correction (e.g., Benjamini–Hochberg across all windows).

```python
def empirical_p_value(observed_stat, neutral_sims):
    """Compute a p-value from neutral simulations."""
    return np.mean(neutral_sims >= observed_stat)

# Example for a single window
observed_iHS = 3.2
neutral_iHS_sims = np.array([...])  # from 10,000 simulations
p = empirical_p_value(observed_iHS, neutral_iHS_sims)
```

**Pitfall.** Simulations assume no selection *and* that the fitted demographic model is correct. Misspecification (e.g., a missing bottleneck) inflates false positives. Always run a sensitivity analysis: vary demographic parameters within their confidence intervals and recompute p-values.

## 13.6  Exercises

1. **Demographic inference.** Use `dadi` (analytical) and `dinf` (deep) on the same human Out-of-Africa SFS. Compare runtimes and parameter estimates.
2. **Selection scan.** Train `diploS/HIC` on simulations matched to *D. melanogaster*. Apply to DGRP whole-genome data. List your top 10 sweep regions.
3. **Phylogenetic GNN.** On the Open Tree of Life mammal subtree, predict body mass at internal nodes. Compare with Brownian-motion ancestral reconstruction.
4. **MLDE in silico.** Use the ProteinGym BLAT_ECOLX landscape. Run 3 rounds of UCB MLDE with an ESM-2 surrogate. Plot best-so-far vs. round.
5. **Train a CNN sweep detector.** Use msprime to simulate 10,000 neutral regions and 10,000 selective-sweep regions (hard sweep, `s = 0.01`, age `= 0.1 × N_e` generations). Train a simple 1D CNN on haplotype matrices (not the full 2D diploSHIC). Evaluate AUROC on a held-out test set. How well can you distinguish sweeps from neutrals when the recombination rate is high vs. low?
6. **GP-based MLDE on a real landscape.** Use the ProteinGym GB1 (protein G) fitness landscape. Simulate an active-learning campaign: start with 96 random single mutants, then run 3 rounds of 96 variants each using Thompson sampling with ESM-2 embeddings. Compare to random selection. How many rounds does it take to reach 90% of maximum fitness?
7. **Phylogenetic GNN for ancestral sequence reconstruction.** Take a known protein family (e.g., globins) with a well-resolved tree. Mask the sequences at the root (ancestral). Train a PhyloGNN (node features = one-hot of sequences) to predict the root sequence from the leaves. Compare to standard maximum-likelihood ancestral reconstruction (e.g., PAML). Which method is more accurate on held-out leaves?
8. **Demographic inference with neural networks.** Use `stdpopsim` to simulate datasets under a two-population split-with-migration model. Train a small MLP to predict split time and migration rate from the joint (2D) SFS. How precise are the estimates compared to `dadi` (analytical likelihood)? Test under model misspecification (e.g., the true model has a bottleneck but the MLP was trained on a simple split).

## 13.7  Further reading

- Coop, G. *Population and Quantitative Genetics* (open textbook).
- Kelleher, J. *msprime.* PLoS Comp Biol (2016).
- Wittmann, B. *Advances in machine learning for directed evolution.* Curr Opin Struct Biol (2021).
- Korfmann, K. *et al.* (2023). *Deep learning in population genetics.* Genome Biology and Evolution — review with code examples.
- Flagel, L. *et al.* (2019). *The unreasonable effectiveness of convolutional neural networks in population genetic inference.* Molecular Biology and Evolution — early CNN work.
- Yang, K. K. *et al.* (2021). *Machine-learning-guided directed evolution for protein engineering.* Nature Methods — comprehensive MLDE review.
- Schrider, D. R. (2023). *A deep learning approach to estimating the strength of natural selection from genomic data.* Genetics — diploSHIC and extensions.

## See also

- [Chapter 7 — Genomics & Gene Regulation](chapter_07_genomics.md)
- [Chapter 14 — Ecology & Conservation](chapter_14_ecology.md)
- [Chapter 17 — Biotechnology & Bioengineering](chapter_17_biotech.md)


---
<sub>Support DaScient, Inc. (a non-profit promoting accessible intelligence and community learning) via [Donations](https://cash.app/dascient/).</sub>
