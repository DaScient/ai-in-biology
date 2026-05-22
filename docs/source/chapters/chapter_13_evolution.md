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

## 13.2  Detecting selection

| Signal | Statistic | ML augmentation |
|--------|-----------|------------------|
| Reduced diversity | π, θ_W | CNN on haplotype matrices (`diploS/HIC`) |
| Long haplotypes | iHS, XP-EHH | RNN over windowed scans |
| Allele-frequency time series | s vs. drift | LSTM, Transformer |
| Co-evolution | DCA, ESM contacts | Transformer (`evoformer`) |

A useful guard: *always* match neutral background simulations to the empirical recombination map and demographic history.

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

## 13.4  Directed evolution with ML

Machine-learning-guided directed evolution (MLDE) iterates:

1. *Library design* — diverse mutations sampled from a PLM prior.
2. *Wet-lab screen* — fluorescence / activity readout.
3. *Surrogate fit* — GP or PLM-head on the labeled subset.
4. *Acquisition* — propose next round (UCB, Thompson, Bayesian optimization).

Three rounds typically beat any single round of random mutagenesis at ~10× fewer wet-lab variants.

## 13.5  Pitfalls

- **Demography ↔ selection confounding.** A bottleneck mimics positive selection. Always condition selection scans on a fitted demographic model.
- **Phylogeny error.** Single-tree analyses ignore inference uncertainty; use posterior summaries.
- **PLM bias.** Protein language models prefer the natural training distribution; do not interpret PLM scores as fitness without calibration.

## 13.6  Exercises

1. **Demographic inference.** Use `dadi` (analytical) and `dinf` (deep) on the same human Out-of-Africa SFS. Compare runtimes and parameter estimates.
2. **Selection scan.** Train `diploS/HIC` on simulations matched to *D. melanogaster*. Apply to DGRP whole-genome data. List your top 10 sweep regions.
3. **Phylogenetic GNN.** On the Open Tree of Life mammal subtree, predict body mass at internal nodes. Compare with Brownian-motion ancestral reconstruction.
4. **MLDE in silico.** Use the ProteinGym BLAT_ECOLX landscape. Run 3 rounds of UCB MLDE with an ESM-2 surrogate. Plot best-so-far vs. round.

## 13.7  Further reading

- Coop, G. *Population and Quantitative Genetics* (open textbook).
- Kelleher, J. *msprime.* PLoS Comp Biol (2016).
- Wittmann, B. *Advances in machine learning for directed evolution.* Curr Opin Struct Biol (2021).
- Korfmann, K. *Deep learning in population genetics.* Genome Biol Evol (2023).

## See also

- [Chapter 7 — Genomics & Gene Regulation](chapter_07_genomics.md)
- [Chapter 14 — Ecology & Conservation](chapter_14_ecology.md)
- [Chapter 17 — Biotechnology & Bioengineering](chapter_17_biotech.md)
