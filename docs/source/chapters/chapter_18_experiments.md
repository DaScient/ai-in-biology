# Chapter 18 — Experimental Design in the AI Era

> *"The right experiment is the one that maximally reduces the posterior entropy of the question you actually care about."*

## Learning objectives

- Apply Bayesian optimal experimental design (BOED) to choose perturbations and assays.
- Run active-learning workflows for label-efficient training in biology.
- Plan power analyses for AI-driven studies that go beyond classical p-values.
- Design experiments that are *robust* to model error: orthogonal validation, pre-registration, replication.

## 18.1  The classical and the Bayesian view

Classical design optimizes *what experiment minimizes variance* for a chosen estimator. Bayesian design optimizes *what experiment maximizes expected information gain*

```
EIG(d) = 𝔼_{y|d}[ H(θ) − H(θ | y, d) ]
```

For complex biological models, EIG is approximated with nested Monte Carlo, normalizing flows, or amortized variational inference (e.g. `BOLFI`, `DEEP-EI`).

## 18.2  Active learning, in practice

| Setting | Pool size | Acquisition |
|---------|-----------|-------------|
| Small (≤ 10⁴) | Enumerate | UCB, EI, BALD |
| Medium (10⁴–10⁶) | Sample | Stochastic batch BALD |
| Large (≥ 10⁶) | Streaming | Coresets, semi-supervised |

For sequence libraries, *diverse* batches (e.g. by k-means on PLM embeddings) outperform top-k UCB when the surrogate is poorly calibrated.

## 18.3  Power, then more power

For machine-learning studies, a "p-value < 0.05" is rarely the right target. Replace with:

- **Effect-size CI on a clinically / biologically meaningful metric.**
- **Bootstrap CIs at the *sample* (patient, cell line) level**, not the data point.
- **A pre-registered Δ** (minimal detectable improvement) and the n required to detect it.

For comparison of two models, McNemar / Wilcoxon on paired predictions is appropriate; permutation tests handle complex metrics.

## 18.4  Worked example — BALD acquisition

```python
import torch

def bald_score(model, X, n_samples=20):
    """Bayesian Active Learning by Disagreement, MC dropout approximation."""
    model.train()                                   # enable dropout
    preds = torch.stack([model(X) for _ in range(n_samples)])  # (S, N, K)
    p_mean = preds.mean(0)
    h_mean = -(p_mean * p_mean.log()).sum(-1)
    e_h = -(preds * preds.log()).sum(-1).mean(0)
    return h_mean - e_h                             # (N,)
```

Acquire the top-`b` indices. Validate that BALD beats random by ≥ 2× sample efficiency on a held-out task.

## 18.5  Designing against model error

- **Orthogonal assays.** Confirm hits with a fundamentally different readout (e.g. SPR after a yeast display screen).
- **Pre-registration.** Lock the analysis plan before unblinding.
- **Negative controls.** Include known no-effect inputs and scrambled controls.
- **Replication.** Plan power for replication, not just discovery.

## 18.6  Pitfalls

- **Acquisition-function tunnel vision.** UCB exploits early; cool β over rounds.
- **Pool poisoning.** Self-supervised pre-training on the candidate pool inflates apparent active-learning gains. Be honest about leakage.
- **Multiple comparisons.** A genome-wide ablation needs FDR control; a 20-way model comparison needs Holm or BH.

## 18.7  Exercises

1. **BALD vs. random.** On `permuted MNIST`-style biological splits, plot test accuracy vs. labels acquired for BALD and random.
2. **BOED for kinetics.** Estimate the EIG of three candidate sampling time points for a 4-parameter MM enzyme model.
3. **Pre-registration draft.** Write an OSF-style pre-registration for a small CRISPR screen analyzed with a sequence model.
4. **Replication plan.** Given a discovery effect size of 0.4 and σ = 1, what `n` is needed for 80 % power to replicate at α = 0.05?

## 18.8  Further reading

- Foster, A. *Variational Bayesian optimal experimental design.* NeurIPS (2019).
- Settles, B. *Active Learning Literature Survey.* (2009, still relevant).
- Nosek, B. *Preregistration is hard.* AmPsych (2018).
- Kapoor, S.; Narayanan, A. *Leakage and the reproducibility crisis in ML-based science.* Patterns (2023).

## See also

- [Chapter 6 — Modeling Living Systems](chapter_06_modeling.md)
- [Chapter 17 — Biotechnology & Bioengineering](chapter_17_biotech.md)
- [Chapter 19 — Ethics of AI in Biology](chapter_19_ethics.md)
