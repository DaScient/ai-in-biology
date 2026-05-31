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

### 18.1a  The classical and the Bayesian view — extended with a concrete example

The definition above is abstract. Here is a **simple toy problem** that illustrates Bayesian optimal experimental design (BOED) end-to-end.

**Problem.** A chemical reaction has an unknown rate constant \( k \). You may measure the concentration at a single time point \( t \). Which \( t \) maximizes information about \( k \)?

**Model.** \( C(t) = C_0 e^{-kt} + \epsilon \), with \( \epsilon \sim \mathcal{N}(0, \sigma^2) \). Prior: \( k \sim \text{LogNormal}(\mu = 0, \sigma = 0.5) \), where \( \mu \) and \( \sigma \) are the mean and standard deviation of the *underlying normal* (i.e. of \( \log k \)), not of \( k \) itself.

**BOED approach.**

1. For each candidate \( t \), simulate many possible outcomes \( y \) given prior draws of \( k \).
2. Compute the posterior over \( k \) given \( y \).
3. Estimate \( \text{EIG} = H(p(k)) - \mathbb{E}_{y \mid t}\big[\, H(p(k \mid y)) \,\big] \).

**Implementation using Monte Carlo and kernel density estimation (KDE):**

```python
import numpy as np
from scipy.stats import norm, lognorm
from sklearn.neighbors import KernelDensity

def eig_estimation(t, n_prior=1000, n_sim=100, sigma=0.05, C0=1.0):
    # Prior samples of k (LogNormal)
    prior_k = lognorm.rvs(s=0.5, scale=np.exp(0), size=n_prior)  # mean log = 0
    # Prior entropy (analytical for LogNormal with location mu, shape s):
    #   H = mu + 0.5 + 0.5*log(2*pi*s**2)
    mu, s = 0.0, 0.5
    prior_entropy = mu + 0.5 + 0.5 * np.log(2 * np.pi * s ** 2)

    total_posterior_entropy = 0.0
    for _ in range(n_sim):
        # Sample a true k from the prior
        k_true = np.random.choice(prior_k)
        # Simulate measurement at time t
        C_obs = C0 * np.exp(-k_true * t) + np.random.normal(0, sigma)
        # Posterior weights via importance sampling
        log_lik = norm.logpdf(C_obs, loc=C0 * np.exp(-prior_k * t), scale=sigma)
        log_lik -= np.max(log_lik)  # numerical stability
        weights = np.exp(log_lik)
        weights /= weights.sum()
        # Estimate posterior entropy from weighted samples (KDE on resamples)
        posterior_samples = np.random.choice(prior_k, size=500, p=weights)
        kde = KernelDensity(bandwidth=0.05).fit(posterior_samples.reshape(-1, 1))
        log_dens = kde.score_samples(posterior_samples.reshape(-1, 1))
        posterior_entropy = -np.mean(log_dens)  # approximate
        total_posterior_entropy += posterior_entropy
    avg_posterior_entropy = total_posterior_entropy / n_sim
    return prior_entropy - avg_posterior_entropy

# Evaluate EIG over a grid of t
t_grid = np.linspace(0.1, 5.0, 20)
eig_values = [eig_estimation(t) for t in t_grid]
best_t = t_grid[np.argmax(eig_values)]
print(f"Optimal measurement time: {best_t:.2f}")
```

**Expected result.** EIG peaks near \( t \approx 1/k_{\text{median}} \), balancing sensitivity to \( k \) (longer times) against noise (shorter times). The experiment is chosen *without* knowing the true parameter value.

**Pitfall.** EIG estimation is computationally heavy (nested Monte Carlo). For large models, use amortized inference (e.g. normalizing flows) to approximate posterior entropy.

## 18.2  Active learning, in practice

| Setting | Pool size | Acquisition |
|---------|-----------|-------------|
| Small (≤ 10⁴) | Enumerate | UCB, EI, BALD |
| Medium (10⁴–10⁶) | Sample | Stochastic batch BALD |
| Large (≥ 10⁶) | Streaming | Coresets, semi-supervised |

For sequence libraries, *diverse* batches (e.g. by k-means on PLM embeddings) outperform top-k UCB when the surrogate is poorly calibrated.

### 18.2a  Active learning in practice — batch selection strategies

Section 18.2 mentions top-k UCB and diverse batches. Here are **four acquisition strategies** and their trade-offs:

| Strategy | When to use | Pros | Cons |
|----------|-------------|------|------|
| **UCB (Upper Confidence Bound)** | Surrogate well-calibrated, exploitation priority | Finds high fitness quickly | May select redundant variants |
| **BALD (Bayesian Active Learning by Disagreement)** | Model uncertainty high, need global exploration | Maximizes information gain | Computationally expensive (requires MC dropout) |
| **Diverse UCB (clustering + UCB)** | Batch selection where diversity matters | Balances exploration/exploitation | Cluster number is heuristic |
| **Random sampling** | Baseline, when no surrogate exists | Simple, no model bias | Sample-inefficient |

**Batched BALD acquisition.** The same mutual-information score, but returning a batch of indices:

```python
import numpy as np

def bald_acquisition(model, X_pool, n_samples=20, batch_size=16):
    """BALD: acquire points with highest mutual information between
    predictions and model parameters."""
    model.train()  # enable dropout
    preds = []
    for _ in range(n_samples):
        preds.append(model(X_pool).detach().numpy())
    preds = np.array(preds)  # (n_samples, n_pool, n_classes)
    # Entropy of the mean prediction
    p_mean = preds.mean(axis=0)  # (n_pool, n_classes)
    entropy_mean = -np.sum(p_mean * np.log(p_mean + 1e-10), axis=1)
    # Mean entropy of individual predictions
    entropy_indiv = -np.sum(preds * np.log(preds + 1e-10), axis=2).mean(axis=0)
    bald_scores = entropy_mean - entropy_indiv  # (n_pool,)
    # Select top batch_size indices
    top_idx = np.argsort(bald_scores)[-batch_size:]
    return top_idx
```

**Comparison experiment (protein fitness landscape).** Use a known landscape (e.g. ProteinGym) and compare UCB, BALD, diverse UCB, and random over 5 rounds of 96 variants each; plot best fitness vs. round.

**Pitfall.** BALD's mutual information is defined for classification; for regression, use **variance reduction** or **expected improvement** (see 18.4a).

## 18.3  Power, then more power

For machine-learning studies, a "p-value < 0.05" is rarely the right target. Replace with:

- **Effect-size CI on a clinically / biologically meaningful metric.**
- **Bootstrap CIs at the *sample* (patient, cell line) level**, not the data point.
- **A pre-registered Δ** (minimal detectable improvement) and the n required to detect it.

For comparison of two models, McNemar / Wilcoxon on paired predictions is appropriate; permutation tests handle complex metrics.

### 18.3a  Beyond p-values — effect sizes and the bootstrap

Section 18.3 suggests replacing p-values with effect-size CIs. Here is a **bootstrap procedure** for comparing two models on a paired dataset (e.g. the same test patients).

**Goal.** Estimate the difference in AUROC between Model A and Model B, with a confidence interval.

```python
from sklearn.utils import resample
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_model_comparison(y_true, pred_A, pred_B, n_bootstrap=2000, alpha=0.05):
    """Compute a bootstrap CI for ΔAUROC = AUROC_A - AUROC_B.

    Returns mean difference, lower, upper, and a one-sided p-value
    (proportion of bootstrap replicates with Δ <= 0).
    """
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = resample(range(n), replace=True, n_samples=n)
        auc_a = roc_auc_score(y_true[idx], pred_A[idx])
        auc_b = roc_auc_score(y_true[idx], pred_B[idx])
        diffs.append(auc_a - auc_b)
    diffs = np.array(diffs)
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    p_value = np.mean(diffs <= 0)  # one-sided test, H0: Δ <= 0
    return np.mean(diffs), lower, upper, p_value

# Example
y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
pred_A = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.4, 0.9, 0.85, 0.25, 0.75])
pred_B = np.array([0.3, 0.75, 0.35, 0.68, 0.15, 0.42, 0.88, 0.82, 0.28, 0.72])
mean_diff, lower, upper, p = bootstrap_model_comparison(y_true, pred_A, pred_B)
print(f"ΔAUROC = {mean_diff:.3f} [{lower:.3f}, {upper:.3f}], p={p:.4f}")
```

**Interpretation.** If the entire interval lies above 0, Model A is statistically superior. The method assumes no normality and respects the paired structure (same patients for both models).

**Pitfall.** Bootstrap on small test sets (\( n < 50 \)) can be unstable. Prefer a more conservative method (e.g. a permutation test with a pre-specified number of permutations).

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

### 18.4a  Worked example extension — MC-dropout UCB for batched regression

The BALD code above is for classification. Here we extend to **regression** (e.g. predicting protein fitness) and add batched acquisition with a diversity penalty.

For a model with MC dropout, the predictive variance over \( T \) stochastic forward passes is

\[
\text{Var}(y \mid x) = \frac{1}{T} \sum_{t=1}^{T} \big(f_{\theta_t}(x) - \bar{f}(x)\big)^2 ,
\]

which serves as a proxy for disagreement. The acquisition score combines mean and uncertainty in an upper confidence bound:

```python
import numpy as np

def ucb_mc_dropout(model, X_pool, n_samples=20, beta=1.0):
    """UCB with uncertainty from MC dropout (regression)."""
    model.train()  # enable dropout at inference
    preds = []
    for _ in range(n_samples):
        preds.append(model(X_pool).detach().numpy().flatten())
    preds = np.array(preds)  # (n_samples, n_pool)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean + beta * std

def diverse_ucb_mc(model, X_pool, batch_size, beta=1.0, cluster_frac=2):
    """Batched acquisition with a diversity constraint via clustering."""
    from sklearn.cluster import KMeans
    n_clusters = min(batch_size * cluster_frac, len(X_pool))
    # Cluster X_pool (e.g. using embeddings from the last hidden layer)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(X_pool)
    scores = ucb_mc_dropout(model, X_pool, beta=beta)
    selected = []
    for c in range(n_clusters):
        idx_in_cluster = np.where(clusters == c)[0]
        if len(idx_in_cluster) == 0:
            continue
        best = idx_in_cluster[np.argmax(scores[idx_in_cluster])]
        selected.append(best)
        if len(selected) >= batch_size:
            break
    return selected
```

**Expected gain.** Diversity-constrained UCB often outperforms standard UCB in the early rounds of protein engineering, where the surrogate is poorly calibrated.

**Pitfall.** MC dropout requires training with dropout *and* enabling it at inference. Not all models support this; for Gaussian processes, diversity is built into the acquisition function (e.g. q-EI).

## 18.5  Designing against model error

- **Orthogonal assays.** Confirm hits with a fundamentally different readout (e.g. SPR after a yeast display screen).
- **Pre-registration.** Lock the analysis plan before unblinding.
- **Negative controls.** Include known no-effect inputs and scrambled controls.
- **Replication.** Plan power for replication, not just discovery.

### 18.5a  Orthogonal assays and negative controls — a CRISPR-screen example

Section 18.5 lists orthogonal assays. Here is a **concrete confirmatory design** for a CRISPR screen.

**Scenario.** A genome-wide CRISPR knockout screen identified gene X as essential for proliferation. Off-target effects may cause false positives.

**Orthogonal validation plan.**

1. **CRISPR interference (CRISPRi)** — a different mechanism (dCas9 repressor, not DNA cutting) targeting the same gene.
2. **cDNA rescue** — overexpress the gene in the knockout background; if the phenotype reverses, the effect is specific.
3. **Two independent guide RNAs** — each targeting a different exon of gene X; concordant phenotypes make off-target effects unlikely.

**Statistical design.**

- Pre-register the criteria for "validation": e.g. at least two orthogonal methods show effect size > 2 standard deviations above negative controls.
- Control: non-targeting guides (20 per plate).
- Sample size: power calculation to detect a 30 % reduction in proliferation with \( \alpha = 0.01 \), \( \beta = 0.2 \) (80 % power).

```python
from statsmodels.stats.power import tt_ind_solve_power

effect_size = 0.3  # Cohen's d (standardized effect size), not a raw rate difference
alpha = 0.01
power = 0.8
n = tt_ind_solve_power(effect_size=effect_size, alpha=alpha,
                       power=power, ratio=1.0, alternative='two-sided')
print(f"Required n per group: {n:.0f}")
```

**Pitfall.** Orthogonal assays may measure different biological processes. A gene could be essential for proliferation by one mechanism but not another — the absence of validation does not prove the original hit was false.

## 18.6  Pitfalls

- **Acquisition-function tunnel vision.** UCB exploits early; cool β over rounds.
- **Pool poisoning.** Self-supervised pre-training on the candidate pool inflates apparent active-learning gains. Be honest about leakage.
- **Multiple comparisons.** A genome-wide ablation needs FDR control; a 20-way model comparison needs Holm or BH.

**Acquisition-function tunnel vision (extended).** UCB with a fixed β over-exploits early and never explores beyond the initial high-fitness region. *Remedy:* decay β over rounds, or use Thompson sampling, which naturally balances exploration and exploitation.

**Pool poisoning (extended).** If you pre-train a model on the entire candidate pool (including held-out variants), its uncertainty estimates are artificially low for those variants, inflating active-learning performance. *Solution:* never pre-train on candidates; split candidates into a training pool and a separate candidate pool at the start. *Diagnostic:* compare active-learning performance when the surrogate is pre-trained on unlabeled candidate embeddings vs. when it starts from random initialization — a large advantage from pre-training signals leakage. Many published active-learning papers in biology inadvertently use pool poisoning by pre-training on all sequences (including test variants) with self-supervised tasks. Always report whether pre-training included any candidate sequences.

## 18.7  Exercises

1. **BALD vs. random.** On `permuted MNIST`-style biological splits, plot test accuracy vs. labels acquired for BALD and random.
2. **BOED for kinetics.** Estimate the EIG of three candidate sampling time points for a 4-parameter MM enzyme model.
3. **Pre-registration draft.** Write an OSF-style pre-registration for a small CRISPR screen analyzed with a sequence model.
4. **Replication plan.** Given a discovery effect size of 0.4 and σ = 1, what `n` is needed for 80 % power to replicate at α = 0.05?
5. **BOED for enzyme kinetics.** For a Michaelis–Menten enzyme (\( V_{\max} \), \( K_m \)), simulate data and compute EIG for three candidate designs: (a) measure rate at a single substrate concentration, (b) measure at two concentrations, (c) measure at four concentrations (same total number of measurements). Which design yields the narrowest posterior on \( V_{\max} \) and \( K_m \)?
6. **Compare acquisition functions on a landscape.** Use the ProteinGym `AAT` (aminotransferase) landscape. Run 5 rounds of 48 variants each, using (i) random, (ii) UCB, (iii) BALD, (iv) diverse UCB, and (v) Thompson sampling. Plot best fitness and mean fitness of the selected batch. Which method reaches the highest best fitness fastest? Which has the highest mean batch fitness (useful for pooled screening)?
7. **Bootstrap model comparison on clinical data.** Take a public clinical prediction dataset (e.g. MIMIC mortality). Train two models (e.g. logistic regression and random forest). Use the bootstrap to compute the CI for ΔAUROC. Is the difference statistically significant? Repeat after applying a Bonferroni correction for multiple comparisons.
8. **Orthogonal validation design for a drug target.** You have identified a gene candidate for a cancer drug target via a CRISPR screen. Write a one-page pre-registration document specifying primary and secondary assays, orthogonal validation methods, a sample-size calculation, and success criteria. Use the template from the Open Science Framework.

## 18.8  Further reading

- Foster, A. *Variational Bayesian optimal experimental design.* NeurIPS (2019).
- Settles, B. *Active Learning Literature Survey.* (2009, still relevant).
- Nosek, B. *Preregistration is hard.* AmPsych (2018).
- Kapoor, S.; Narayanan, A. *Leakage and the reproducibility crisis in ML-based science.* Patterns (2023).
- Rainforth, T. et al. *Bayesian experimental design: a review.* Journal of the American Statistical Association (2024) — modern methods including amortized BOED.
- Garnett, R. *Bayesian Optimization* (book, 2023) — comprehensive treatment of acquisition functions.
- Button, K. S. et al. *Power failure: why small sample size undermines the reliability of neuroscience.* Nature Reviews Neuroscience (2013) — still relevant for AI biology studies.

## See also

- [Chapter 6 — Modeling Living Systems](chapter_06_modeling.md)
- [Chapter 17 — Biotechnology & Bioengineering](chapter_17_biotech.md)
- [Chapter 19 — Ethics of AI in Biology](chapter_19_ethics.md)
