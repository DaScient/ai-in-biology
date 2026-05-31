# Chapter 23 — Limits & Open Questions

> *"Knowing where a method fails is more useful than one more decimal place of where it succeeds."*

## Learning objectives

- Catalogue the principled limits of current biological AI: data, identifiability, generalization, and interpretability.
- Distinguish *epistemic* limits (fixable with more data/compute) from *fundamental* ones (intrinsic to the problem).
- Calibrate expectations using out-of-distribution and causal reasoning failures.
- Frame open research questions precisely enough to be falsifiable.

## 23.1  A taxonomy of limits

| Limit | Nature | Example |
|-------|--------|---------|
| **Data** | Epistemic | Rare diseases, non-model organisms, under-sampled populations |
| **Identifiability** | Often fundamental | Many parameter sets fit the same time series (Chapter 6) |
| **Generalization** | Mixed | Human-trained models fail on plants, new scanners, new sites |
| **Causality** | Fundamental | Correlation in observational omics ≠ mechanism |
| **Interpretability** | Mixed | Attention is not attribution (Chapter 3) |

The most common error in the field is treating a **fundamental** limit as if more data will solve it.

## 23.1a  A taxonomy of limits — extended with diagnostic tests

A label like "fundamental" only helps if you can *test* whether you are hitting a real barrier or merely an engineering problem. The table below pairs each limit with a quantitative diagnostic, a pass condition, and what to conclude in either case.

| Limit | Diagnostic test | Pass condition | If fails (fundamental) | If passes (fixable) |
|-------|----------------|----------------|------------------------|---------------------|
| **Data scarcity** | Learning curve: performance vs. sample size (log–log) | Slope > 0.3 (still improving) | Performance plateaus despite more data → model capacity or inductive bias wrong | Collect more data or use data augmentation |
| **Identifiability** | Fisher information matrix condition number | < 1e8 | Parameters cannot be uniquely recovered → need a different experiment | Collect different measurements (e.g., perturbations) |
| **Generalization (OOD)** | Performance drop on shifted distribution relative to in-distribution | < 10% drop | Drop > 30% → model relies on spurious correlations | Collect training data from a broader distribution or use domain adaptation |
| **Interpretability** | Faithfulness of explanation (e.g., attention vs. perturbation) | Spearman correlation > 0.7 | Correlation < 0.3 → explanations are misleading | Use a more faithful method (e.g., Integrated Gradients, SHAP) |
| **Causality** | Performance on interventional data (e.g., CRISPR) vs. observational | R² intervention > 0.5 × R² observational | Much lower → model captures correlations, not causes | Need causal discovery or interventional training |

**Implementing the generalization diagnostic (out-of-distribution shift):**

```python
def ood_generalization_gap(model, X_id, y_id, X_ood, y_ood):
    """Compute performance drop from in-distribution to out-of-distribution."""
    score_id = model.score(X_id, y_id)   # e.g., accuracy or R^2
    score_ood = model.score(X_ood, y_ood)
    gap = (score_id - score_ood) / score_id if score_id > 0 else 1.0
    return gap, score_id, score_ood
```

**Pitfall:** Many papers report only in-distribution performance, hiding poor OOD generalization. Always report the OOD gap when claiming a model is "general."

## 23.2  The data wall

Foundation models exploit abundant, cheap data. Biology has pockets where data are intrinsically scarce:

- **Rare phenotypes** — by definition few examples exist.
- **Causal/perturbational data** — expensive relative to observational data.
- **Long time horizons** — evolution, ecology, and aging unfold over decades.

Self-supervision and simulation help, but **no amount of unlabeled sequence substitutes for the missing causal experiment.**

## 23.2a  The data wall — extended with a learning-curve extrapolation method

Before launching an expensive data-collection campaign, estimate *when more data will stop helping*. A practical tool is the **scaling law**: test error \( E \) often follows

\[
E = a \cdot N^{-b} + c
\]

where \( N \) is sample size, \( b \) is the learning rate (typically 0.1–0.5), and \( c \) is the irreducible error.

**Fit the scaling law from subsampled data:**

```python
from scipy.optimize import curve_fit
import numpy as np

def scaling_law(N, a, b, c):
    return a * N**(-b) + c

def predict_performance_at_scale(N_train, errors, target_N):
    """Given training set sizes and errors, predict error at target_N."""
    popt, _ = curve_fit(scaling_law, N_train, errors, p0=[1.0, 0.2, 0.0])
    a, b, c = popt
    predicted_error = scaling_law(target_N, a, b, c)
    # Also estimate how much error would reduce by doubling data
    current_error = scaling_law(N_train[-1], a, b, c)
    doubling_error = scaling_law(2 * N_train[-1], a, b, c)
    reduction = current_error - doubling_error
    return predicted_error, reduction, b
```

**Interpretation:** If \( b < 0.1 \), returns to scale are diminishing and more data will not help much. If \( b > 0.3 \), collecting more data is worthwhile. For many biological tasks (e.g., predicting clinical outcomes from genomics), \( b \) is often low (0.05–0.15) because of high irreducible noise.

**Pitfall:** Scaling laws assume the data distribution is stationary. For biological systems under evolution (e.g., rapidly mutating viruses), the distribution shifts, and scaling laws may overestimate improvement.

## 23.3  Identifiability and the limits of fit

A model can fit the data perfectly and still be wrong about mechanism. Diagnostics:

- **Profile likelihood / posterior width** — flat directions signal non-identifiability.
- **Sloppy models** — many biological models have a few stiff and many sloppy parameter directions; predictions can be robust even when parameters are not.
- **Held-out *perturbations*** — the honest test, not held-out random samples.

## 23.3a  Identifiability and the limits of fit — extended with a practical example

Consider a concrete biological case: fitting a two-compartment pharmacokinetic (PK) model to sparse data.

**Model:**

\[
C(t) = A e^{-\alpha t} + B e^{-\beta t}
\]

Parameters: \( A, B, \alpha, \beta \). From a single IV bolus with only 4 time points, which parameters are identifiable?

**Simulation and identifiability analysis:**

```python
import numpy as np
from scipy.optimize import curve_fit

def pk_model(t, A, alpha, B, beta):
    return A * np.exp(-alpha * t) + B * np.exp(-beta * t)

# Generate data with known parameters
true_params = [10.0, 0.5, 5.0, 0.05]   # A, alpha, B, beta
t = np.array([0.1, 1.0, 4.0, 12.0])    # sparse time points
C_true = pk_model(t, *true_params)
C_obs = C_true + np.random.normal(0, 0.2, size=len(t))

# Fit model
popt, pcov = curve_fit(pk_model, t, C_obs, p0=[8.0, 0.4, 6.0, 0.06])
# Compute condition number of covariance matrix
cond = np.linalg.cond(pcov)
print(f"Condition number: {cond:.2e}")

# Profile likelihood for beta (slow phase)
beta_range = np.linspace(0.01, 0.2, 20)
profile = []
for beta_val in beta_range:
    def fixed_beta_model(t, A, alpha, B):
        return pk_model(t, A, alpha, B, beta_val)
    popt_fixed, _ = curve_fit(fixed_beta_model, t, C_obs, p0=[8.0, 0.4, 6.0])
    resid = C_obs - fixed_beta_model(t, *popt_fixed)
    nll = np.sum(resid**2) / (2 * 0.2**2)   # approximate negative log likelihood
    profile.append(nll)
```

**Expected result:** Condition number > 1e10 (ill-conditioned). The profile for `beta` is flat → `beta` is not identifiable from sparse data. To identify it, you need late time points (e.g., 24 h, 48 h).

**Pitfall:** Even with a well-conditioned Fisher matrix, parameters may be *practically* non-identifiable due to noise. Use profile likelihood with a threshold (e.g., ΔNLL = 3.84 for a 95% CI) to assess.

## 23.4  Worked example — detecting non-identifiability

```python
import numpy as np

def is_identifiable(jacobian: np.ndarray, tol: float = 1e-6) -> dict:
    """Use the Fisher information (J^T J) spectrum to flag flat directions."""
    fim = jacobian.T @ jacobian
    eig = np.linalg.eigvalsh(fim)
    eig = np.clip(eig, 0, None)
    cond = eig.max() / max(eig.min(), 1e-30)   # 1e-30 floors the denominator to avoid divide-by-zero
    n_flat = int((eig < tol * eig.max()).sum())
    return {
        "condition_number": float(cond),
        "n_flat_directions": n_flat,        # >0 implies practical non-identifiability
        "identifiable": n_flat == 0 and cond < 1e8,
    }

# A near-singular Fisher information matrix => parameters cannot be uniquely recovered.
J = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-9]])
print(is_identifiable(J))
```

A large condition number or any flat direction means **the data do not constrain the model** — collect a different measurement rather than a larger one.

## 23.4a  Worked example extension — detecting non-identifiability with realistic data

The example above uses a synthetic Jacobian. Let's extend it to a **biological ODE model** — a simple two-gene toggle switch — with real-world noise.

**Two-gene toggle switch model:**

\[
\frac{du}{dt} = \frac{\alpha_1}{1+v^{\beta}} - \gamma_1 u
\]

\[
\frac{dv}{dt} = \frac{\alpha_2}{1+u^{\gamma}} - \gamma_2 v
\]

Parameters: \( \alpha_1, \alpha_2, \beta, \gamma, \gamma_1, \gamma_2 \). You measure \( u(t) \) and \( v(t) \) at a few time points.

**Compute the sensitivity matrix and assess identifiability:**

```python
import numpy as np
from scipy.integrate import odeint

def toggle_switch(state, t, alpha1, alpha2, beta, gamma, gamma1, gamma2):
    u, v = state
    du = alpha1 / (1 + v**beta) - gamma1 * u
    dv = alpha2 / (1 + u**gamma) - gamma2 * v
    return [du, dv]

def sensitivity_matrix(params, t_obs, u_obs, v_obs, eps=1e-6):
    n_params = len(params)
    n_obs = 2 * len(t_obs)
    S = np.zeros((n_obs, n_params))
    # Compute nominal trajectory
    state0 = [u_obs[0], v_obs[0]]
    sol_nom = odeint(toggle_switch, state0, t_obs, args=tuple(params))
    for i in range(n_params):
        params_pert = params.copy()
        params_pert[i] += eps
        sol_pert = odeint(toggle_switch, state0, t_obs, args=tuple(params_pert))
        # Finite-difference sensitivity for u and v at each time
        S[0::2, i] = (sol_pert[:, 0] - sol_nom[:, 0]) / eps
        S[1::2, i] = (sol_pert[:, 1] - sol_nom[:, 1]) / eps
    return S

# Compute Fisher information = S^T S
S = sensitivity_matrix(params, t_obs, u_obs, v_obs)
FIM = S.T @ S
cond = np.linalg.cond(FIM)
eigvals = np.linalg.eigvalsh(FIM)
flat_directions = np.sum(eigvals < 1e-6 * eigvals.max())
print(f"Condition number: {cond:.2e}, flat directions: {flat_directions}")
```

**Interpretation:** If `flat_directions > 0`, the parameters are not identifiable from this experiment. Redesign it (e.g., different initial conditions, perturbations, or more time points).

**Pitfall:** The sensitivity method assumes linearization around the nominal trajectory. For highly nonlinear systems, use profile likelihood or Bayesian MCMC instead.

## 23.5  Causality: the recurring ceiling

Most biological AI is correlational. To make causal claims you need one of:

- **Interventions** (CRISPR perturbations, drug challenges).
- **Natural experiments** (Mendelian randomization using genetic instruments).
- **Explicit causal models** (structural equations, do-calculus) with stated assumptions.

A predictive model that is excellent at association can still recommend a harmful intervention.

## 23.5a  Causality: the recurring ceiling — a practical guide to causal inference

Which causal method should you reach for? It depends on the data you can obtain. The decision tree below moves from the gold standard down to weaker observational approaches.

```text
Do you have randomized controlled trial (RCT) data?
    │
    ├─ Yes → Use RCT analysis (simple t-test or regression with treatment indicator). Gold standard.
    │
    └─ No → Do you have a natural experiment (e.g., Mendelian randomization, instrumental variable)?
            │
            ├─ Yes → Use instrumental variable regression (2SLS). Requires a valid instrument (e.g., genetic variant).
            │
            └─ No → Do you have time series with interventions (e.g., before/after treatment)?
                    │
                    ├─ Yes → Use difference-in-differences or interrupted time series.
                    │
                    └─ No → Observational data only.
                            │
                            ├─ Use structural causal models (e.g., DAG + do-calculus) if you can specify the graph.
                            └─ Or use causal discovery algorithms (PC, LiNGAM) to infer the graph from data (weak, needs large N).
```

**Mendelian randomization example (2SLS) in Python:**

```python
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

# Assume: X (exposure, e.g., LDL cholesterol), Y (outcome, e.g., heart disease)
# Z (instrument, e.g., a genetic variant associated with LDL but not confounders)
# First stage: regress X on Z
first_stage = sm.OLS(X, sm.add_constant(Z)).fit()
X_hat = first_stage.predict(sm.add_constant(Z))
# Second stage: regress Y on X_hat
second_stage = IV2SLS(Y, sm.add_constant(X_hat), None, sm.add_constant(Z)).fit()
causal_effect = second_stage.params[1]
print(f"Causal effect (LDL on heart disease): {causal_effect:.3f}")
```

**Pitfall:** Mendelian randomization assumes no horizontal pleiotropy (the instrument affects the outcome only through the exposure). Test this with sensitivity analyses (e.g., MR-Egger).

## 23.6  Open questions worth your time

1. **Generalizable cellular models** — a true "foundation model of the cell" that predicts unseen perturbations.
2. **From sequence to phenotype** — closing the genotype–environment–phenotype gap, not just structure.
3. **Sample-efficient causal discovery** in high-dimensional omics.
4. **Reliable uncertainty** that holds under distribution shift, not just in-distribution.
5. **Interpretable mechanism**, not post-hoc saliency — models whose internals correspond to biology.

## 23.6a  Open questions worth your time — extended with falsifiable hypotheses

An open question only drives progress once it is sharpened into something testable. The table below converts each question above into a specific sub-question and the minimum experiment that would answer it.

| Open question | Testable sub-question | Minimum experiment |
|---------------|------------------------|--------------------|
| **Generalizable cellular models** | Can a model pre-trained on 10 million cells predict the effect of a drug on a never-seen cell type better than a model trained only on that cell type? | Hold out one cell type, pre-train on the others, fine-tune on 100 labeled cells. Compare to training from scratch on those 100 cells. |
| **From sequence to phenotype** | Given a bacterial genome, can we predict colony morphology (a complex trait) with > 80% accuracy across 100 diverse strains? | Assemble a dataset of sequenced strains with imaged colonies. Train a transformer on genome + promoter regions. |
| **Sample-efficient causal discovery** | Can a causal discovery algorithm recover the correct direction of 10 synthetic gene regulatory edges from 200 single-cell perturbations (CRISPR) vs. 2000 observational samples? | Simulate data from a known 10-gene network. Compare the PC algorithm, NOTEARS, and a simple correlation baseline. |
| **Reliable uncertainty under shift** | Does a conformal prediction set cover the true label at 90% confidence on a test set from a different batch (e.g., different sequencing platform), without recalibration? | Train on one batch, apply conformal prediction (using a calibration set from the same batch) to another batch. Compute coverage. |
| **Interpretable mechanism** | Does a model that enforces a known biological constraint (e.g., mass conservation) produce more biologically plausible attention maps than a standard transformer? | Train two models on the same task; compare attention maps to ground-truth contacts (e.g., Hi-C). |

**Pitfall:** Many open questions are phrased as "we need better X." By converting them to a testable sub-question, you create a concrete benchmark that can drive progress.

## 23.7  Pitfalls

- **Benchmark saturation illusion.** A solved benchmark can hide an unsolved problem (test-set homology leakage).
- **Extrapolation overconfidence.** Models report high confidence far outside the training envelope.
- **Mechanism by anecdote.** One striking attention map is not evidence of a learned mechanism.

**Benchmark saturation illusion — how to detect it.** A task with a long-standing benchmark (e.g., protein thermostability prediction on ProteinGym) can show steady improvement, but the gains may stem from training-data leakage (homology) rather than genuine generalization. *Detection:* create a new test set of proteins with < 30% identity to any training protein and re-evaluate the state-of-the-art model. The performance drop often reveals the illusion.

**Extrapolation overconfidence — how to detect it.** A model may give high-confidence predictions far outside its training range (e.g., a DNA language model predicting variant effects on a chromosome with completely different GC content). *Detection:* compute the Mahalanobis distance of each test point to the training distribution, group test points by distance percentile, and show calibration (expected vs. observed accuracy) for each group. Poor calibration at high distances indicates overconfidence.

```python
def calibration_by_distance(model, X_train, X_test, y_test, distance_metric='mahalanobis'):
    import numpy as np
    from scipy.spatial.distance import mahalanobis
    from sklearn.covariance import EmpiricalCovariance
    # Fit covariance on training embeddings
    cov_est = EmpiricalCovariance().fit(X_train)
    dists = [mahalanobis(x, X_train.mean(axis=0), cov_est.covariance_) for x in X_test]
    # Bin by distance decile
    deciles = np.percentile(dists, np.linspace(0, 100, 11))
    for i in range(10):
        in_decile = (dists >= deciles[i]) & (dists < deciles[i + 1])
        if sum(in_decile) > 0:
            pred_prob = model.predict_proba(X_test[in_decile])[:, 1]
            acc = np.mean((pred_prob > 0.5) == y_test[in_decile])
            mean_conf = np.mean(pred_prob)
            print(f"Decile {i}: distance {deciles[i]:.2f}-{deciles[i+1]:.2f}, "
                  f"accuracy {acc:.3f}, mean confidence {mean_conf:.3f}")
```

**Pitfall:** Even with good calibration in-distribution, a model may be completely uncalibrated out-of-distribution. Always report OOD calibration.

## 23.8  Exercises

1. **Find the flat direction.** Build a 3-parameter ODE where two parameters are non-identifiable from a single observable. Confirm with `is_identifiable`.
2. **OOD calibration.** Take any classifier from earlier chapters; measure its confidence on inputs drawn progressively outside the training distribution.
3. **Causal vs. predictive.** Construct a synthetic dataset where the best predictor recommends the wrong intervention.
4. **Open-question proposal.** Pick one Section 23.6 question; write a falsifiable experiment that would make progress on it.
5. **Scaling law for a biological task.** Choose a public dataset (e.g., a regression task on gene expression prediction). Subsample training sizes of 10%, 20%, 50%, and 100% of the data. Fit a scaling law \( E = a N^{-b} + c \). Estimate how much error would reduce by doubling the data. Is it worth collecting more?
6. **Identifiability of a pharmacodynamics model.** Simulate data from a simple PD model (e.g., \( E = E_{max} C^n / (EC_{50}^n + C^n) \)). Fit the model and compute the condition number of the Fisher information matrix. Vary the number of concentration points (e.g., 3 vs. 10). At what point do all parameters become identifiable (condition number < 1e6)?
7. **Causal discovery on synthetic gene data.** Use the `causalgraphicalmodels` package to specify a 5-node DAG representing a small gene regulatory network. Simulate data from the model with additive noise. Run the PC algorithm and NOTEARS. Compare the structural Hamming distance (SHD) to the true graph. How many samples are needed to achieve SHD < 2?
8. **Benchmark saturation test.** Take a published state-of-the-art model for a protein fitness prediction task (e.g., ESM-1v on ProteinGym). Create a new test set of variants from a protein family not seen in the model's pre-training (use BLAST to filter). Evaluate the model and compare the reported performance (on the original benchmark) to the new performance. What is the drop? Does it suggest overfitting to homology?

## 23.9  Further reading

- Gutenkunst, R. *Universally sloppy parameter sensitivities in systems biology.* PLoS Comp. Biol. (2007).
- Pearl, J. *The Book of Why* (2018).
- Bender, E. *On the dangers of stochastic parrots.* FAccT (2021).
- Kapoor, S.; Narayanan, A. *Leakage and the reproducibility crisis in ML-based science.* Patterns (2023).
- Hestness, J. et al. *Deep learning scaling is predictable, empirically.* arXiv (2017).
- Ovadia, Y. et al. *Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift.* NeurIPS (2019).

## See also

- [Chapter 6 — Modeling Living Systems](chapter_06_modeling.md)
- [Chapter 18 — Experimental Design in the AI Era](chapter_18_experiments.md)
- [Chapter 24 — A New Biology](chapter_24_new_biology.md)
