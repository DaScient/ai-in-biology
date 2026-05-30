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

## 23.2  The data wall

Foundation models exploit abundant, cheap data. Biology has pockets where data are intrinsically scarce:

- **Rare phenotypes** — by definition few examples exist.
- **Causal/perturbational data** — expensive relative to observational data.
- **Long time horizons** — evolution, ecology, and aging unfold over decades.

Self-supervision and simulation help, but **no amount of unlabeled sequence substitutes for the missing causal experiment.**

## 23.3  Identifiability and the limits of fit

A model can fit the data perfectly and still be wrong about mechanism. Diagnostics:

- **Profile likelihood / posterior width** — flat directions signal non-identifiability.
- **Sloppy models** — many biological models have a few stiff and many sloppy parameter directions; predictions can be robust even when parameters are not.
- **Held-out *perturbations*** — the honest test, not held-out random samples.

## 23.4  Worked example — detecting non-identifiability

```python
import numpy as np

def is_identifiable(jacobian: np.ndarray, tol: float = 1e-6) -> dict:
    """Use the Fisher information (J^T J) spectrum to flag flat directions."""
    fim = jacobian.T @ jacobian
    eig = np.linalg.eigvalsh(fim)
    eig = np.clip(eig, 0, None)
    cond = eig.max() / max(eig.min(), 1e-30)
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

## 23.5  Causality: the recurring ceiling

Most biological AI is correlational. To make causal claims you need one of:

- **Interventions** (CRISPR perturbations, drug challenges).
- **Natural experiments** (Mendelian randomization using genetic instruments).
- **Explicit causal models** (structural equations, do-calculus) with stated assumptions.

A predictive model that is excellent at association can still recommend a harmful intervention.

## 23.6  Open questions worth your time

1. **Generalizable cellular models** — a true "foundation model of the cell" that predicts unseen perturbations.
2. **From sequence to phenotype** — closing the genotype–environment–phenotype gap, not just structure.
3. **Sample-efficient causal discovery** in high-dimensional omics.
4. **Reliable uncertainty** that holds under distribution shift, not just in-distribution.
5. **Interpretable mechanism**, not post-hoc saliency — models whose internals correspond to biology.

## 23.7  Pitfalls

- **Benchmark saturation illusion.** A solved benchmark can hide an unsolved problem (test-set homology leakage).
- **Extrapolation overconfidence.** Models report high confidence far outside the training envelope.
- **Mechanism by anecdote.** One striking attention map is not evidence of a learned mechanism.

## 23.8  Exercises

1. **Find the flat direction.** Build a 3-parameter ODE where two parameters are non-identifiable from a single observable. Confirm with `is_identifiable`.
2. **OOD calibration.** Take any classifier from earlier chapters; measure its confidence on inputs drawn progressively outside the training distribution.
3. **Causal vs. predictive.** Construct a synthetic dataset where the best predictor recommends the wrong intervention.
4. **Open-question proposal.** Pick one Section 23.6 question; write a falsifiable experiment that would make progress on it.

## 23.9  Further reading

- Gutenkunst, R. *Universally sloppy parameter sensitivities in systems biology.* PLoS Comp. Biol. (2007).
- Pearl, J. *The Book of Why* (2018).
- Bender, E. *On the dangers of stochastic parrots.* FAccT (2021).
- Kapoor, S.; Narayanan, A. *Leakage and the reproducibility crisis in ML-based science.* Patterns (2023).

## See also

- [Chapter 6 — Modeling Living Systems](chapter_06_modeling.md)
- [Chapter 18 — Experimental Design in the AI Era](chapter_18_experiments.md)
- [Chapter 24 — A New Biology](chapter_24_new_biology.md)
