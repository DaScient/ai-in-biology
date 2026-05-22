# Chapter 6 — Modeling Living Systems

> *"All models are wrong; in biology, the question is whether they are wrong in a useful and falsifiable way."*

## Learning objectives

- Distinguish between mechanistic (ODE, PDE, agent-based) and statistical (regression, deep) models of biological systems, and articulate where hybrid models live.
- Formalize gene regulatory networks as dynamical systems and fit them from data.
- Use neural ODEs to learn continuous-time dynamics from sparsely sampled trajectories.
- Build a simple agent-based model for cell motility and chemotaxis, and analyze its emergent behavior.

## 6.1  Modeling taxonomy

| Approach | Strength | Weakness | When to use |
|----------|----------|----------|-------------|
| ODE / PDE | Mechanistic, interpretable | Hard to scale to many variables | Pathways with known stoichiometry |
| Boolean / logical | Qualitative, fast | Loses dynamics | Early-stage hypothesis generation |
| Agent-based | Captures heterogeneity & emergence | Computationally heavy | Tissue morphogenesis, ecology |
| Statistical (GLM, GAM) | Quantifies uncertainty | Assumes structural form | Inference under known design |
| Deep learning | Flexible, data-hungry | Black-box risk | Large, well-instrumented systems |
| Hybrid (PINN, NeuralODE, mechanistic-priored DL) | Best of both | Engineering cost | When mechanism is partial |

## 6.2  Gene regulatory networks as dynamical systems

Let `x ∈ ℝⁿ` denote gene expression. A common parameterization:

```
dx_i/dt = β_i · σ(Σ_j W_ij · x_j + b_i) − γ_i · x_i
```

where `σ` is a sigmoid and `W` encodes activation / repression. Fitting `(W, b, β, γ)` from time-series RNA-seq is essentially **dynamics-aware regression**; it benefits enormously from sparsity priors (most genes regulate few others).

## 6.3  Neural ODEs for cellular trajectories

```python
import torch
from torch import nn
from torchdiffeq import odeint

class CellODE(nn.Module):
    """f_theta(x, t) producing dx/dt for a cell-state vector x."""

    def __init__(self, d: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 1, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, d),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_in = t.expand(x.size(0), 1)
        return self.net(torch.cat([x, t_in], dim=-1))

# integrate from x0 over a grid of timepoints
# x_pred = odeint(CellODE(32), x0, t_grid, method="dopri5")
```

This is the basis for tools such as PRESCIENT, CellRank-NeuralODE, and dynamo.

## 6.4  Agent-based model — chemotactic motility

```python
import numpy as np

def chemotaxis_step(pos, grad_fn, speed=1.0, sigma=0.3, dt=0.1):
    """One step of biased random walk up a concentration gradient."""
    g = grad_fn(pos)                                    # (N, 2)
    drift = speed * g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    noise = sigma * np.random.randn(*pos.shape)
    return pos + drift * dt + noise * np.sqrt(dt)
```

Tile many such agents on a 2-D field with a source of attractant — emergent fronts and waves are visible within a few hundred steps.

## 6.5  Identifiability and uncertainty

A model that fits the data is not enough. You must also check:

- **Practical identifiability.** Profile-likelihood or MCMC posterior — wide, multimodal posteriors signal trouble.
- **Posterior predictive checks.** Simulate from the fitted model; do the synthetic data look like the real data?
- **Out-of-distribution generalization.** Hold out *perturbations*, not random samples.

## 6.6  Exercises

1. **GRN inference.** Fit a sparse linear ODE GRN to the DREAM5 in-silico dataset. Report AUROC on edge recovery.
2. **Neural ODE on dynamo.** Use the `dynamo` package to compute RNA velocity. Replace its analytical solver with the `CellODE` above. Compare trajectories.
3. **Tipping point.** Add a slow drift term to your chemotaxis ABM such that the collective state bifurcates. Plot the variance / autocorrelation of an order parameter near the bifurcation (early-warning signals).
4. **Identifiability.** For a 3-gene oscillator, sample 10 / 100 / 1000 time points. Show how the marginal posterior over the degradation rate narrows.

## 6.7  Further reading

- Chen, R. T. Q. *Neural ordinary differential equations.* NeurIPS (2018).
- Raissi, M. *Physics-informed neural networks.* JCP (2019).
- Aalto, A. *Gene regulatory network inference from sparse data.* Nat Commun (2020).
- Macklin, P. *Open-source agent-based modeling for biology.* (PhysiCell, 2018).

## See also

- [Chapter 9 — Single-Cell Intelligence](chapter_09_single_cell.md)
- [Chapter 10 — Development & Morphogenesis](chapter_10_development.md)
- [Chapter 14 — Ecology & Conservation](chapter_14_ecology.md)
