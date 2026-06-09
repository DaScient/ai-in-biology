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

### 6.1a  Modeling taxonomy: extended decision framework

The table above gives a high-level taxonomy. The following **decision matrix** helps a researcher choose between model classes based on concrete criteria.

| Criterion | Mechanistic (ODE/PDE/ABM) | Statistical (Regression/ML) | Hybrid (Neural ODE/PINN) |
|-----------|---------------------------|-----------------------------|---------------------------|
| **Data regime** | Sparse, structured perturbations | Large, high-dimensional observations | Moderate, with known constraints |
| **Prior knowledge** | Strong (known reactions, forces) | Weak (exploratory) | Partial (some equations known, some unknown) |
| **Interpretability** | High (parameters have meaning) | Low to moderate | Moderate (if structure preserved) |
| **Extrapolation** | Potentially good (if correct) | Poor outside training distribution | Better than pure statistical |
| **Computational cost** | Low to moderate (simulation) | Training expensive, inference cheap | Training very expensive |
| **Uncertainty quantification** | Well-developed (MCMC, profile likelihood) | Often ad hoc | Active research area |

**Rule of thumb:** Start with the simplest model that can answer your question. For predicting dynamics of a well-studied pathway, an ODE may suffice. For discovering new regulatory interactions from omics data, a statistical model is necessary. For forecasting under interventions not seen in training, hybrid models are promising.

**Pitfall:** Hybrid models can inherit the weaknesses of both approaches — poorly constrained parameters from the mechanistic part plus overfitting from the neural part. Always validate with out-of-distribution predictions.

## 6.2  Gene regulatory networks as dynamical systems

Let `x ∈ ℝⁿ` denote gene expression. A common parameterization:

```
dx_i/dt = β_i · σ(Σ_j W_ij · x_j + b_i) − γ_i · x_i
```

where `σ` is a sigmoid and `W` encodes activation / repression. Fitting `(W, b, β, γ)` from time-series RNA-seq is essentially **dynamics-aware regression**; it benefits enormously from sparsity priors (most genes regulate few others).

### 6.2a  From ODEs to sparse inference

Below is a **practical inference pipeline** for fitting such models from time-series data.

**Step 1: Discretize** the ODE using Euler or Runge–Kutta:

```
x_i(t+Δt) ≈ x_i(t) + Δt · [β_i·σ(Σ_j W_ij·x_j(t) + b_i) − γ_i·x_i(t)]
```

**Step 2: Define a loss function** that penalizes both prediction error and parameter magnitude:

```
L(W,β,γ,b) = Σ_t Σ_i (x_i_pred(t) − x_i_data(t))² + λ₁ Σ_ij |W_ij| + λ₂ Σ_i (β_i² + γ_i²)
```

**Step 3: Optimize** using gradient descent (if using neural ODE frameworks) or specialized solvers (e.g., `scipy.optimize` with sparsity constraints).

```python
import torch
import torch.nn as nn


class GRN_ODE(nn.Module):
    """Learnable gene regulatory network as ODE."""

    def __init__(self, n_genes, n_hidden=64):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_genes, n_genes) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_genes))
        self.beta = nn.Parameter(torch.ones(n_genes))
        self.gamma = nn.Parameter(torch.ones(n_genes))
        self.sparsity_mask = None

    def forward(self, t, x):
        # x: (batch, n_genes)
        interaction = torch.sigmoid(x @ self.W.t() + self.b)
        dx = self.beta * interaction - self.gamma * x
        return dx

    def apply_sparsity(self, mask):
        """Fix certain edges to zero (known prior)."""
        self.W.data *= mask
        self.sparsity_mask = mask

    def l1_regularization(self):
        return torch.sum(torch.abs(self.W))


# Training with torchdiffeq
from torchdiffeq import odeint


def fit_grn(model, t_grid, x0, x_data, l1_lambda=0.01, lr=1e-3, epochs=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_pred = odeint(model, x0, t_grid)  # (len(t_grid), batch, n_genes)
        mse = nn.MSELoss()(x_pred, x_data)
        l1 = model.l1_regularization()
        loss = mse + l1_lambda * l1
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: MSE {mse.item():.4f}, L1 {l1.item():.4f}")
    return model
```

**Pitfall:** Sparse ODEs often have many local optima. Run multiple random initializations and report the distribution of recovered edges, not just a single best fit.

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

### 6.3a  Neural ODEs: when to use and when to avoid

**Good use cases:**

- **Irregularly sampled time series** (e.g., patient visits, ecological censuses) — neural ODEs naturally handle arbitrary time grids.
- **Latent dynamics inference** — combine with a VAE to learn low-dimensional dynamics from high-dimensional observations (e.g., scRNA trajectories).
- **Continuum limits** — when you believe the underlying process is continuous and smooth.

**Poor use cases:**

- **Very high-dimensional state** (e.g., >1000 genes directly) — solving the ODE becomes expensive; better to first compress with an autoencoder.
- **Discrete or abrupt changes** (e.g., cell division, phase transitions) — standard neural ODEs assume Lipschitz continuity; consider neural jump ODEs or hybrid models.
- **Small dataset (<100 trajectories)** — neural ODEs overfit easily; use simpler mechanistic models.

**Memory efficiency:** The adjoint method (used by `torchdiffeq`) trades memory for compute — fine for small models, but for large models you may need to checkpoint intermediate states.

```python
# Using torchdiffeq with checkpointing (reduces memory at cost of recomputation)
from torchdiffeq import odeint_adjoint

x_pred = odeint_adjoint(model, x0, t_grid, method='dopri5',
                        adjoint_options={'norm': 'seminorm'})
```

**Pitfall:** Neural ODEs are sensitive to the choice of numerical solver. `dopri5` (adaptive) is standard but may be slow; `euler` is fast but inaccurate. Always test with a known solution.

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

### 6.4a  Agent-based models: from code to emergence

Let us expand to a **minimal but complete ABM** for cell proliferation and competition, then analyze emergent properties.

**Model description:** Cells on a 2D grid. Each cell has a position, a type (A or B), a proliferation rate (probability to divide per time step), and a death rate. Cells compete for space: if a cell divides and the target site is occupied, the division fails (contact inhibition).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


class CellCompetitionABM:
    def __init__(self, size=100, init_A=500, init_B=500,
                 r_A=0.05, r_B=0.05, d_A=0.01, d_B=0.02):
        self.grid = np.zeros((size, size), dtype=int)  # 1 = A, 2 = B
        # Place initial cells randomly
        coords_A = np.random.choice(size * size, init_A, replace=False)
        coords_B = np.random.choice(size * size, init_B, replace=False)
        self.grid.flat[coords_A] = 1
        self.grid.flat[coords_B] = 2
        self.params = {'r_A': r_A, 'r_B': r_B, 'd_A': d_A, 'd_B': d_B}
        self.history = []

    def step(self):
        new_grid = self.grid.copy()
        occupied = self.grid > 0
        # Identify cells that will divide (based on rate)
        div_A = (self.grid == 1) & (np.random.rand(*self.grid.shape) < self.params['r_A'])
        div_B = (self.grid == 2) & (np.random.rand(*self.grid.shape) < self.params['r_B'])
        # Identify cells that die
        die_A = (self.grid == 1) & (np.random.rand(*self.grid.shape) < self.params['d_A'])
        die_B = (self.grid == 2) & (np.random.rand(*self.grid.shape) < self.params['d_B'])

        # Remove dead cells
        new_grid[die_A | die_B] = 0

        # Attempt divisions: find an empty neighbor (Moore neighborhood)
        for (r, c) in zip(*np.where(div_A)):
            neighbors = self._empty_neighbors(r, c, occupied)
            if neighbors:
                nr, nc = neighbors[np.random.choice(len(neighbors))]
                new_grid[nr, nc] = 1
        for (r, c) in zip(*np.where(div_B)):
            neighbors = self._empty_neighbors(r, c, occupied)
            if neighbors:
                nr, nc = neighbors[np.random.choice(len(neighbors))]
                new_grid[nr, nc] = 2

        self.grid = new_grid
        self.history.append({
            't': len(self.history),
            'A': (self.grid == 1).sum(),
            'B': (self.grid == 2).sum(),
        })

    def _empty_neighbors(self, r, c, occupied):
        """Return list of (nr,nc) empty neighbor positions."""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1]:
                    if not occupied[nr, nc]:
                        neighbors.append((nr, nc))
        return neighbors

    def run(self, steps=100):
        for _ in range(steps):
            self.step()
        return self.history
```

**Emergent property:** Even if A has a higher proliferation rate, B may win if it has a lower death rate. Simulate and plot the population dynamics to see this.

## 6.5  Identifiability and uncertainty

A model that fits the data is not enough. You must also check:

- **Practical identifiability.** Profile-likelihood or MCMC posterior — wide, multimodal posteriors signal trouble.
- **Posterior predictive checks.** Simulate from the fitted model; do the synthetic data look like the real data?
- **Out-of-distribution generalization.** Hold out *perturbations*, not random samples.

### 6.5a  Practical diagnostics for biological models

**Profile likelihood method:** For each parameter θ_k, fix its value and re-optimize all other parameters; compute the likelihood ratio relative to the MLE. A flat profile indicates non-identifiability.

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2


def profile_likelihood(model, data, param_names, param_index,
                       param_range, fixed_params=None):
    """
    Compute profile likelihood for a single parameter.
    model: function that returns negative log-likelihood given parameter vector
    data: observations
    param_names: list of names
    param_index: index of parameter to profile
    param_range: array of values to test
    fixed_params: dict of other parameters fixed to specific values (optional)
    """
    nll_at_mle = model(data, **{p: fixed_params[p] for p in param_names if p != param_names[param_index]})
    profile = []
    for val in param_range:
        # Fix parameter to val, optimize others
        def objective(x):
            full_params = {p: fixed_params.get(p, None) for p in param_names}
            full_params[param_names[param_index]] = val
            # x contains free parameters (excluding fixed and profiled)
            # mapping left as exercise
            return model(data, **full_params)
        # Optimize (simplified)
        res = minimize(objective, x0=[0] * len(free_params), method='L-BFGS-B')
        profile.append(2 * (res.fun - nll_at_mle))  # likelihood ratio statistic
    # Compute confidence interval threshold (chi2, df=1)
    threshold = chi2.ppf(0.95, 1)
    ci_lower = param_range[np.where(np.array(profile) <= threshold)[0][0]]
    ci_upper = param_range[np.where(np.array(profile) <= threshold)[0][-1]]
    return param_range, profile, ci_lower, ci_upper
```

**Practical check:** For a 3-gene oscillator model (repressilator), vary the degradation rate. If the profile does not exceed the threshold over a wide range, the parameter is not identifiable from the given data — you need a different experiment (e.g., perturbation).

**Pitfall:** Profile likelihood assumes correct model specification. If the model is wrong, profiles can be artificially flat or misleading.

## 6.6  Exercises

1. **GRN inference.** Fit a sparse linear ODE GRN to the DREAM5 in-silico dataset. Report AUROC on edge recovery.
2. **Neural ODE on dynamo.** Use the `dynamo` package to compute RNA velocity. Replace its analytical solver with the `CellODE` above. Compare trajectories.
3. **Tipping point.** Add a slow drift term to your chemotaxis ABM such that the collective state bifurcates. Plot the variance / autocorrelation of an order parameter near the bifurcation (early-warning signals).
4. **Identifiability.** For a 3-gene oscillator, sample 10 / 100 / 1000 time points. Show how the marginal posterior over the degradation rate narrows.
5. **(6.6d) GRN edge recovery with sparsity priors.** Simulate data from a known 5-gene ODE network (you design the `W` matrix). Add Gaussian noise. Fit the model (Section 6.2a) with and without the true sparsity mask. Report AUROC for edge recovery.
6. **(6.6e) Mutation-driven coexistence.** Extend the cell competition ABM (Section 6.4a) with a mutation operator: when a cell divides, with probability 0.001 its offspring changes type (A→B or B→A). Observe how mutation-driven coexistence emerges.
7. **(6.6f) ODE vs. neural ODE for predator–prey.** Simulate Lotka–Volterra dynamics (known ODE). Generate noisy observations at irregular times. Fit both a classical ODE (with known form, unknown parameters) and a neural ODE (unknown dynamics). Compare prediction error on held-out trajectories. Which one extrapolates better to unseen initial conditions?
8. **(6.6g) ABM parameter calibration.** Take the cell competition ABM above. Simulate a time series of A and B counts. Use approximate Bayesian computation (ABC) or a simple grid search to recover `r_A, r_B, d_A, d_B` from the time series. Report correlation between true and inferred values.
9. **(6.6h) Identifiability for a gene regulatory network.** Generate data from a 4-gene linear ODE (`dx/dt = A x`). Add noise. Fit the model with the full `A` matrix (16 parameters). Compute the condition number of the Fisher information matrix (see Chapter 23). How many parameters are practically identifiable with N = 10, 50, 200 time points?

## 6.7  Further reading

- Chen, R. T. Q. *Neural ordinary differential equations.* NeurIPS (2018).
- Raissi, M. *Physics-informed neural networks.* JCP (2019).
- Aalto, A. *Gene regulatory network inference from sparse data.* Nat Commun (2020).
- Macklin, P. *Open-source agent-based modeling for biology.* (PhysiCell, 2018).
- Rackauckas, C. *et al.* (2020). "Universal differential equations for scientific machine learning." *arXiv* — foundational paper for hybrid models.
- Fröhlich, F. *et al.* (2019). "Inference for stochastic chemical kinetics using moment equations and hybrid models." *Bioinformatics* — practical identifiability.
- Lueckmann, J-M. *et al.* (2021). "Benchmarking simulation-based inference." *PMLR* — comparing methods for parameter estimation in biological models.
- Metzcar, J. *et al.* (2019). "A review of cell-based computational modeling in cancer biology." *JCO Clinical Cancer Informatics* — ABM applications.

## See also

- [Chapter 9 — Single-Cell Intelligence](chapter_09_single_cell.md)
- [Chapter 10 — Development & Morphogenesis](chapter_10_development.md)
- [Chapter 14 — Ecology & Conservation](chapter_14_ecology.md)


---
<sub>Support DaScient, Inc. (a non-profit promoting accessible intelligence and community learning) via [Donations](https://cash.app/dascient/).</sub>
