# Chapter 19 — Ethics of AI in Biology

> *"In biology the cost of a wrong prediction is rarely a bad recommendation; it is a misdiagnosed patient, a released organism, or a leaked genome."*

## Learning objectives

After studying this chapter you will be able to:

- Map the dominant ethical risk classes in biological AI: privacy, fairness, consent, dual-use, and ecological harm.
- Apply a structured risk-assessment workflow before deploying a model on biological or clinical data.
- Distinguish *fairness* notions (demographic parity, equalized odds, calibration) and reason about why they cannot all hold at once.
- Implement basic privacy and bias-audit checks as part of a model-release checklist.

## 19.1  Why biology raises the stakes

General machine-learning ethics (transparency, accountability, fairness) all apply, but biology adds properties that sharpen the risk:

| Property | Consequence |
|----------|-------------|
| **Data are about people** | Genomes, scans, and records are identifying and often immutable. |
| **Errors reach the body** | A model output can change a treatment, a release, or a diagnosis. |
| **Information is dual-use** | The same model that designs a vaccine can inform a pathogen. |
| **Effects are irreversible** | A gene drive or an introduced species cannot be recalled. |

The unifying principle: **the burden of proof scales with the irreversibility and reach of the decision the model informs.**

## 19.1a  Concrete risk scenarios

Anchoring each abstract risk class to a real-world example makes the duty of care tangible.

| Risk class | Example | Consequence | Mitigation |
|------------|---------|-------------|------------|
| **Irreversibility** | Release of gene-drive mosquitoes in a field trial | Edited allele spreads across population boundaries; cannot be recalled | Phased testing (laboratory → confined field → limited release); molecular containment (split drives) |
| **Heritability** | Germline genome editing (CRISPR in human embryos) | Offspring inherit edits; unknown long-term effects | Moratorium on clinical germline editing; public deliberation |
| **Dual-use** | AI-designed toxin with high lethality and stability | Lowered barrier for bioweapons development | Sequence screening (Chapter 17); model refusal training; access control |
| **Identity** | Re-identification of anonymized genomes from consumer databases | Disclosure of disease predisposition without consent | Differential privacy; consent for secondary use; legal penalties |
| **Ecological scale** | AI-optimized pesticide deployment | Local extinction of non-target insects; ecosystem disruption | Environmental impact assessment; adaptive management with monitoring |

**Key insight:** The same AI model that accelerates drug discovery can also accelerate harm. The ethical duty scales with the magnitude of potential harm, not just the intent of the user.

**Pitfall:** Generic "responsible AI" checklists often omit biology-specific risks. Always add a **dual-use and ecological impact** section to your model documentation.

## 19.2  Privacy and consent

- **Re-identification.** Genomic data are quasi-identifiers; even "anonymized" SNP sets can be matched to named individuals via genealogy databases.
- **Consent decay.** Broad consent collected for one study rarely covers training a foundation model years later. Document the chain of consent.
- **Federated and private learning.** Differential privacy (DP-SGD), federated learning, and secure enclaves let models learn without centralizing raw records — at a measurable utility cost.

## 19.2a  Differential privacy for genomic data

Suppose you want to predict a phenotype (e.g., diabetes) from SNPs, but you cannot release individual genotypes. **Differential privacy (DP)** guarantees that the model's output does not depend too much on any single individual. **DP-SGD** (Abadi et al., 2016) clips per-sample gradients and adds calibrated noise during training.

```python
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset


class SNPClassifier(nn.Module):
    """Minimal feed-forward classifier over a SNP vector."""

    def __init__(self, n_snps):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_snps, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)


# X_snps: (n_samples, n_snps) binary; y: (n_samples,) binary
dataset = TensorDataset(torch.FloatTensor(X_snps), torch.FloatTensor(y))
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SNPClassifier(n_snps=X_snps.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Attach the privacy engine: noise_multiplier trades utility for privacy.
privacy_engine = PrivacyEngine()
model, optimizer, loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    noise_multiplier=1.0,   # higher = more privacy
    max_grad_norm=1.0,      # per-sample gradient clipping bound
)

for epoch in range(10):
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        pred = model(x_batch).squeeze()
        loss = nn.BCELoss()(pred, y_batch)
        loss.backward()
        optimizer.step()

# Report the privacy budget actually spent.
epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"Privacy budget used: ε={epsilon:.2f}")
```

**Utility–privacy trade-off.** Sweep the noise multiplier (e.g., 0.5, 1.0, 2.0, 5.0), record test accuracy and the spent `ε`, and plot accuracy vs. `ε`. For genomic data, `ε ≈ 3–5` often yields acceptable utility (accuracy drop < 5%).

**Pitfall:** DP-SGD reduces accuracy, especially for rare variants. For small sample sizes (< 10,000) it can degrade utility to near-random. In that regime, prefer federated learning with secure aggregation.

## 19.3  Fairness is plural and contested

For a clinical classifier with score `s`, sensitive attribute `a`, and label `y`:

- **Demographic parity**: `P(ŷ=1 | a)` equal across groups.
- **Equalized odds**: `P(ŷ=1 | y, a)` equal across groups.
- **Calibration**: `P(y=1 | s, a)` equal across groups.

An impossibility result (Kleinberg et al., 2016; Chouldechova, 2017) shows that, except in degenerate cases, calibration and equalized odds cannot both hold when base rates differ. **You must choose which fairness criterion matches the clinical harm model — there is no universal answer.**

## 19.3a  A worked fairness trade-off

To make the impossibility tangible, simulate two populations whose risk scores are *perfectly calibrated* by construction, then check equalized odds.

```python
import numpy as np
from sklearn.metrics import confusion_matrix


def simulate_calibrated_risk(n=10000):
    """Generate a perfectly calibrated risk score and matching labels."""
    risk = np.random.uniform(0, 1, n)         # score
    y = np.random.binomial(1, risk)            # P(y=1 | risk) = risk
    return risk, y


# Group A and Group B differ in prevalence via the score distribution.
risk_A, y_A = simulate_calibrated_risk(n=10000)
risk_B, y_B = simulate_calibrated_risk(n=10000)

t = 0.2                                        # treat if risk > t
pred_A = (risk_A > t).astype(int)
pred_B = (risk_B > t).astype(int)


def fpr_fnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn), fn / (fn + tp)


fpr_A, fnr_A = fpr_fnr(y_A, pred_A)
fpr_B, fnr_B = fpr_fnr(y_B, pred_B)
print(f"Group A: FPR={fpr_A:.3f}, FNR={fnr_A:.3f}")
print(f"Group B: FPR={fpr_B:.3f}, FNR={fnr_B:.3f}")
```

When the groups have different base rates, calibration holds by construction but the false-positive and false-negative rates diverge, so equalized odds (`FPR_A = FPR_B` and `FNR_A = FNR_B`) cannot also hold.

**Moral:** Choose the fairness metric that matches the clinical harm model:

- **Calibration** — when individual risk estimates drive shared decision-making.
- **Equalized odds** — when algorithmic decisions may cause disparate impact (e.g., loan approval, pretrial detention).

**Pitfall:** Many papers report only one fairness metric, implying the model is "fair." Always report at least two (including calibration) and discuss the trade-offs.

## 19.4  Worked example — a subgroup bias audit

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def subgroup_audit(y_true, y_score, group, min_n: int = 50) -> dict:
    """Per-group AUROC with a flag for disparities beyond 0.05."""
    report, aucs = {}, {}
    for g in np.unique(group):
        mask = group == g
        if mask.sum() < min_n or len(np.unique(y_true[mask])) < 2:
            report[g] = "insufficient data"
            continue
        aucs[g] = roc_auc_score(y_true[mask], y_score[mask])
        report[g] = round(aucs[g], 3)
    if aucs:
        report["max_disparity"] = round(max(aucs.values()) - min(aucs.values()), 3)
        report["flag"] = report["max_disparity"] > 0.05
    return report
```

A disparity flag is the *start* of an investigation — diagnose whether it stems from sample size, label noise, or genuine model bias before mitigating.

## 19.4a  Bias across thresholds — equal opportunity difference

A model can be fair at one operating point and biased at another. The **equal opportunity difference** measures the true-positive-rate gap between groups at a threshold `t`: \( \text{EOD}(t) = |TPR_A(t) - TPR_B(t)| \). A model is fair if `EOD(t) ≤ δ` for every clinically relevant `t`.

```python
def equal_opportunity_difference(y_true_A, y_score_A,
                                 y_true_B, y_score_B,
                                 n_thresholds=100):
    """Maximum equal opportunity difference over a threshold sweep."""
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_A, tpr_B = [], []
    for t in thresholds:
        pred_A = (y_score_A >= t).astype(int)
        pred_B = (y_score_B >= t).astype(int)
        _, _, fn_A, tp_A = confusion_matrix(y_true_A, pred_A).ravel()
        _, _, fn_B, tp_B = confusion_matrix(y_true_B, pred_B).ravel()
        tpr_A.append(tp_A / (tp_A + fn_A) if (tp_A + fn_A) > 0 else 0)
        tpr_B.append(tp_B / (tp_B + fn_B) if (tp_B + fn_B) > 0 else 0)
    eod = np.abs(np.array(tpr_A) - np.array(tpr_B))
    return eod.max(), thresholds[np.argmax(eod)], thresholds, eod
```

**Use case:** For a sepsis-prediction model you might require `EOD ≤ 0.05` across all thresholds. If the maximum EOD exceeds 0.05, the model may disadvantage one group at the clinically relevant threshold.

**Pitfall:** EOD only captures true-positive-rate differences. It ignores false-positive differences, which may matter just as much (e.g., the harms of unnecessary treatment).

## 19.5  Dual-use and responsible disclosure

Generative biology models can lower the barrier to harm. Minimum responsible practice:

1. **Threat-model before training** — could the capability uplift a malicious actor?
2. **Capability evaluations** — red-team the model for hazardous outputs.
3. **Staged release** — gate weights, add refusal layers, log regulated requests (see Chapter 17).
4. **Coordinated disclosure** — notify biosecurity bodies before publishing dangerous methods.

## 19.5a  A model refusal layer

A refusal layer screens a request *before* generation, blocking prompts that mention hazardous agents or match known toxin motifs and logging the attempt for review.

```python
import re

HAZARDOUS_KEYWORDS = [
    "botulinum", "ricin", "anthrax", "smallpox", "ebola",
    "neurotoxin", "hemorrhagic", "binary toxin", "select agent",
]

HAZARDOUS_MOTIFS = [
    re.compile(r"C.C.{10,}C", re.IGNORECASE),   # placeholder toxin-fold pattern
    # Add curated patterns from a screening database.
]


def safe_generate(prompt, model):
    """Refuse hazardous requests; otherwise delegate to the model."""
    prompt_lower = prompt.lower()
    for kw in HAZARDOUS_KEYWORDS:
        if kw in prompt_lower:
            return ("Refused: request mentions a hazardous agent. "
                    "Please contact the institutional biosafety committee.")

    seq_match = re.search(r"[ACDEFGHIKLMNPQRSTVWY]{10,}", prompt.upper())
    if seq_match:
        seq = seq_match.group()
        for motif in HAZARDOUS_MOTIFS:
            if motif.search(seq):
                return ("Refused: requested sequence matches a known hazardous "
                        "motif. Logging this request for review.")

    return model.generate(prompt)
```

**Logging:** Record every refused request with a timestamp, the authenticated user ID (if available), and the prompt. Audit the log regularly.

**Pitfall:** A refusal layer is a deterrent, not a guarantee — a determined actor can rephrase prompts or switch models. It must sit inside a broader posture of access control, usage limits, and human review.

## 19.6  Ecological and non-human ethics

- **Released organisms** (engineered microbes, gene drives) demand reversibility plans and ecosystem modeling (Chapter 14).
- **Animal welfare** in AI-driven experiments: markerless methods reduce harm but do not remove the duty of IACUC review (Chapter 12).
- **Indigenous data sovereignty** (CARE principles) governs biodiversity and genomic data from specific communities.

## 19.7  Pitfalls

- **Fairness theater.** Reporting one parity metric (e.g., demographic parity) while a more clinically relevant metric (e.g., calibration by race) goes unmeasured. A model that equalizes false-positive rates can still have wildly different false-negative rates, so patients in one group are missed more often. Always publish a **fairness dashboard** of multiple metrics and discuss the trade-offs.
- **Consent laundering.** Treating public availability as consent for model training. Participants in a public dataset (e.g., the 1000 Genomes Project) did not consent to foundation-model training or commercial reuse. Best practice: contact the data access committee, negotiate a data use agreement, and respect restrictions (no commercial use, no re-identification). **Remedy:** add a "Consent and Data Use" section to every model card, naming the original consent scope and any deviations.
- **"Open" as a default.** Openness is a virtue for reproducibility and a hazard for dual-use; the right answer is capability-dependent.

## 19.8  Exercises

1. **Impossibility in practice.** Take a clinical dataset with unequal base rates. Show empirically that you cannot simultaneously satisfy calibration and equalized odds.
2. **DP cost curve.** Train a classifier with DP-SGD at several `ε` budgets. Plot accuracy vs. privacy.
3. **Re-identification.** Demonstrate (on synthetic data) how a 30-SNP fingerprint uniquely indexes individuals in a cohort.
4. **Dual-use rubric.** Write a one-page threat model for a hypothetical enzyme-design model and recommend a release strategy.
5. **DP-SGD for a genomic model.** Apply the `opacus` example to a small genomic dataset (e.g., 10,000 individuals, 1,000 SNPs). Sweep the noise multiplier from 0.1 to 5.0 and plot test AUROC vs. `ε`. At what `ε` does utility drop below a usable threshold (e.g., AUROC < 0.7)? Is that acceptable for a clinical application?
6. **Fairness trade-off simulation.** Simulate a clinical risk model with base rates 5% and 20% for two groups, perfectly calibrated in each. Pick a decision threshold and compute FPR and FNR per group. Then adjust one group's threshold to equalize FPR — what happens to calibration? Which approach is preferable for a cancer screening test? For a pretrial risk assessment?
7. **Refusal layer for a protein-design API.** Build a small Flask or FastAPI app that returns a (mock or real) protein sequence and implements keyword and motif screening. Test benign prompts ("design a binder to IL-7R") and harmful prompts ("design a toxin similar to botulinum"), and log the attempts.
8. **Consent laundering audit.** Find a public genomic dataset (e.g., from GEO), read its original consent language, and decide whether it permits training commercial AI models. If not, draft a data use agreement that would allow such training while respecting participant intent.

## 19.9  Further reading

- Kleinberg, J. *Inherent trade-offs in the fair determination of risk scores.* ITCS (2017).
- Chouldechova, A. *Fair prediction with disparate impact.* Big Data (2017).
- Char, D. S. *Implementing machine learning in health care.* NEJM (2018).
- Carroll, S. R. *The CARE principles for Indigenous data governance.* Data Sci. J. (2020).
- Abadi, M. et al. *Deep learning with differential privacy.* CCS (2016) — the original DP-SGD paper.
- Hardt, M. et al. *Equality of opportunity in supervised learning.* NeurIPS (2016) — formal fairness definitions.
- Bender, E. M. et al. *On the dangers of stochastic parrots: can language models be too big?* FAccT (2021) — relevant to foundation models in biology.
- National Academies of Sciences, Engineering, and Medicine. *Biosecurity in the age of AI.* Report (2024) — comprehensive recommendations.

## See also

- [Chapter 16 — Medicine & Healthcare](chapter_16_medicine.md)
- [Chapter 17 — Biotechnology & Bioengineering](chapter_17_biotech.md)
- [Chapter 20 — Policy & Regulation](chapter_20_policy.md)
