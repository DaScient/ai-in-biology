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

## 19.2  Privacy and consent

- **Re-identification.** Genomic data are quasi-identifiers; even "anonymized" SNP sets can be matched to named individuals via genealogy databases.
- **Consent decay.** Broad consent collected for one study rarely covers training a foundation model years later. Document the chain of consent.
- **Federated and private learning.** Differential privacy (DP-SGD), federated learning, and secure enclaves let models learn without centralizing raw records — at a measurable utility cost.

## 19.3  Fairness is plural and contested

For a clinical classifier with score `s`, sensitive attribute `a`, and label `y`:

- **Demographic parity**: `P(ŷ=1 | a)` equal across groups.
- **Equalized odds**: `P(ŷ=1 | y, a)` equal across groups.
- **Calibration**: `P(y=1 | s, a)` equal across groups.

An impossibility result (Kleinberg et al., 2016; Chouldechova, 2017) shows that, except in degenerate cases, calibration and equalized odds cannot both hold when base rates differ. **You must choose which fairness criterion matches the clinical harm model — there is no universal answer.**

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

## 19.5  Dual-use and responsible disclosure

Generative biology models can lower the barrier to harm. Minimum responsible practice:

1. **Threat-model before training** — could the capability uplift a malicious actor?
2. **Capability evaluations** — red-team the model for hazardous outputs.
3. **Staged release** — gate weights, add refusal layers, log regulated requests (see Chapter 17).
4. **Coordinated disclosure** — notify biosecurity bodies before publishing dangerous methods.

## 19.6  Ecological and non-human ethics

- **Released organisms** (engineered microbes, gene drives) demand reversibility plans and ecosystem modeling (Chapter 14).
- **Animal welfare** in AI-driven experiments: markerless methods reduce harm but do not remove the duty of IACUC review (Chapter 12).
- **Indigenous data sovereignty** (CARE principles) governs biodiversity and genomic data from specific communities.

## 19.7  Pitfalls

- **Fairness theater.** Reporting one parity metric while the clinically relevant harm is governed by another.
- **Consent laundering.** Treating public availability as consent for model training.
- **"Open" as a default.** Openness is a virtue for reproducibility and a hazard for dual-use; the right answer is capability-dependent.

## 19.8  Exercises

1. **Impossibility in practice.** Take a clinical dataset with unequal base rates. Show empirically that you cannot simultaneously satisfy calibration and equalized odds.
2. **DP cost curve.** Train a classifier with DP-SGD at several `ε` budgets. Plot accuracy vs. privacy.
3. **Re-identification.** Demonstrate (on synthetic data) how a 30-SNP fingerprint uniquely indexes individuals in a cohort.
4. **Dual-use rubric.** Write a one-page threat model for a hypothetical enzyme-design model and recommend a release strategy.

## 19.9  Further reading

- Kleinberg, J. *Inherent trade-offs in the fair determination of risk scores.* ITCS (2017).
- Chouldechova, A. *Fair prediction with disparate impact.* Big Data (2017).
- Char, D. S. *Implementing machine learning in health care.* NEJM (2018).
- Carroll, S. R. *The CARE principles for Indigenous data governance.* Data Sci. J. (2020).

## See also

- [Chapter 16 — Medicine & Healthcare](chapter_16_medicine.md)
- [Chapter 17 — Biotechnology & Bioengineering](chapter_17_biotech.md)
- [Chapter 20 — Policy & Regulation](chapter_20_policy.md)
