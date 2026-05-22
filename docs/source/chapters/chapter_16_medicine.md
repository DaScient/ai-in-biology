# Chapter 16 — Medicine & Healthcare

> *"A clinical AI model that cannot tell you why it is uncertain is not yet a medical instrument."*

## Learning objectives

- Differentiate the three main classes of medical AI: diagnostic, prognostic, and therapeutic-recommendation systems.
- Apply foundation models for medical imaging (RAD-DINO, CXR-Foundation, Path-Foundation) and clinical text (Med-PaLM 2, ClinicalBERT, GatorTron).
- Quantify and report subgroup performance, calibration, and decision-curve analysis.
- Trace the regulatory and reimbursement path from a research model to a deployable Software-as-a-Medical-Device (SaMD).

## 16.1  Three classes of clinical AI

| Class | Question | Example |
|-------|----------|---------|
| Diagnostic | What is going on now? | Diabetic retinopathy grading |
| Prognostic | What will happen, when? | 30-day readmission, survival |
| Therapeutic | What should we do? | Antibiotic stewardship, ventilator settings |

The regulatory burden, evidence requirements, and risk of automation bias all increase from top to bottom.

## 16.2  Foundation models in medicine

| Modality | Notable model | Pretraining |
|----------|---------------|-------------|
| Chest X-ray | CXR-Foundation, RAD-DINO | 1–5 M images |
| Pathology | Path-Foundation, UNI, Virchow | 100 k–1 M slides |
| EHR text | Med-PaLM 2, GatorTron, Clinical-Camel | 90 B–500 B tokens |
| ECG | ECG-Foundation, HeartBEiT | 5–10 M recordings |
| Multimodal | LLaVA-Med, BioMedCLIP | Image + text pairs |

Common pattern: linear probe + small adapter → near-supervised performance with 50–500 labels.

## 16.3  Evaluation — beyond AUC

A clinical evaluation report should include:

1. **Discrimination** (AUROC, AUPRC) — overall and per-subgroup (sex, race, age, site, scanner).
2. **Calibration** (Brier score, calibration plot) — well-calibrated models support shared decision-making.
3. **Decision-curve analysis** (Vickers) — net benefit at varying clinical thresholds.
4. **Robustness** — performance under common shifts: contrast injection, year, scanner manufacturer.
5. **Clinically meaningful endpoints** — not "the model would have caught it" but "patients lived longer / had fewer complications".

## 16.4  Worked example — survival from imaging

```python
import torch
from torch import nn
from torchsurv.loss import cox

class ImagingCox(nn.Module):
    """CNN encoder feeding a single-risk Cox proportional hazards head."""

    def __init__(self, backbone, hidden=256):
        super().__init__()
        self.backbone = backbone           # e.g. RAD-DINO ViT
        self.head = nn.Sequential(nn.Linear(768, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.backbone(x)               # (B, 768)
        return self.head(h).squeeze(-1)    # log-hazard

# Train with partial likelihood from torchsurv
# loss = cox.neg_partial_log_likelihood(risk, event, time)
```

Report concordance (`c-index`) overall and on each demographic stratum.

## 16.5  Regulatory and deployment path

1. **Intended use statement** — population, environment, decision impact.
2. **Risk management** (ISO 14971), software lifecycle (IEC 62304).
3. **Algorithm change protocol** for models that learn over time (FDA "predetermined change control plan").
4. **Real-world performance monitoring** — drift dashboards, fairness alarms, manual sample audits.

## 16.6  Pitfalls

- **Shortcut learning.** A model that "diagnoses" pneumonia from text in radiographs (`PORT` markers, scanner artifacts) is hazardous.
- **Spectrum bias.** Models trained on tertiary-care cohorts often fail in primary care.
- **Label noise.** "Pathologist truth" itself has 10–30 % inter-rater disagreement on many tasks; treat as upper bound.
- **Equity.** Disparities in the training data become disparities in care. Stratified reporting is mandatory.

## 16.7  Exercises

1. **Calibration.** Take a public CXR pneumonia classifier; plot its reliability diagram on PadChest. Fit Platt and isotonic recalibrators; compare Brier scores.
2. **Subgroup audit.** On MIMIC-CXR, compare model AUROC across self-reported race / sex / age strata. Document any disparity > 0.05.
3. **Decision curve.** For a sepsis early-warning model, draw the net-benefit curve. At which thresholds is the model useful?
4. **Drift drill.** Simulate scanner-replacement drift (intensity histogram shift). Quantify performance drop and design a monitoring alarm.

## 16.8  Further reading

- Rajpurkar, P. *AI in health and medicine.* Nat Med (2022).
- McKinney, S. *International evaluation of an AI system for breast cancer screening.* Nature (2020).
- Singhal, K. *Med-PaLM 2.* Nat Med (2023).
- Wiens, J. *Do no harm: a roadmap for responsible ML in healthcare.* Nat Med (2019).

## See also

- [Chapter 8 — Protein Structure & Design](chapter_08_protein.md)
- [Clinical API](../api/clinical.md)
- [Chapter 19 — Ethics of AI in Biology](chapter_19_ethics.md)
