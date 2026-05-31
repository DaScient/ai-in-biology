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

### 16.1a  Three classes of clinical AI — extended with regulatory examples

The list above is abstract. Grounding each class in **real-world FDA-cleared examples** and their risk classifications clarifies the evidence bar each must clear.

| Class | Example | FDA product code | Risk class | Key evidence required |
|-------|---------|------------------|------------|-----------------------|
| **Diagnostic** | IDx-DR (diabetic retinopathy detection from retinal images) | PIB | Class II (510(k)) | Sensitivity > 85%, specificity > 82% vs. ophthalmologist grading |
| **Diagnostic** | Viz.ai (large vessel occlusion detection from CT) | QAS | Class II (De Novo) | Reduced time-to-treatment (median 48 min reduction) |
| **Prognostic** | HeartFlow FFR-CT (fractional flow reserve from coronary CTA) | QNP | Class II (De Novo) | Agreement with invasive FFR (81–86% accuracy) |
| **Prognostic** | Optellum (lung nodule malignancy prediction) | QJR | Class II (510(k)) | AUC 0.87, calibration curve close to identity |
| **Therapeutic** | IBM Watson for Oncology (withdrawn) | N/A (never approved) | N/A | Failed due to lack of clinical validation |
| **Therapeutic** | DoseMe (vancomycin dosing) | PEC | Class II (510(k)) | Improved therapeutic drug levels (57% vs. 38% target attainment) |

**Key insight:** No therapeutic-recommendation AI has received FDA approval for autonomous treatment decisions. All require clinician oversight. The path from research to deployment is longest for therapeutic systems.

**Pitfall:** Many "AI breakthroughs" reported in press releases are not FDA-approved. Always check the FDA's 510(k) database or De Novo classification list before citing a product as clinically available.

## 16.2  Foundation models in medicine

| Modality | Notable model | Pretraining |
|----------|---------------|-------------|
| Chest X-ray | CXR-Foundation, RAD-DINO | 1–5 M images |
| Pathology | Path-Foundation, UNI, Virchow | 100 k–1 M slides |
| EHR text | Med-PaLM 2, GatorTron, Clinical-Camel | 90 B–500 B tokens |
| ECG | ECG-Foundation, HeartBEiT | 5–10 M recordings |
| Multimodal | LLaVA-Med, BioMedCLIP | Image + text pairs |

Common pattern: linear probe + small adapter → near-supervised performance with 50–500 labels.

### 16.2a  Foundation models in medicine — decision matrix and practical advice

When several foundation models target the same modality, a **decision matrix** comparing size, pre-training data, cost, and licensing helps narrow the choice.

| Model | Modality | Size | Pre-training data | Best for | Inference cost | Open weights? |
|-------|----------|------|-------------------|----------|----------------|---------------|
| **RAD-DINO** | Chest X-ray | 100M | CheXpert (224k images) + MIMIC-CXR | Zero-shot abnormality detection | Low (runs on single GPU) | No (requires license) |
| **CXR-Foundation** | Chest X-ray | 50M | CheXpert + PadChest + MIMIC | Fine-tuning for rare findings | Low | Yes (HuggingFace) |
| **Path-Foundation** | Histopathology (whole-slide) | 300M | TCGA (28k slides) + external data | Cancer subtyping, biomarker prediction | High (needs GPU cluster) | Limited |
| **Med-PaLM 2** | Clinical text (QA) | 340B | PubMed, clinical notes, MIMIC-III | Question answering, discharge summary generation | Very high (TPU required) | No (API only) |
| **ClinicalBERT** | Clinical text | 110M | MIMIC-III notes | Predict mortality, readmission | Low | Yes |
| **GatorTron** | Clinical text | 8.9B | >90M clinical notes | Complex NLP (information extraction) | High | Yes (restricted) |

**Recommendation:**

- For **chest X-ray**, start with CXR-Foundation (open weights, good performance).
- For **histopathology**, Path-Foundation is powerful but requires significant compute; consider a smaller model (e.g., ResNet-50 pre-trained on ImageNet + fine-tuned on TCGA) as a baseline.
- For **clinical text**, ClinicalBERT is sufficient for most tasks (readmission, phenotype extraction). Use Med-PaLM 2 only for complex QA requiring medical reasoning (and only if you have API access).

**Pitfall:** Foundation models pre-trained on US/European data may perform poorly on other populations. Always validate on local data before deployment.

## 16.3  Evaluation — beyond AUC

A clinical evaluation report should include:

1. **Discrimination** (AUROC, AUPRC) — overall and per-subgroup (sex, race, age, site, scanner).
2. **Calibration** (Brier score, calibration plot) — well-calibrated models support shared decision-making.
3. **Decision-curve analysis** (Vickers) — net benefit at varying clinical thresholds.
4. **Robustness** — performance under common shifts: contrast injection, year, scanner manufacturer.
5. **Clinically meaningful endpoints** — not "the model would have caught it" but "patients lived longer / had fewer complications".

### 16.3a  Evaluation — beyond AUC: decision curve analysis

**What DCA does:** Decision curve analysis (DCA) measures the *net benefit* of using a model at different probability thresholds, accounting for the relative harm of false positives vs. false negatives.

**Net benefit formula:**

\[
\text{Net Benefit} = \frac{\text{True Positives}}{n} - \frac{\text{False Positives}}{n} \times \left( \frac{p}{1-p} \right)
\]

where \(p\) is the threshold probability at which you would intervene (e.g., treat a patient if predicted risk \(>p\)). The term \(p/(1-p)\) is the odds at that threshold.

**Implementation:**

```python
import numpy as np
from sklearn.metrics import confusion_matrix

def decision_curve(y_true, y_pred_prob, thresholds=np.linspace(0.01, 0.99, 50)):
    """
    Compute net benefit for a range of thresholds.
    Returns thresholds, net_benefit_model, net_benefit_treat_all, net_benefit_treat_none.
    """
    n = len(y_true)
    nb_model = []
    nb_treat_all = []
    nb_treat_none = []

    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Net benefit for model
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        nb_model.append(nb)

        # Treat all (if you treated every patient): every positive is a TP,
        # every negative is a FP.
        nb_all = (np.sum(y_true) / n) - ((n - np.sum(y_true)) / n) * (t / (1 - t))
        nb_treat_all.append(nb_all)

        # Treat none (net benefit = 0 by definition)
        nb_treat_none.append(0.0)

    return thresholds, nb_model, nb_treat_all, nb_treat_none

# Example usage
y_true = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0])
y_pred_prob = np.array([0.1, 0.3, 0.8, 0.2, 0.7, 0.9, 0.15, 0.85, 0.25, 0.05])
thresholds, nb_model, nb_treat_all, _ = decision_curve(y_true, y_pred_prob)

# Plot
import matplotlib.pyplot as plt
plt.plot(thresholds, nb_model, label='Model', linewidth=2)
plt.plot(thresholds, nb_treat_all, label='Treat all', linestyle='--', color='gray')
plt.axhline(y=0, color='black', linestyle=':', label='Treat none')
plt.xlabel('Threshold probability')
plt.ylabel('Net benefit')
plt.legend()
plt.title('Decision Curve Analysis')
```

**Interpretation:**

- The model is clinically useful where its net benefit exceeds both "treat all" and "treat none".
- Typical clinical thresholds: 5% (screening), 15% (outpatient treatment), 30% (invasive procedure), 50% (surgery).
- If the model's curve never exceeds "treat all", it provides no benefit over treating everyone.

**Pitfall:** DCA requires specifying the threshold range relevant to the clinical context. For low-probability events (e.g., rare cancer), use a log-scaled threshold axis.

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

### 16.4a  Worked example extension — survival with competing risks

The Cox model above treats death from other causes as censoring. For many medical problems, multiple causes of death **compete** (e.g., cancer-specific death vs. cardiovascular death). The Fine-Gray model extends the analysis to competing risks.

**Why competing risks matter:** Standard Cox models treat death from other causes as censoring, which overestimates the probability of the event of interest. Fine-Gray models the cumulative incidence function (CIF) directly.

**Implementation using `lifelines`:**

```python
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# Assume we have:
# X: imaging features (after CNN encoder)
# T: time to event (months)
# E: event indicator (0 = censored, 1 = cancer death, 2 = cardiovascular death, 3 = other)

# Fit Fine-Gray model for cause 1 (cancer death)
# Note: lifelines does not natively support Fine-Gray; use `competing_risks` module or R
# Here we use a simpler approach: treat competing events as censoring (conservative)

# For demonstration, we use Cox but adjust interpretation
model = CoxPHFitter()
df = pd.DataFrame(X)
df['time'] = T
df['event'] = (E == 1).astype(int)  # treat non-cancer death as censored

model.fit(df, duration_col='time', event_col='event')
c_index = concordance_index(df['time'], -model.predict_partial_hazard(df), df['event'])
print(f"Concordance index (cancer death): {c_index:.3f}")

# For proper competing risks, use R's `cmprsk` package via rpy2
# or the `scikit-survival` library with Fine-Gray implementation.
```

**For proper implementation using `scikit-survival`:**

```python
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.preprocessing import OneHotEncoder

# Format data as structured array
dtype = [('status', bool), ('time', float)]
y = np.array([(True, T[i]) if E[i] == 1 else (False, T[i]) for i in range(len(T))], dtype=dtype)

# Fit Cox model
estimator = CoxPHSurvivalAnalysis()
estimator.fit(X, y)
c_index = concordance_index_censored(y['status'], y['time'], -estimator.predict(X))[0]
```

**Pitfall:** The Fine-Gray model assumes proportional subdistribution hazards, which may not hold. Always check Schoenfeld residuals for the cause-specific model.

## 16.5  Regulatory and deployment path

1. **Intended use statement** — population, environment, decision impact.
2. **Risk management** (ISO 14971), software lifecycle (IEC 62304).
3. **Algorithm change protocol** for models that learn over time (FDA "predetermined change control plan").
4. **Real-world performance monitoring** — drift dashboards, fairness alarms, manual sample audits.

### 16.5a  Regulatory and deployment path — PCCP template

A Predetermined Change Control Plan (PCCP) lets a team pre-authorize specific model changes with the FDA. Below is a **concrete template** that a team can adapt for submission.

**PCCP document outline (for a model that retrains quarterly):**

1. **SaMD Pre-Specifications (SPS)**
   - **Input data types:** Chest X-ray (DICOM), format: CXR, pixel spacing 0.1–0.5 mm.
   - **Outputs:** Probability of pneumonia (0–1), bounding boxes if present.
   - **Architecture:** DenseNet-121 with last layer retrained; embedding dimension 1024.
   - **Performance thresholds:** AUROC ≥ 0.90 (95% CI lower bound ≥ 0.88) on validation set.

2. **Algorithm Change Protocol (ACP)**
   - **Data update:** New MIMIC-CXR patients added quarterly; excludes patients already used.
   - **Retraining trigger:** When new data adds ≥ 500 positive pneumonia cases.
   - **Validation procedure:** 5-fold spatial CV (by hospital site), hold out most recent 3 months as test.
   - **Performance guardrails:**
     - **Rollback threshold:** If subgroup AUROC (by hospital) drops by > 0.05 compared to previous version.
     - **Manual review:** Any new version with AUROC < 0.88 requires re-audit by clinical committee.
   - **Human-in-the-loop:** High-confidence predictions (>0.95) can auto-document; uncertain cases (0.3–0.7) require radiologist review.

3. **Change logging**
   - Each version recorded: training data version, date, performance metrics, code commit hash.
   - Audit trail accessible to FDA upon request.

4. **Post-market monitoring**
   - Real-world performance dashboard updated weekly.
   - If real-world AUROC drops below 0.85 for more than 10 consecutive days, alert to clinical safety officer.

**Pitfall:** A PCCP is not a waiver from clinical validation. The initial submission must still include robust evidence. The PCCP only governs *changes* after approval.

## 16.6  Pitfalls

- **Shortcut learning.** A model that "diagnoses" pneumonia from text in radiographs (`PORT` markers, scanner artifacts) is hazardous.
- **Spectrum bias.** Models trained on tertiary-care cohorts often fail in primary care.
- **Label noise.** "Pathologist truth" itself has 10–30 % inter-rater disagreement on many tasks; treat as upper bound.
- **Equity.** Disparities in the training data become disparities in care. Stratified reporting is mandatory.

### 16.6a  Extended pitfalls — shortcut learning and spectrum bias

**Shortcut learning example:** A pneumonia model trained on MIMIC-CXR learned to detect the "portable X-ray" text overlay (present in sicker patients) rather than radiographic signs. When tested on non-portable images, performance collapsed from AUROC 0.92 to 0.58.

**Detection method:** Train a classifier to predict image acquisition parameters (portable vs. non-portable, contrast injection, patient orientation) from the image alone. If AUROC > 0.85, the model may be using shortcuts.

**Mitigation:**

- During training, balance the dataset by acquisition parameter.
- Use adversarial training to remove shortcut features.
- Test on a held-out dataset from a different hospital with different protocols.

**Spectrum bias:** A model developed on tertiary-care cancer center data (high disease prevalence, severe cases) fails in primary care (low prevalence, mild cases). Fix by:

- Collecting data from multiple settings (primary, secondary, tertiary).
- Oversampling mild cases during training.
- Reporting performance stratified by disease severity.

**Pitfall:** Spectrum bias is often missed because researchers test on data from the same institution (similar spectrum). Always test on external data with different case mix.

## 16.7  Exercises

1. **Calibration.** Take a public CXR pneumonia classifier; plot its reliability diagram on PadChest. Fit Platt and isotonic recalibrators; compare Brier scores.
2. **Subgroup audit.** On MIMIC-CXR, compare model AUROC across self-reported race / sex / age strata. Document any disparity > 0.05.
3. **Decision curve.** For a sepsis early-warning model, draw the net-benefit curve. At which thresholds is the model useful?
4. **Drift drill.** Simulate scanner-replacement drift (intensity histogram shift). Quantify performance drop and design a monitoring alarm.
5. **Decision curve analysis for a sepsis model.** Use a public dataset (e.g., MIMIC-IV) to predict sepsis within 6 hours. Train a simple model (e.g., XGBoost on vital signs). Plot the decision curve. At what threshold is the model superior to "treat all" (assume treating all suspected sepsis has high antibiotic resistance risk)? What is the net benefit at that threshold?
6. **Fine-Gray vs. Cox for cancer survival.** Use a public cancer dataset with multiple causes of death (e.g., SEER). Fit both a Cox model (treating other causes as censored) and a Fine-Gray model. Plot the predicted cumulative incidence of cancer death at 5 years for a high-risk patient. How different are the estimates?
7. **Shortcut detection in chest X-ray.** Train a DenseNet on CheXpert to predict "No Finding". Train a separate classifier to predict image orientation (AP vs. PA) from the same images. Correlate the two predictions. Does the "No Finding" model perform worse on AP images? If so, it may be using orientation as a shortcut.
8. **Draft a PCCP for a readmission model.** Take a published model for 30-day readmission prediction. Write a 2-page PCCP covering SPS, ACP, and performance guardrails. Include a rollback rule and a human-in-the-loop decision point.

## 16.8  Further reading

- Rajpurkar, P. *AI in health and medicine.* Nat Med (2022).
- McKinney, S. *International evaluation of an AI system for breast cancer screening.* Nature (2020).
- Singhal, K. *Med-PaLM 2.* Nat Med (2023).
- Wiens, J. *Do no harm: a roadmap for responsible ML in healthcare.* Nat Med (2019).
- FDA. *Marketing Submission Recommendations for a Predetermined Change Control Plan for Machine Learning-Enabled Device Software Functions.* (2023) — official guidance document.
- Vickers, A. J. et al. *Decision curve analysis: a practical guide for clinicians.* J Clin Oncol (2019) — step-by-step tutorial.
- Oakden-Rayner, L. et al. *Hidden stratification in clinical AI models.* JAMA Netw Open (2020) — spectrum bias case studies.
- Zech, J. R. et al. *Confounding variables can degrade generalization performance of radiology deep learning models.* Nat Mach Intell (2018) — shortcut learning examples.

## See also

- [Chapter 8 — Protein Structure & Design](chapter_08_protein.md)
- [Clinical API](../api/clinical.md)
- [Chapter 19 — Ethics of AI in Biology](chapter_19_ethics.md)
