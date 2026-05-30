# Chapter 20 — Policy & Regulation

> *"Regulation is the slow encoding of a society's risk tolerance into rules; AI in biology is moving faster than that encoding."*

## Learning objectives

- Summarize the major regulatory regimes that govern biological AI: medical-device law, data-protection law, and biosafety oversight.
- Trace a model from research artifact to regulated product (SaMD) and identify the evidence required at each gate.
- Explain how "learning" systems are governed despite changing after approval.
- Build a compliance-aware documentation checklist that travels with a model.

## 20.1  The regulatory landscape

| Domain | Representative framework | Scope |
|--------|--------------------------|-------|
| Medical AI (US) | FDA SaMD guidance; 510(k) / De Novo / PMA | Clinical safety & effectiveness |
| Medical AI (EU) | MDR 2017/745 + AI Act (high-risk) | Devices + horizontal AI rules |
| Data protection | GDPR (EU), HIPAA (US) | Personal & health data |
| Biosafety | Cartagena Protocol; national IBCs; DURC policy | GMOs, dual-use research |
| Genetic data | GINA (US) | Non-discrimination |

No single regime covers a clinical genomics model; **most real systems sit at the intersection of three or four.**

## 20.2  Software as a Medical Device (SaMD)

A model that informs diagnosis or treatment is usually a *device*. The path:

1. **Classify risk** — by the seriousness of the condition and the role of the output (informing vs. driving).
2. **Choose a pathway** — substantial-equivalence (510(k)), De Novo (novel low/moderate risk), or PMA (high risk).
3. **Assemble evidence** — analytical validation, clinical validation, human-factors testing.
4. **Quality system** — ISO 13485, risk management (ISO 14971), software lifecycle (IEC 62304).

## 20.3  Governing models that keep learning

Static approval clashes with continual learning. The accepted instrument is a **Predetermined Change Control Plan (PCCP)**: the manufacturer specifies, *in advance*, what the model may change (data, retraining triggers, performance bounds) and how change is verified.

```text
PCCP (sketch)
- SaMD Pre-Specifications (SPS): which inputs/outputs/architecture may change
- Algorithm Change Protocol (ACP): data management, retraining, validation, update
- Performance guardrails: rollback if subgroup AUROC drops > 0.03
```

This converts "the model improved silently" from a liability into a documented, auditable process.

## 20.4  Worked example — a model card as a compliance artifact

```python
from dataclasses import dataclass, field, asdict
import json

@dataclass
class ModelCard:
    name: str
    intended_use: str
    population: str
    training_data: str
    metrics: dict
    subgroup_metrics: dict
    known_limitations: list = field(default_factory=list)
    regulatory_status: str = "research use only"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

card = ModelCard(
    name="CXR-Pneumonia-v1",
    intended_use="Triage support for adult chest radiographs",
    population="Adults >=18; ED and inpatient",
    training_data="MIMIC-CXR + CheXpert (de-identified)",
    metrics={"AUROC": 0.91, "Brier": 0.08},
    subgroup_metrics={"by_sex": {"F": 0.90, "M": 0.92}},
    known_limitations=["Not validated pediatric", "Scanner shift untested"],
)
print(card.to_json())
```

A machine-readable model card makes intended use, evaluation, and limitations *travel with the weights* — the backbone of an auditable submission.

## 20.5  Data governance across borders

- **Lawful basis.** GDPR requires an explicit basis (consent, public interest, research exemption) for processing health data.
- **Cross-border transfer.** Genomic datasets are often sovereignty-restricted; federated analysis can keep data in-country.
- **Secondary use.** Reusing clinical data to train a model usually needs IRB review and a documented basis distinct from the original care.

## 20.6  Pitfalls

- **"Decision support" loophole.** Calling a high-impact model "support" does not exempt it if clinicians cannot realistically override it.
- **Frozen-then-forgotten.** Deploying a static model and never monitoring drift breaches post-market obligations.
- **Compliance ≠ ethics.** A legal model can still be unjust; Chapter 19 obligations persist.

## 20.7  Exercises

1. **Classify a device.** For three hypothetical models (sepsis alarm, wellness step-counter, tumor-board recommender), assign an FDA risk class and justify.
2. **Draft a PCCP.** Write the SPS + ACP for a retrained-quarterly readmission model, including a rollback rule.
3. **GDPR basis.** For a multi-site EU genomics study, identify the lawful basis and the transfer mechanism.
4. **Model-card lint.** Extend the `ModelCard` above with a validator that fails if subgroup metrics or limitations are empty.

## 20.8  Further reading

- US FDA. *Marketing Submission Recommendations for a Predetermined Change Control Plan* (2023).
- European Parliament. *Regulation on Artificial Intelligence (AI Act)* (2024).
- Gerke, S. *The need for a system view to regulate AI/ML-based SaMD.* npj Digit. Med. (2020).
- Mitchell, M. *Model cards for model reporting.* FAT* (2019).

## See also

- [Chapter 16 — Medicine & Healthcare](chapter_16_medicine.md)
- [Chapter 19 — Ethics of AI in Biology](chapter_19_ethics.md)
- [Chapter 21 — Societal Transformation](chapter_21_society.md)
