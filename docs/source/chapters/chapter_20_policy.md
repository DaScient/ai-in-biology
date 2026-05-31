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

### 20.1a  Decision tree for navigating multiple regimes

Use the following decision tree to determine which regulations apply to an AI-in-biology project.

```text
Is the model intended for clinical use (diagnosis, prognosis, treatment)?
    │
    ├─ Yes → FDA (US) or CE mark (EU) as a medical device (SaMD).
    │         Determine risk class: Class I (low), II (moderate), III (high).
    │         If the model learns post-deployment → Predetermined Change Control Plan (PCCP).
    │
    └─ No → Is the model using human data (genomic, clinical, imaging)?
            │
            ├─ Yes → Data protection laws apply:
            │         • GDPR (EU) if data from EU residents or processing in EU.
            │         • HIPAA (US) if covered entity or business associate.
            │         • Informed consent required; re-identification risk must be managed.
            │
            └─ No → Is the model designing or synthesizing DNA/RNA/proteins?
                    │
                    ├─ Yes → Biosafety regulations (NIH Guidelines, Cartagena Protocol).
                    │         If dual-use potential → Institutional Biosafety Committee (IBC) review.
                    │         If select agents involved → Federal Select Agent Program (US).
                    │
                    └─ No → Non-clinical, non-human, non-dual-use: minimal regulation.
                              But still follow ethical best practices (Chapter 19).
```

**Example application.** A model that predicts sepsis from EHR data and recommends antibiotic dosing:

- **Clinical use** → FDA Class II (moderate risk) because it makes a treatment recommendation.
- **Human data** → HIPAA (if US) and possibly GDPR if EU patients.
- **No DNA synthesis** → no biosafety.

Thus, the regulatory burden is high.

**Pitfall.** Many researchers mistakenly believe that "research use only" labeling exempts them from all regulation. If the model is deployed in a clinical workflow (even as decision support), it is a medical device under FDA guidance.

## 20.2  Software as a Medical Device (SaMD)

A model that informs diagnosis or treatment is usually a *device*. The path:

1. **Classify risk** — by the seriousness of the condition and the role of the output (informing vs. driving).
2. **Choose a pathway** — substantial-equivalence (510(k)), De Novo (novel low/moderate risk), or PMA (high risk).
3. **Assemble evidence** — analytical validation, clinical validation, human-factors testing.
4. **Quality system** — ISO 13485, risk management (ISO 14971), software lifecycle (IEC 62304).

### 20.2a  510(k) submission checklist for AI/ML SaMD

A practical checklist for preparing a 510(k) submission for an AI/ML-enabled SaMD, based on FDA guidance documents (2021–2024).

1. **Device description**
   - Intended use statement (population, setting, decision type).
   - Device functionality (input, output, user interface).
   - Architecture diagram (data flow, model type, retraining plan if any).

2. **Predicate device identification**
   - Identify a legally marketed predicate device with similar intended use and technology.
   - Provide a side-by-side comparison of technological characteristics.

3. **Software documentation**
   - Software requirements specification (SRS).
   - Software design specification (SDS).
   - Software development lifecycle plan (based on IEC 62304).
   - Cybersecurity risk assessment (data at rest, in transit, access control).

4. **Verification and validation**
   - **Analytical validation:** model performance on a reference dataset (sensitivity, specificity, AUC, calibration). Report subgroup performance (age, sex, race, comorbidity).
   - **Clinical validation:** performance on a prospective or retrospective clinical study. Must match the intended-use population.
   - **Human-factors validation:** usability testing with intended users (e.g., clinicians).

5. **Algorithm Change Protocol (ACP) — if the model learns post-deployment**
   - Specify what can change (retraining data, hyperparameters, architecture).
   - Specify triggers for retraining (e.g., drift detection).
   - Specify the validation procedure for each new version.
   - Define performance guardrails and a rollback procedure.

6. **Risk management (ISO 14971)**
   - Hazard identification (e.g., false negative leading to a missed diagnosis).
   - Risk mitigation (e.g., human override, confidence threshold).
   - Residual risk assessment.

7. **Labeling**
   - For clinicians: indications, contraindications, warnings (e.g., "Not validated for pediatric use").
   - For patients (if patient-facing): explanation of AI use, limitations, how to get help.

**Pitfall.** Many submissions are rejected because the predicate device is not sufficiently similar. For novel AI/ML devices, the De Novo pathway (for low-to-moderate risk devices without a predicate) is often more appropriate than 510(k).

## 20.3  Governing models that keep learning

Static approval clashes with continual learning. The accepted instrument is a **Predetermined Change Control Plan (PCCP)**: the manufacturer specifies, *in advance*, what the model may change (data, retraining triggers, performance bounds) and how change is verified.

```text
PCCP (sketch)
- SaMD Pre-Specifications (SPS): which inputs/outputs/architecture may change
- Algorithm Change Protocol (ACP): data management, retraining, validation, update
- Performance guardrails: rollback if subgroup AUROC drops > 0.03
```

This converts "the model improved silently" from a liability into a documented, auditable process.

### 20.3a  A detailed PCCP template

A ready-to-adapt PCCP template based on FDA's 2023 final guidance.

**Predetermined Change Control Plan for [Device Name]**

**1. SaMD Pre-Specifications (SPS)**

| Aspect | Permitted changes | Prohibited changes |
|--------|-------------------|--------------------|
| Input data types | Chest X-ray (DICOM), any vendor, any scanner model | Adding MRI or CT inputs |
| Output format | Probability (0–1) and binary prediction at threshold 0.5 | Changing to multi-class without new validation |
| Model architecture | DenseNet-121 (last layer retrainable); embedding dimension 1024 | Changing to Transformer; changing embedding dimension > 2048 |
| Performance metric | AUROC, sensitivity, specificity at fixed threshold | Changing metric definition |
| Population | Adults ≥ 18 years, emergency department | Adding pediatric or outpatient |

**2. Algorithm Change Protocol (ACP)**

- **Retraining trigger:** when cumulative new data adds ≥ 500 positive cases (confirmed by culture).
- **Data management:** new data must follow the same acquisition protocol, de-identified. Training data excludes patients in the validation set.
- **Validation:** 5-fold cross-validation by hospital site. Hold out the most recent 3 months as a test set.
- **Acceptance criteria:** the new model must have:
  - AUROC ≥ 0.90 (lower bound of 95% CI ≥ 0.88).
  - Subgroup AUROC (by hospital) ≥ 0.85 for all sites.
  - Calibration slope 0.9–1.1, intercept –0.1 to 0.1.
- **Rollback rule:** if any subgroup AUROC drops by > 0.05 compared to the previous version, automatically roll back to the last approved version. Alert the clinical safety officer.

**3. Change documentation**

- Each version is logged with: date, training-data version, code commit hash, validation metrics, approval status.
- Audit trail accessible to the FDA within 48 hours.

**4. Post-market monitoring**

- Weekly performance dashboard: AUROC, calibration, subgroup performance.
- If performance drift triggers a pre-defined threshold, alert for potential retraining or rollback.

**Pitfall.** A PCCP must be submitted **before** any change is implemented. You cannot retroactively apply a PCCP. Plan early.

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

### 20.4a  Extending the model card with validation sections

For a regulatory submission, extend the card with explicit validation, regulatory-status, and data-governance sections.

```python
from dataclasses import dataclass

@dataclass
class RegulatoryModelCard:
    # Basic info
    name: str
    version: str
    date: str

    # Intended use
    intended_use: str
    population: str
    setting: str
    contraindications: list

    # Performance (analytical validation)
    analytical_validation: dict  # e.g., {'AUROC': 0.91, '95% CI': [0.89, 0.93]}
    subgroup_performance: dict   # e.g., {'by_sex': {'F': 0.90, 'M': 0.92}}
    calibration: dict            # e.g., {'Brier': 0.08, 'slope': 0.95}

    # Clinical validation (if available)
    clinical_validation: dict    # e.g., {'prospective_study_AUC': 0.88}

    # Limitations
    known_limitations: list
    failure_modes: list          # e.g., 'highly saturated images'

    # Regulatory status
    regulatory_status: str       # '510(k) pending', 'De Novo granted', 'research only'
    predicate_device: str        # if 510(k)
    pccp_attached: bool

    # Consent and data governance
    training_data_source: str
    consent_scope: str
    data_use_agreement: str      # URL or reference

    def validate(self):
        """Ensure required fields for submission."""
        required = ['intended_use', 'population', 'analytical_validation', 'known_limitations']
        for field in required:
            if not getattr(self, field, None):
                raise ValueError(f"Missing required field: {field}")
        # Check subgroup performance not empty
        if not self.subgroup_performance:
            raise ValueError("Subgroup performance must be reported")
        return True
```

**Use.** This card travels with the model weights. For an FDA submission, expand it to a full document with narrative, but the structured fields ensure completeness.

**Pitfall.** A model card is not a substitute for a 510(k) submission. It is a summary for transparency; regulators require detailed technical files.

## 20.5  Data governance across borders

- **Lawful basis.** GDPR requires an explicit basis (consent, public interest, research exemption) for processing health data.
- **Cross-border transfer.** Genomic datasets are often sovereignty-restricted; federated analysis can keep data in-country.
- **Secondary use.** Reusing clinical data to train a model usually needs IRB review and a documented basis distinct from the original care.

### 20.5a  Federated learning for cross-border genomic data

Suppose three sites (US, UK, Germany) each hold genotype and phenotype data that cannot leave the country. We can train a shared disease-risk model without centralizing data using federated learning (here with the `flwr` / Flower framework).

```python
# Client code (runs at each site)
import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset

class GenomicClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param).to(self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for epoch in range(5):  # local epochs
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(x_batch).squeeze()
                loss = torch.nn.BCELoss()(output, y_batch)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                output = self.model(x_batch).squeeze()
                pred = (output > 0.5).float()
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)
        accuracy = correct / total
        return accuracy, len(self.test_loader.dataset), {"accuracy": accuracy}

# Server code (central coordinator)
def start_server():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
```

**Regulatory benefit.** No raw genomic data crosses borders; only model updates (gradients) are shared, and these can be made differentially private (Chapter 19). This helps satisfy GDPR Article 46 (appropriate safeguards) without needing explicit cross-border transfer agreements.

**Pitfall.** Even gradients can leak information about training data (gradient-inversion attacks). Add differential privacy on the client side before sending updates.

## 20.6  Pitfalls

- **"Decision support" loophole.** Calling a high-impact model "support" does not exempt it if clinicians cannot realistically override it.
- **Frozen-then-forgotten.** Deploying a static model and never monitoring drift breaches post-market obligations.
- **Compliance ≠ ethics.** A legal model can still be unjust; Chapter 19 obligations persist.

### 20.6a  Extended pitfalls

**Decision-support loophole.** A model that strongly influences a clinician's decision (e.g., by providing a probability of cancer) is still a medical device if the clinician cannot reasonably override it. The FDA has clarified that "clinical decision support" is not an automatic exemption. If the model is intended to replace or substantially alter clinical judgment, it requires regulation. *Mitigation:* if you claim "decision support" to avoid regulation, ensure that the clinician has access to all underlying data, that the model output is easily overridden with a single click, and that the model is not integrated into a closed-loop system (e.g., automatic insulin dosing).

**Frozen-then-forgotten.** Deploying a model and never updating it, despite known performance drift, violates post-market surveillance obligations. The FDA requires manufacturers to monitor real-world performance and take corrective action when needed. *Solution:* implement automated drift detection (see Chapter 16) and a retraining or rollback plan as part of your PCCP.

## 20.7  Exercises

1. **Classify a device.** For three hypothetical models (sepsis alarm, wellness step-counter, tumor-board recommender), assign an FDA risk class and justify.
2. **Draft a PCCP.** Write the SPS + ACP for a retrained-quarterly readmission model, including a rollback rule.
3. **GDPR basis.** For a multi-site EU genomics study, identify the lawful basis and the transfer mechanism.
4. **Model-card lint.** Extend the `ModelCard` above with a validator that fails if subgroup metrics or limitations are empty.
5. **Classify a device under FDA guidance.** For each of the following, assign a risk class (I, II, III) and justify:
   - An app that tracks step count and suggests walking goals.
   - An AI that identifies stroke on CT images and alerts the radiologist.
   - An AI that recommends a chemotherapy regimen based on genomic markers (no human in the loop).
   - A model that predicts 5-year mortality for heart-failure patients to help with advanced care planning.
6. **Draft a PCCP for a readmission model.** Using the template in §20.3a, write a 2-page PCCP for a model that predicts 30-day readmission and triggers a nurse follow-up call. Include SPS, ACP, performance guardrails, and a rollback rule.
7. **Federated learning simulation.** Use the Flower framework to simulate three clients with synthetic genomic data (e.g., random SNPs + a linear phenotype). Run federated averaging for 10 rounds. Compare the global model's accuracy to a model trained on pooled data (centralized). How much accuracy is lost due to federated learning? Add differential privacy and measure the additional loss.
8. **Model card for a real model.** Take a published AI model in biology (e.g., from HuggingFace or a recent paper). Create a regulatory model card following the extended dataclass in §20.4a. Identify missing information that would be required for an FDA submission, and propose how to obtain it.

## 20.8  Further reading

- US FDA. *Marketing Submission Recommendations for a Predetermined Change Control Plan* (2023).
- European Parliament. *Regulation on Artificial Intelligence (AI Act)* (2024).
- Gerke, S. *The need for a system view to regulate AI/ML-based SaMD.* npj Digit. Med. (2020).
- Mitchell, M. *Model cards for model reporting.* FAT* (2019).
- US FDA. *Marketing Submission Recommendations for a Predetermined Change Control Plan for Machine Learning-Enabled Device Software Functions* (final guidance, 2023).
- European Parliament. *EU AI Act: full text and implications for medical devices* (2024).
- UK MHRA. *Guidance on AI as a Medical Device* (2024).

## See also

- [Chapter 16 — Medicine & Healthcare](chapter_16_medicine.md)
- [Chapter 19 — Ethics of AI in Biology](chapter_19_ethics.md)
- [Chapter 21 — Societal Transformation](chapter_21_society.md)
