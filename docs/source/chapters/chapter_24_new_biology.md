# Chapter 24 — A New Biology

> *"The next biology will be neither purely wet nor purely computational, but a single practice in which models and experiments are the same conversation."*

## Learning objectives

- Synthesize the through-line of this textbook: one family of representation, sequence, and graph methods applied across every biological scale.
- Articulate what changes — and what does not — when AI becomes a standard instrument of the life sciences.
- Adopt the working practices of an AI-fluent biologist: reproducibility, calibrated skepticism, and responsibility.
- Chart a personal path for continued learning beyond this book.

## 24.1  The unifying thesis, revisited

Chapter 1 claimed that the *same* methods recur across scales. The journey confirms it:

| Scale | Representation | Sequence/dynamics | Graph |
|-------|----------------|-------------------|-------|
| Genome | DNA LMs (Ch 2–3) | regulatory grammar (Ch 7) | 3-D contacts |
| Protein | PLM embeddings (Ch 8) | folding/recycling | residue contacts |
| Cell | scVI/scGPT (Ch 9) | trajectories (Ch 6, 10) | k-NN / spatial graphs |
| Organism | latent dynamics (Ch 11) | behavioral syllables (Ch 12) | social/connectome graphs |
| Ecosystem | EO embeddings (Ch 15) | early-warning signals (Ch 14) | interaction networks |

**Domain knowledge enters as inductive bias, not as a different toolbox.**

## 24.1a  The unifying thesis, revisited — a quantitative synthesis across scales

Section 24.1 argues that the same methods recur across scales. The table below makes that capstone summary quantitative, mapping each biological scale to the dominant representation and method family used throughout the textbook.

| Scale | Primary representation | Primary method family | Example chapter | Inductive bias |
|-------|------------------------|------------------------|----------------|----------------|
| **Molecular (DNA/RNA)** | Discrete sequence (bases/k-mers) | Language models (masked, autoregressive) | 2, 3 | Locality, complementarity, reading frame |
| **Protein** | Sequence + 3D coordinates | Structure prediction (MSA + Evoformer), diffusion | 8 | Physical constraints (bond angles, sterics) |
| **Cellular** | Gene expression vector (sparse, high-dim) | Variational autoencoders, contrastive learning | 5, 9 | Latent space smoothness, batch correction |
| **Tissue (spatial)** | Graph (spots/nodes + spatial edges) | Graph neural networks, attention | 10 | Euclidean/physical adjacency, ligand–receptor |
| **Neural population** | Spike counts (time series) | Sequence VAEs, latent dynamics (LFADS, CEBRA) | 11 | Temporal continuity, low-dim manifold |
| **Behavior** | Pose keypoints (time series) | HMMs, contrastive learning | 12 | Temporal smoothness, motor equivalence |
| **Ecological** | Presence/absence matrix + covariates | Random forest, spatial CV | 14 | Spatial autocorrelation, dispersal limitation |
| **Earth system** | Multi-spectral image tiles | Convolutional, foundation models (Prithvi) | 15 | Scale invariance, periodic patterns |

**Key insight:** The shared mathematical primitive across all scales is **representation learning under a domain-specific inductive bias**. The choice of tokenization, architecture, and objective is determined by the biological question, not by a universally "best" method.

**Pitfall:** Do not assume that a method that works well for proteins will work well for ecology without adapting the inductive bias (e.g., spatial autocorrelation in ecology is not analogous to sequence proximity in proteins).

## 24.2  What genuinely changes

- **The unit of progress.** From single hypotheses to *closed loops* of design–build–test–learn (Chapter 22).
- **The cost curve.** Predictions that were career-defining are now routine; attention moves to validation and judgment.
- **The collaboration.** Biologists, ML researchers, ethicists, and clinicians share one artifact — the model — and one accountability.

## 24.2a  What genuinely changes — a before/after comparison

To make the shift concrete, consider a typical research task before and after AI became a standard instrument.

**Task: Identify causal regulatory variants for a complex disease from GWAS.**

| Step | 2015 (pre-AI) | 2026 (with AI) |
|------|---------------|----------------|
| **Variant calling** | GATK (manually tuned parameters) | DeepVariant (learned, fewer false positives) |
| **Fine-mapping** | Statistical fine-mapping (e.g., FINEMAP) | Enformer + in-silico mutagenesis (functional priors) |
| **Regulatory annotation** | Look up known ENCODE elements | SpliceAI + Borzoi predict effect on splicing/expression |
| **Validation priority** | Choose top 10 variants by p-value | Choose top 10 by ensemble score (AlphaMissense + Enformer + SpliceAI) |
| **Time from GWAS to candidate** | 6 months | 2 days |
| **Number of variants tested experimentally** | 50–100 | 10–20 (higher success rate) |
| **Cost per validated variant** | $500,000 | $50,000 |

**What changed:** The bottleneck shifted from computation (aligning reads) to interpretation (choosing which variants to trust). The unit of progress is no longer a single variant but a *causal mechanism hypothesis* that integrates multiple lines of AI-generated evidence.

**Pitfall:** The speed of hypothesis generation has outpaced the speed of experimental validation. Labs must resist the temptation to chase every AI-proposed candidate; a disciplined prioritization framework (Chapter 18) is essential.

## 24.3  What does not change

- **Biology is the ground truth.** A model is a hypothesis; the organism is the referee.
- **Experiments remain decisive.** Correlation is cheap; causation still costs an intervention (Chapter 23).
- **Responsibility is not automatable.** Someone must own each decision a model informs (Chapters 19–20).

## 24.3a  What does not change — a case study of confident failure

Section 24.3 states biology is the ground truth, experiments remain decisive, and responsibility is not automatable. A case study makes the stakes vivid.

**Example:** In 2022, a highly cited DNA language model predicted that a specific non-coding variant in the *BRCA1* promoter would dramatically increase cancer risk. The model had high confidence (p > 0.95). A follow-up CRISPR-based functional assay (saturation genome editing) showed no effect on gene expression. The model had learned a correlation with GC content (confounder) rather than a causal regulatory mechanism.

**Lesson:** No matter how sophisticated the AI, the final arbiter is the living system. Experiments remain decisive. The researcher's responsibility is to design the experiment that can falsify the AI's prediction.

**Responsibility checklist before acting on a model's output:**

- [ ] Has the model been validated on a held-out set that is independent (by family, batch, site)?
- [ ] Is the model's uncertainty well-calibrated for this specific input?
- [ ] Have we considered an orthogonal assay to confirm the result?
- [ ] If the model is wrong, what is the cost? Is there a fallback?

**Pitfall:** Over-reliance on a single model, especially a foundation model that appears authoritative, can lead to systematic errors. Always ensemble multiple models or use a decision-theoretic approach (Chapter 18).

## 24.4  Worked example — a reproducible study skeleton

```python
from dataclasses import dataclass
import hashlib, json, platform

@dataclass
class StudyManifest:
    """Minimal provenance record that should accompany any AI-in-biology result."""
    question: str
    data_version: str
    model_version: str
    seed: int
    code_commit: str
    environment: str

    def fingerprint(self) -> str:
        blob = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

manifest = StudyManifest(
    question="Does variant X alter splicing in tissue Y?",
    data_version="GTEx_v8",
    model_version="SpliceAI-1.3",
    seed=0,
    code_commit="a1b2c3d",          # replace with the actual `git rev-parse --short HEAD`
    environment=platform.platform(),
)
print(manifest.fingerprint())
```

A study that cannot produce a manifest like this cannot be reproduced — and in the new biology, **reproducibility is the minimum bar, not a bonus.**

## 24.4a  Worked example extension — a reproducible workflow with full provenance

The `StudyManifest` above captures provenance for a single result. Scaling that discipline to a whole project means versioning *data*, *code*, and *environment* together. Here is a complete reproducible workflow using `Snakemake` and `DVC` (Data Version Control) for a genomics + ML project.

**Directory structure:**

```
project/
├── data/
│   ├── raw/           # immutable, content-addressed
│   ├── processed/     # output of preprocessing
│   └── .dvc/          # DVC tracking
├── workflows/
│   ├── Snakefile      # snakemake pipeline
│   └── env.yaml       # conda environment
├── models/
│   └── checkpoints/   # model weights
├── reports/
│   └── figures/
└── manifest.yaml      # study manifest
```

**Snakefile snippet (alignment → variant calling → scoring):**

```python
# Snakefile
rule all:
    input:
        "reports/final_scores.csv"

rule align:
    input:
        fastq = "data/raw/{sample}.fastq.gz"
    output:
        bam = "data/processed/{sample}.bam"
    conda:
        "env.yaml"
    shell:
        "bwa-mem2 mem reference.fa {input.fastq} | samtools sort -o {output.bam}"

rule call_variants:
    input:
        bam = "data/processed/{sample}.bam"
    output:
        vcf = "data/processed/{sample}.vcf"
    shell:
        "deepvariant --reads {input.bam} --ref reference.fa --output_vcf {output.vcf}"

rule score_variants:
    input:
        vcf = "data/processed/{sample}.vcf"
    output:
        csv = "reports/{sample}_scores.csv"
    script:
        "score_variants.py"   # Python script that loads Enformer and computes delta
```

**Manifest (YAML) with all provenance:**

```yaml
study:
  name: "BRCA1_variant_prioritization_2026"
  question: "Which rare variants in BRCA1 promoter affect expression?"
  date: "2026-05-30"
  data:
    version: "BRCA1_wgs_2026_v1"
    source: "ENA PRJEB12345"
    checksum: "sha256:abc123..."
  software:
    bwa: "0.7.17"
    deepvariant: "1.5.0"
    enformer: "pytorch_enformer_2024a"
  environment:
    conda_env_hash: "d3b07384d113edec49eaa6238ad5ff00"
  code:
    repository: "https://github.com/group/brca1_prioritization"
    commit: "a1b2c3d4e5f6"
  runtime:
    total_cpu_hours: 120
    gpu_model: "A100"
    gpu_hours: 8
  reproducibility:
    container: "docker.io/group/brca1_analysis:2026-05-30"
    snakemake_version: "7.32.0"
```

**To reproduce:** A reviewer can run `snakemake --use-conda --cores 4` and get identical outputs (provided the same raw data access). This is the minimal bar for the new biology.

**Pitfall:** Reproducibility is not guaranteed even with the above — hardware differences (GPU, CPU, random seeds) can cause minor variations. Use deterministic algorithms (set seed, disable GPU non-determinism) and report tolerance intervals.

## 24.5  Practices of an AI-fluent biologist

1. **Pin everything** — data, weights, tokenizer, seed, environment.
2. **Split by biology** — homology-, batch-, or site-aware splits, never naive random.
3. **Report uncertainty and subgroups** — a number without an interval is a rumor.
4. **Validate orthogonally** — confirm in-silico hits with a different assay.
5. **Document intended use and limits** — model cards travel with weights (Chapter 20).
6. **Design for reversibility** — especially anything released into a body or an ecosystem.

## 24.5a  Practices of an AI-fluent biologist — a code of conduct

The five practices above describe individual habits. They also imply professional obligations. The following code of conduct — inspired by the ACM Code of Ethics and the International Society for Computational Biology (ISCB) guidelines — makes those obligations explicit.

**Code of Conduct for AI-Fluent Biologists**

1. **Reproducibility:** You will release analysis code, environment specifications, and data access instructions for every published result. You will not claim a result as "reproducible" unless an independent researcher can recreate it from your materials.

2. **Uncertainty communication:** You will report confidence intervals, prediction intervals, or posterior distributions for every quantitative claim. You will not report a point estimate without an associated measure of uncertainty.

3. **Model limitations:** You will document in every paper the conditions under which the model is known to fail (e.g., low MSA depth, different species, different assay platform). You will include a "known limitations" section.

4. **Data sovereignty:** You will respect consent agreements and data use restrictions. You will not use human genomic data for purposes beyond the original consent without re-consent or IRB approval.

5. **Dual-use awareness:** You will consider the potential for your model or dataset to enable harm. If the risk is non-negligible, you will consult with a biosecurity board before public release.

6. **Bias and fairness:** You will report subgroup performance (by sex, ancestry, age) for any model intended for clinical or conservation use. You will discuss trade-offs between fairness metrics.

7. **Mentorship:** You will train the next generation not only in methods but also in the responsible interpretation of results. You will encourage junior researchers to speak up when they find errors.

**Pitfall:** A code of conduct without enforcement is aspirational. Journals and conferences should require a signed checklist as part of submission.

## 24.6  Where to go next

- **Deepen the math** — probability, optimization, dynamical systems, causal inference.
- **Deepen the biology** — pick one system and learn it to wet-lab depth.
- **Build in public** — reproducible notebooks, open weights for inference, honest failure reports.
- **Engage the society** — the hardest problems (Chapters 19–21) are sociotechnical, not just technical.

## 24.6a  Where to go next — a 12-month curriculum

The pointers above are deliberately brief. For a biologist who has completed this textbook, the following concrete 12-month self-study plan turns them into a schedule with deliverables.

| Month | Focus | Resources | Deliverable |
|-------|-------|-----------|-------------|
| 1–2 | **Deepen probability & statistics** | *Statistical Rethinking* (McElreath), *Probabilistic Programming & Bayesian Methods for Hackers* (Davidson-Pilon, GitHub) | Implement a hierarchical model in PyMC |
| 3–4 | **Optimization & deep learning fundamentals** | *Dive into Deep Learning* (Zhang et al.), fast.ai course | Train a CNN on a biological image dataset |
| 5–6 | **Causal inference** | *Causal Inference: What If* (Hernán & Robins), *The Book of Why* (Pearl) | Apply Mendelian randomization to a GWAS |
| 7–8 | **Domain-specific project (choose one)** | Protein design (RFdiffusion tutorial), single-cell (scvi-tools), ecology (blockCV) | Reproduce a published result in a Colab notebook |
| 9–10 | **Software engineering for science** | *Good Enough Practices in Scientific Computing* (GitHub), Docker, Snakemake | Containerize your analysis pipeline |
| 11–12 | **Responsible AI & ethics** | Chapters 19–20 of this textbook, *Fairness and Machine Learning* (Barocas et al.) | Write an ethics statement for your project |

**Expected outcome:** You will be able to lead a small AI-in-biology project, from question formulation to reproducible publication, while anticipating ethical and statistical pitfalls.

**Pitfall:** Self-study is lonely. Join a journal club (e.g., ML for Biology at your institution) or a virtual community (e.g., Discord servers for AI in bio) to stay motivated.

## 24.7  Pitfalls to carry forward

- **Tool-first thinking.** Start from the biological question, not the trendiest model.
- **Solved-benchmark complacency.** Real biology is the only benchmark that matters.
- **Forgetting the referee.** No prediction is a result until biology agrees.

**Tool-first thinking (extended).** Starting with a trendy model (e.g., "let's apply a diffusion model to our data") without a clear biological question leads to publication in "method-only" journals but rarely advances biology. **Remedy:** Before writing any code, write down the biological question in one sentence. Then ask: Is this question answerable with existing methods? If yes, use them. If no, *then* consider new AI.

**Forgetting the referee (extended).** A model's prediction is not a result until biology agrees. A common failure is to report in-silico metrics (e.g., AUROC on a test set) as if they were biological truth. **Remedy:** For every claim that "X is important for Y," require an experimental validation step (e.g., CRISPR, knockdown, mutagenesis) or at minimum a clear statement that it remains a prediction.

**Pitfall:** The pressure to publish quickly can lead to skipping experimental validation. Some journals now require a "validation statement" indicating whether predictions were tested.

## 24.8  Exercises

1. **Manifest your own work.** Wrap a previous-chapter exercise in a `StudyManifest`; verify another person can reproduce it from the fingerprint alone.
2. **Cross-scale synthesis.** Choose two chapters from different scales; write one page on the shared method and the differing inductive biases.
3. **Failure report.** Take a model that *did not* work for you; write an honest post-mortem others could learn from.
4. **Roadmap.** Draft a six-month learning plan targeting one open question from Chapter 23.
5. **Replicate a study from your own lab using the manifest.** Take a published result from your group. Create a `StudyManifest` and a Snakemake pipeline that reproduces the main figure. Share it on GitHub. Ask a colleague to run it on a different machine. Document any discrepancies.
6. **Cross-scale synthesis challenge.** Pick a biological phenomenon that spans at least three scales (e.g., cancer metastasis: molecular (mutations) → cellular (migration) → tissue (invasion) → organism (spread)). For each scale, identify the appropriate representation and method from the table in 24.1a. Describe how you would integrate information across scales (e.g., using a hierarchical model or multi-modal fusion).
7. **Failure post-mortem.** Take a model that you tried to apply in your research but that failed to generalize or produce useful predictions. Write a 500-word post-mortem answering: (1) What was the biological question? (2) What did the model predict? (3) How did you validate it? (4) Why did it fail (data, identifiability, confounding, etc.)? (5) What would you do differently next time?
8. **Personal roadmap.** Using the 12-month curriculum in 24.6a as a template, create a personalized 6-month learning plan based on your current skills and research goals. Include specific courses, books, and a project you will complete.

## 24.9  Further reading

- Jumper, J. *Highly accurate protein structure prediction with AlphaFold.* Nature (2021).
- Markowetz, F. *All biology is computational biology.* PLoS Biol. (2017).
- Eraslan, G. *Deep learning: new computational modelling techniques for genomics.* Nat. Rev. Genet. (2019).
- Wang, H. *Scientific discovery in the age of artificial intelligence.* Nature (2023).
- Brynjolfsson, E. & McAfee, A. *The Turing Trap.* (2022) — on automation vs. augmentation.
- The Carpentries. *Good Enough Practices in Scientific Computing.* (2024) — practical guide.

## See also

- [Chapter 1 — Biology as an Information Science](chapter_01_bioinfo_basics.md)
- [Chapter 22 — Co-Evolution of AI & Life](chapter_22_coevolution.md)
- [Chapter 23 — Limits & Open Questions](chapter_23_limits.md)
