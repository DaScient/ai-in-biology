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

## 24.2  What genuinely changes

- **The unit of progress.** From single hypotheses to *closed loops* of design–build–test–learn (Chapter 22).
- **The cost curve.** Predictions that were career-defining are now routine; attention moves to validation and judgment.
- **The collaboration.** Biologists, ML researchers, ethicists, and clinicians share one artifact — the model — and one accountability.

## 24.3  What does not change

- **Biology is the ground truth.** A model is a hypothesis; the organism is the referee.
- **Experiments remain decisive.** Correlation is cheap; causation still costs an intervention (Chapter 23).
- **Responsibility is not automatable.** Someone must own each decision a model informs (Chapters 19–20).

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

## 24.5  Practices of an AI-fluent biologist

1. **Pin everything** — data, weights, tokenizer, seed, environment.
2. **Split by biology** — homology-, batch-, or site-aware splits, never naive random.
3. **Report uncertainty and subgroups** — a number without an interval is a rumor.
4. **Validate orthogonally** — confirm in-silico hits with a different assay.
5. **Document intended use and limits** — model cards travel with weights (Chapter 20).
6. **Design for reversibility** — especially anything released into a body or an ecosystem.

## 24.6  Where to go next

- **Deepen the math** — probability, optimization, dynamical systems, causal inference.
- **Deepen the biology** — pick one system and learn it to wet-lab depth.
- **Build in public** — reproducible notebooks, open weights for inference, honest failure reports.
- **Engage the society** — the hardest problems (Chapters 19–21) are sociotechnical, not just technical.

## 24.7  Pitfalls to carry forward

- **Tool-first thinking.** Start from the biological question, not the trendiest model.
- **Solved-benchmark complacency.** Real biology is the only benchmark that matters.
- **Forgetting the referee.** No prediction is a result until biology agrees.

## 24.8  Exercises

1. **Manifest your own work.** Wrap a previous-chapter exercise in a `StudyManifest`; verify another person can reproduce it from the fingerprint alone.
2. **Cross-scale synthesis.** Choose two chapters from different scales; write one page on the shared method and the differing inductive biases.
3. **Failure report.** Take a model that *did not* work for you; write an honest post-mortem others could learn from.
4. **Roadmap.** Draft a six-month learning plan targeting one open question from Chapter 23.

## 24.9  Further reading

- Jumper, J. *Highly accurate protein structure prediction with AlphaFold.* Nature (2021).
- Markowetz, F. *All biology is computational biology.* PLoS Biol. (2017).
- Eraslan, G. *Deep learning: new computational modelling techniques for genomics.* Nat. Rev. Genet. (2019).
- Wang, H. *Scientific discovery in the age of artificial intelligence.* Nature (2023).

## See also

- [Chapter 1 — Biology as an Information Science](chapter_01_bioinfo_basics.md)
- [Chapter 22 — Co-Evolution of AI & Life](chapter_22_coevolution.md)
- [Chapter 23 — Limits & Open Questions](chapter_23_limits.md)
