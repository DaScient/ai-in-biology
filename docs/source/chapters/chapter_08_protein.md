# Chapter 8 — Protein Structure & Design

> *"Structure is destiny: most of what a protein does is implicit in how it folds."*

## Learning objectives

- Sketch the AlphaFold2 / ESMFold / RoseTTAFold-AA architectures and identify the role of each major component (MSA encoder, Evoformer / triangular updates, structure module, recycling).
- Compute and interpret pLDDT, PAE, TM-score, and lDDT.
- Use a protein language model (PLM) embedding for a downstream task: function annotation, mutation effect prediction, or binder design.
- Generate candidate de novo proteins with a diffusion model (RFdiffusion / Chroma) and score them with structure prediction + sequence design.

## 8.1  Predicting structure from sequence

The dominant paradigm:

1. **Build evolutionary context.** Either an MSA (AF2, RoseTTAFold) or a single-sequence PLM (ESMFold, OmegaFold).
2. **Iterate pair representation.** Triangular multiplicative / attention updates refine pairwise residue features that approximate distances and angles.
3. **Decode to 3-D.** The structure module yields backbone frames; side chains are added via a rotamer prediction step.
4. **Recycle.** The output of one pass becomes the input of the next.

Empirical truths:

- For single-domain monomers with ≥100 effective sequences in the MSA, AF2 generally reaches lDDT > 0.85.
- pLDDT < 70 strongly predicts disorder (use that as a *feature*, not a bug).
- PAE is the right confidence metric for inter-domain *orientations* (e.g. multimers).

## 8.2  Protein language models

ESM-2, ProtBERT, ProGen2, and AMPLIFY are encoder / decoder transformers trained on UniRef. Their per-token embeddings are useful for:

| Task | Typical readout |
|------|------------------|
| Function (GO, EC) | Mean-pool → MLP |
| Mutation effect | log-likelihood ratio (wt vs. mut) |
| Contact prediction | Symmetrized attention map |
| Subcellular location | CLS-token → linear |

A good rule of thumb: a *frozen* 650 M ESM-2 embedding plus a linear head beats most task-specific models trained on < 10 000 labeled proteins.

## 8.3  Worked example — zero-shot mutation effect

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def zero_shot_effect(plm, tokenizer, wt: str, mut_pos: int, mut_aa: str) -> float:
    """LLR(mut/wt) at a single position using a masked PLM."""
    ids = tokenizer.encode(wt, return_tensors="pt")
    mask_id = tokenizer.mask_token_id
    masked = ids.clone()
    masked[0, mut_pos + 1] = mask_id          # +1 for CLS
    logits = plm(masked).logits[0, mut_pos + 1]
    log_p = F.log_softmax(logits, dim=-1)
    return (log_p[tokenizer.convert_tokens_to_ids(mut_aa)]
            - log_p[tokenizer.convert_tokens_to_ids(wt[mut_pos])]).item()
```

This scoring scheme correlates ~0.5 Spearman with deep mutational scanning fitness on average — strong enough to *triage* libraries before wet-lab work.

## 8.4  De novo design — diffusion in structure space

RFdiffusion treats protein backbones as noisy frames in SE(3) and denoises them. A typical binder-design loop:

1. *Hotspot* — choose a few residues on the target to be contacted.
2. *RFdiffusion* — sample 100–1000 backbone scaffolds containing a binder of desired length.
3. *ProteinMPNN* — design sequences for each scaffold (target plddt > 90 in silico).
4. *AF2 / ESMFold rescore* — keep designs whose predicted complex matches the diffusion model's intended pose (`RMSD < 2 Å`, `PAE_inter < 10`).
5. *Order and test* — typical hit rates after this filter: 1–10 % bind detectably.

## 8.5  Pitfalls

- **MSA depth shortcut.** Reports of "AF2 success" often hide that the protein has a deep family. Always report effective N.
- **PLM training-set leakage.** Many "benchmarks" overlap UniRef; deduplicate at ≤ 30 % identity to the training set.
- **In-silico self-consistency ≠ real binding.** Use orthogonal metrics (ipTM in AF2-multimer) and *always* validate experimentally.

## 8.6  Exercises

1. **Fold and score.** Run ESMFold on the SARS-CoV-2 spike RBD. Compare to PDB 6M0J. Report TM-score.
2. **DMS replication.** Reproduce the ProteinGym GFP DMS leaderboard for ESM-2 650 M zero-shot.
3. **Design a mini-binder.** Use the RFdiffusion + ProteinMPNN tutorial to design 50-residue binders to IL-7Rα. Filter as above; report your top 5 by `PAE_inter`.
4. **Disorder as a feature.** Build a logistic regression that predicts intrinsic disorder using mean pLDDT in 30-residue windows. Compare to MobiDB ground truth.

## 8.7  Further reading

- Jumper, J. *Highly accurate protein structure prediction with AlphaFold.* Nature (2021).
- Lin, Z. *ESMFold.* Science (2023).
- Watson, J. L. *De novo design of protein structure and function with RFdiffusion.* Nature (2023).
- Dauparas, J. *Robust deep learning–based protein sequence design using ProteinMPNN.* Science (2022).

## See also

- [Protein API](../api/protein.md)
- [Protein folding tutorial](../tutorials/protein_folding.md)
- [Chapter 17 — Biotechnology & Bioengineering](chapter_17_biotech.md)
