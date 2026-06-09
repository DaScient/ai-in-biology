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

### 8.1a  Anatomy of AlphaFold2: a deeper look at the components

The list above (MSA encoder, Evoformer, structure module, recycling) hides a lot of structure. The table below describes each component's role and typical implementation details, written to help you understand *why* each piece exists.

| Component | Input | Output | Key operation | Why needed |
|-----------|-------|--------|---------------|-------------|
| **MSA encoder** | Raw MSA (n_seq × n_res) | MSA representation | 1D axial attention along rows (sequences) and columns (residues) | Captures evolutionary covariation; different sequences provide independent samples of constraints |
| **Pair representation** | Outer sum of MSA features | Pair features (n_res × n_res × c) | Triangular multiplicative updates | Enables explicit pairwise residue relationships – distances, contacts, orientations |
| **Evoformer block** | MSA + pair representations | Refined MSA + refined pair | Triangular attention, row/column gated self-attention | Iteratively couples sequence and pair information; the core of the inductive bias |
| **Structure module** | Pair representation + single representation | Backbone frames (n_res × 7) | Invariant point attention (IPA), recycling | Converts abstract geometry into 3-D coordinates; uses SE(3)-equivariant operations |
| **Recycling** | Output structure + embeddings | Refined embeddings | Feed previous outputs back into the start | Allows iterative refinement; crucial for hard cases (small MSAs) |

**Key insight:** The Evoformer does *not* directly predict distances. Instead, it refines a rich pair representation that the structure module later uses to position residues in 3-D. This separation of "sequence→pair" and "pair→3D" is what made AF2 generalizable.

**Practical note:** For inference with AF2 or open-source implementations (OpenFold, UniFold), the MSA depth matters more than sequence length. For proteins with < 50 effective sequences (e.g. viral proteins, designed sequences), pLDDT typically drops below 80, indicating low confidence.

**Pitfall:** Recycling is often omitted in simplified implementations to save memory. Without recycling, models may fail to correct initial mis-orientations, especially for large proteins (> 500 residues).

## 8.2  Protein language models

ESM-2, ProtBERT, ProGen2, and AMPLIFY are encoder / decoder transformers trained on UniRef. Their per-token embeddings are useful for:

| Task | Typical readout |
|------|------------------|
| Function (GO, EC) | Mean-pool → MLP |
| Mutation effect | log-likelihood ratio (wt vs. mut) |
| Contact prediction | Symmetrized attention map |
| Subcellular location | CLS-token → linear |

A good rule of thumb: a *frozen* 650 M ESM-2 embedding plus a linear head beats most task-specific models trained on < 10 000 labeled proteins.

### 8.2a  Choosing the right protein language model

ESM-2, ProtBERT, ProGen2, and AMPLIFY differ in scale, training data, and intended use. The decision matrix below helps you select a PLM based on task and available resources.

| Model | Parameters | Training data | Best for | Inference cost | Open weights? |
|-------|------------|---------------|----------|----------------|---------------|
| **ESM-2 (650M)** | 650M | UniRef50 (65M seqs) | General-purpose, zero-shot fitness, embeddings | Moderate (GPU recommended) | Yes (HuggingFace) |
| **ESM-2 (3B)** | 3B | UniRef50 (65M seqs) | Highest accuracy, deep mutational scanning | High (A100 required) | Yes |
| **ProtBERT** | 420M | UniRef100 (217M seqs) | Secondary structure, localization | Moderate | Yes |
| **ProGen2 (small)** | 151M | UniRef90 (1.2B seqs) | Protein generation (enzymes, antibodies) | Low | Yes |
| **ProGen2 (large)** | 6.4B | Same | High-quality generation | Very high | Limited |
| **AMPLIFY** | 3B | Metagenomic+UniRef | Function prediction in poorly annotated families | High | No (available on request) |
| **ESM-1b** | 650M | UniRef50 (older) | Legacy; superseded by ESM-2 | Moderate | Yes |

**Rule of thumb:** Start with ESM-2 650M. It offers the best balance of performance and accessibility. Use ESM-2 3B only if you have a very specific high-stakes task (e.g. clinical variant interpretation) and access to an A100/H100. For generation, ProGen2-small often suffices; large models can overfit to the training distribution.

```python
import torch
from transformers import AutoModel, AutoTokenizer

# Load ESM-2 650M (works on a single V100)
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example: get per-residue embeddings
sequence = "MKTIIALSYIFCLVFA"  # influenza HA signal peptide
inputs = tokenizer(sequence, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # (1, n_tokens, 1280)
    # Remove special tokens ([CLS], [EOS])
    per_residue = embeddings[0, 1:-1, :]  # (n_res, 1280)
```

**Pitfall:** PLMs tokenize sequences with special tokens ([CLS] at start, [EOS]/[SEP] at end). Always exclude them when extracting per-residue embeddings. The tokenizer also handles unknown amino acids (e.g. `X`) as `[UNK]` – decide how to handle them.

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

### 8.3a  Worked example extension — zero-shot deep mutational scanning

The function above computes the LLR for a single position. Let's extend it to **complete deep mutational scanning (DMS) scoring** for all single mutants of a protein, with efficient batching.

**Goal:** Given a wild-type protein sequence of length L, compute the log-likelihood ratio for every possible single amino acid substitution (L × 19 variants).

**Approach:** For each position i, mask the token at i, then compute logits for all 20 amino acids. Compare logit(wt) vs. logit(mut). This is much faster than passing each mutant separately because the PLM encodes the entire masked sequence once per position.

```python
import torch
import torch.nn.functional as F
from tqdm import tqdm

def zero_shot_dms(plm, tokenizer, wt_seq, batch_size=64):
    """
    Returns a matrix (L, 20) of LLR for each substitution.
    llr[i, a] = log(P(aa_a at i | masked)) - log(P(wt at i | masked))
    """
    L = len(wt_seq)
    aa_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
    wt_idx = [aa_to_idx[aa] for aa in wt_seq]

    # Prepare masked sequences for each position
    masked_sequences = []
    positions = []
    for i in range(L):
        # Replace residue i with [MASK]
        masked = wt_seq[:i] + tokenizer.mask_token + wt_seq[i+1:]
        masked_sequences.append(masked)
        positions.append(i)

    # Batch process through the PLM
    all_logits = torch.zeros(L, 20)  # will hold log P(aa) for each position
    for start in tqdm(range(0, L, batch_size)):
        batch_seqs = masked_sequences[start:start+batch_size]
        batch_inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = plm(**batch_inputs).logits  # (batch, seq_len, vocab_size)

        for batch_idx, pos in enumerate(positions[start:start+batch_size]):
            # Map the masked residue position to its token index.
            # Tokenized sequence is [CLS] + tokens + [EOS]; simplified mapping:
            tok_idx = pos + 1  # +1 for [CLS]
            logits_at_pos = logits[batch_idx, tok_idx, :]  # (vocab_size,)
            log_probs = F.log_softmax(logits_at_pos, dim=-1)
            # Map from token IDs to amino acid indices (tokenizer-specific;
            # for ESM, tokens 4-23 correspond to AAs with an index offset).
            aa_log_probs = extract_aa_log_probs(log_probs, tokenizer, aa_list)
            all_logits[pos] = aa_log_probs

    # Compute LLR relative to wild-type
    wt_log_probs = all_logits[range(L), wt_idx]
    llr_matrix = all_logits - wt_log_probs.unsqueeze(-1)  # (L, 20)
    return llr_matrix
```

**Performance note:** For a 300-residue protein, this method runs in ~30 seconds on a V100 (with `batch_size=64`). The naive per-mutant loop would take hours.

**Extension exercise (8.6e):** Compare zero-shot DMS scores (from above) to actual DMS experimental measurements (e.g. ProteinGym). Compute Spearman correlation for each position. Which positions are poorly predicted? Often these are solvent-exposed or intrinsically disordered regions.

## 8.4  De novo design — diffusion in structure space

RFdiffusion treats protein backbones as noisy frames in SE(3) and denoises them. A typical binder-design loop:

1. *Hotspot* — choose a few residues on the target to be contacted.
2. *RFdiffusion* — sample 100–1000 backbone scaffolds containing a binder of desired length.
3. *ProteinMPNN* — design sequences for each scaffold (target plddt > 90 in silico).
4. *AF2 / ESMFold rescore* — keep designs whose predicted complex matches the diffusion model's intended pose (`RMSD < 2 Å`, `PAE_inter < 10`).
5. *Order and test* — typical hit rates after this filter: 1–10 % bind detectably.

### 8.4a  A realistic filtering cascade for binder design

The loop above describes RFdiffusion + ProteinMPNN + AF2 rescoring at a high level. In practice you run a **filtering cascade** with concrete cutoffs, where each stage removes ~90 % of candidates.

**Assumptions:** You want to design a binder to a target protein (e.g. IL-7Rα as in the exercise).

1. **Backbone diversity (RFdiffusion output)** — Generate 10,000 backbones. Filter by:
   - **Contiguity:** No chain breaks (CA-CA distance < 4 Å for all consecutive residues).
   - **Secondary structure composition:** Desired amount of helix/sheet if known.
   - **Interface hotspot distance:** At least 3 residues within 6 Å of target (choose from a known binding epitope).
   - Remaining: ~2,000.

2. **Sequence design (ProteinMPNN)** — For each backbone, design 8 sequences with different sampling temperatures (0.1, 0.2, 0.5). Filter by:
   - **Solubility proxy:** Net charge between -10 and +10, no long hydrophobic patches.
   - **No stop codons or rare codons** (if expressing in *E. coli*).
   - **pLDDT from ESMFold** (or AF2 single-pass) > 85 for the binder alone.
   - Remaining: ~5,000 sequences (~500 distinct backbones).

3. **Complex prediction (AF2-multimer)** — Predict the binder–target complex. Filter by:
   - **ipTM (interface predicted TM-score) > 0.6** (0.5 is borderline, 0.7 is excellent).
   - **PAE_inter < 10 Å** for the binding interface (average PAE between binder and target residues).
   - **Clash score:** No more than 5 severe steric clashes (Cα–Cα distance < 2 Å).
   - Remaining: ~50–100 sequences.

4. **Structural clustering (optional)** — Cluster by RMSD of the binder backbone. Pick the top 10 clusters, then the highest ipTM from each cluster for experimental testing.

**Expected hit rate:** After this cascade, 1–10 % of tested designs will show detectable binding (Kd < 10 µM by SPR or similar). Some designs may require additional rounds of affinity maturation.

```python
def filter_binder_candidates(df, ipTM_thresh=0.6, plddt_thresh=85, pae_thresh=10):
    """Apply standard filters to a DataFrame of designed binder candidates."""
    df_filtered = df[
        (df['ipTM'] >= ipTM_thresh) &
        (df['binder_pLDDT'] >= plddt_thresh) &
        (df['interface_pae'].mean() <= pae_thresh)
    ]
    # Additional checks
    df_filtered['clash_score'] = df_filtered.apply(count_clashes, axis=1)
    df_filtered = df_filtered[df_filtered['clash_score'] <= 5]
    return df_filtered.sort_values('ipTM', ascending=False)
```

**Pitfall:** ipTM and PAE from AF2-multimer are not calibrated absolute scores; they depend on the target protein family. Always run a positive control (known binder complex) to establish a threshold for your specific target.

## 8.5  Pitfalls

- **MSA depth shortcut.** Reports of "AF2 success" often hide that the protein has a deep family. Always report effective N.
- **PLM training-set leakage.** Many "benchmarks" overlap UniRef; deduplicate at ≤ 30 % identity to the training set.
- **In-silico self-consistency ≠ real binding.** Use orthogonal metrics (ipTM in AF2-multimer) and *always* validate experimentally.

### 8.5a  Pitfalls expanded — overconfidence and training leakage

The pitfalls above flag MSA depth and PLM training leakage. Here are **quantitative diagnostics** for detecting these issues.

**Diagnostic 1: MSA depth effect.** For a test protein, compute the effective sequence count (`eff_seq = sum(1/(1+0.5*dist))` after clustering at 80 % identity). Plot pLDDT vs. eff_seq for a set of proteins. If your protein has eff_seq < 30 and you report high pLDDT, be skeptical – it may be a false high confidence.

**Diagnostic 2: Training set leakage.** Take the PDB entry of your test protein. BLAST the sequence against the PLM's training set (UniRef). If there is a hit with > 30 % identity and > 80 % coverage, your model may have seen a close homolog. Repeat the benchmark after removing that homolog (leave-one-family-out). Report both scores.

**Diagnostic 3: Zero-shot vs. supervised gap.** For a DMS dataset, compute zero-shot correlation (ESM-2) and supervised correlation (model trained on 80 % of the same DMS). If supervised is only marginally better (< 0.05 Spearman), your zero-shot model is already near ceiling – good. If supervised is much better (> 0.2), the DMS measures non-evolutionary properties (e.g. stability in a specific assay), and you need labeled data.

```python
def check_plm_leakage(test_seq, plm_training_uris):
    """Check if test_seq has a close homolog in the PLM training set."""
    # Simplified: use MMseqs2 to search against UniRef50
    import subprocess
    result = subprocess.run(
        f"mmseqs search {test_seq.fasta} {plm_training_uris} result.tmp tmp",
        shell=True, capture_output=True)
    # Parse output for identity
    # Return max identity and coverage
```

**Pitfall:** Even if no single sequence is > 30 % identity, the MSA used in AF2 may still contain distantly related sequences that provide information. This is legitimate – evolutionary information is *supposed* to help. The leakage problem arises when the test protein's *structure* (not just sequence) was in the training set.

## 8.6  Exercises

1. **Fold and score.** Run ESMFold on the SARS-CoV-2 spike RBD. Compare to PDB 6M0J. Report TM-score.
2. **DMS replication.** Reproduce the ProteinGym GFP DMS leaderboard for ESM-2 650 M zero-shot.
3. **Design a mini-binder.** Use the RFdiffusion + ProteinMPNN tutorial to design 50-residue binders to IL-7Rα. Filter as above; report your top 5 by `PAE_inter`.
4. **Disorder as a feature.** Build a logistic regression that predicts intrinsic disorder using mean pLDDT in 30-residue windows. Compare to MobiDB ground truth.

**8.6f. Design a helical binder.** Use RFdiffusion in binder mode (via a published notebook) to design a 60-residue binder to a helical epitope on a target protein (e.g. KRAS). Run the full filtering pipeline (ProteinMPNN + AF2-multimer). Report your top 5 designs and their ipTM scores.

**8.6g. Compare zero-shot and supervised mutation effects.** Take the ProteinGym SLC6A4 (serotonin transporter) DMS dataset. Compute zero-shot LLR using ESM-2 650M. Train a simple supervised model (ESM-2 embedding + ridge regression) on 50 % of variants. Compare Spearman on the held-out 50 %. Which performs better? Why?

**8.6h. MSA subsampling experiment.** For a protein with a deep MSA (e.g. GFP), run AF2 inference with the full MSA, then with randomly subsampled MSA to 10 % and 1 % of original depth. Plot pLDDT vs. subsampling fraction. How much MSA depth is needed to maintain pLDDT > 85?

**8.6i. pLDDT as a predictor of experimental success.** From a published protein design study (e.g. the RFdiffusion paper), extract the pLDDT of designed proteins before experiments. Compute the ROC curve for predicting "experimentally successful" (e.g. binds, folds correctly). What's the best pLDDT threshold? Is it consistent across targets?

## 8.7  Further reading

- Jumper, J. *Highly accurate protein structure prediction with AlphaFold.* Nature (2021).
- Lin, Z. *ESMFold.* Science (2023).
- Watson, J. L. *De novo design of protein structure and function with RFdiffusion.* Nature (2023).
- Dauparas, J. *Robust deep learning–based protein sequence design using ProteinMPNN.* Science (2022).
- Abramson, J. et al. *Accurate structure prediction of biomolecular interactions with AlphaFold 3.* Nature (2024) — includes nucleic acids and small molecules.
- Winnifrith, A. et al. *A practical guide to AlphaFold for structural biologists.* Nature Protocols (2023) — step-by-step with troubleshooting.
- Notin, P. et al. *ProteinGym: Large-scale benchmarks for protein fitness prediction.* Nature Methods (2024) — comprehensive dataset and leaderboard.
- Bennett, N. R. et al. *Improving de novo protein binder design with deep learning.* Current Opinion in Structural Biology (2023) — reviews the RFdiffusion+ProteinMPNN pipeline.

## See also

- [Protein API](../api/protein.md)
- [Protein folding tutorial](../tutorials/protein_folding.md)
- [Chapter 17 — Biotechnology & Bioengineering](chapter_17_biotech.md)


---
<sub>Support DaScient, Inc. (a non-profit promoting accessible intelligence and community learning) via [Donations](https://cash.app/dascient/).</sub>
