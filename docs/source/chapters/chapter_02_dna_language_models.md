# Chapter 2 — DNA Language Models

> *"Treat the genome as a corpus, and natural-language tools become genomic tools."*

## Learning objectives

- Explain the analogy (and the limits of the analogy) between human language and nucleotide sequence.
- Describe the three dominant tokenization strategies for DNA: character, k-mer, and byte-pair (BPE) / SentencePiece.
- Train a small masked-language model on bacterial sequences and use its embeddings for a downstream task.
- Diagnose common failure modes: tokenizer artifacts, repetitive-region collapse, GC-content bias.

## 2.1  Why "language" at all?

Language models exploit two properties also present in genomes:

1. **Locality with long-range structure.** Words depend on neighbors but also on document-scale context (subject–verb agreement, narrative). Promoters interact with enhancers across hundreds of kilobases.
2. **Compositional reuse.** A small alphabet recombines into a vast vocabulary. The genome reuses motifs, domains, and entire exons.

The analogy fails where biology has no equivalent — there is no "speaker intent", and the channel includes mutation, recombination, and selection.

### 2.1a  The analogy, deeper: linguistic and genomic structure

The analogy between natural language and DNA is powerful but often over-sold. A more precise, structured comparison:

| Property | Natural Language | Genome |
|----------|------------------|--------|
| **Alphabet size** | 20–30 characters (incl. punctuation) | 4 bases (plus methylation as diacritic) |
| **Word boundaries** | Spaces, punctuation (mostly unambiguous) | No universal tokenization; codons, k-mers, motifs are overlapping |
| **Grammar** | Context-free (in Chomsky hierarchy) + statistical constraints | More complex: overlapping reading frames, RNA secondary structure, chromatin state context |
| **Long-range dependencies** | Subject–verb agreement across tens of words | Enhancer–promoter across hundreds of kilobases; chromosome looping |
| **Ambiguity** | Homonyms, polysemy | Genetic code is nearly unambiguous; but regulatory elements can be interpreted differently depending on cellular state |
| **Generative process** | Human intent, communicative goal | Mutation, selection, drift, recombination – no intent |
| **Training data scale** | Trillions of tokens (web crawl) | Millions to billions of base pairs (human genome 3B) – far less than language models |

The practical consequence: **direct transfer of NLP architectures works better than expected, but fails in predictable ways.** For example, BPE tokenization (byte-pair encoding) designed for natural language introduces artifacts in genomes because the most frequent "subwords" are often simple repeats (ATATAT…) that have no functional meaning. Genomic tokenization must be biology-aware.

**Recommended practice:** Always benchmark tokenization strategies on a downstream task reflective of your biological question, not just perplexity.

## 2.2  Tokenization

| Strategy | Vocabulary | Typical use | Trade-off |
|----------|------------|-------------|-----------|
| Character (`A C G T N`) | 5 | Convolutional models, small transformers | Long sequences, simple |
| k-mer (k = 3…6) | 4ᵏ + special tokens | DNABERT-1, classical models | Captures local motif structure; explodes vocabulary |
| Byte-pair encoding | 4 k–32 k | Nucleotide Transformer, DNABERT-2 | Adaptive granularity |
| Single-nucleotide + RoPE | 5 | HyenaDNA, Caduceus | Million-base context windows |

A practical default: start with non-overlapping 6-mers for masked-LM pre-training, and switch to BPE if your downstream task involves variable-length functional elements.

### 2.2a  Tokenization benchmarking protocol

Extending the discussion above, here is a rigorous way to compare tokenizers.

**Metrics for genomic tokenizers:**

1. **Perplexity on a held-out test set** – lower is better, but can be gamed by over-segmentation (e.g., single-base tokens always achieve low perplexity but lose long-range information).
2. **Downstream task performance** – e.g., promoter classification, splice-site prediction, variant effect scoring.
3. **Reverse-complement consistency** – For a tokenizer that is not naturally RC-invariant (e.g., most BPE tokenizers), compute the similarity of token sequences for a string and its reverse complement. High similarity suggests biological robustness.
4. **Token distribution** – Zipf's law (frequency vs. rank) should be roughly power-law; an extreme long tail (many unique tokens) indicates over-segmentation.

```python
from collections import Counter
import numpy as np

def tokenizer_zipf(tokens):
    """Return exponent of fitted power law to token frequencies."""
    counts = Counter(tokens)
    freqs = np.array(sorted(counts.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)
    log_freq = np.log(freqs)
    log_rank = np.log(ranks)
    # simple linear fit
    slope, intercept = np.polyfit(log_rank, log_freq, 1)
    return -slope  # Zipf exponent: ~1 for natural language, higher for more uniform
```

**Pitfall:** A Zipf exponent far from 1 (e.g., > 2) suggests tokenization is not capturing the natural structure. For the human genome with 6-mers, the exponent is about 1.3.

## 2.3  Architectures at a glance

- **Encoder-only (BERT-style).** Mask 15 % of tokens; predict them. Good for representation learning, variant scoring, classification. DNABERT, Nucleotide Transformer.
- **Decoder-only (GPT-style).** Predict next token. Good for generation: synthetic promoters, codon-optimized CDS. ProGen2, EVO.
- **State-space / SSM.** Sub-quadratic attention via convolution-like recurrences. Enables full-chromosome context. HyenaDNA, Mamba-DNA.

## 2.4  Worked example — masked-LM on bacterial 16S

```python
import torch
from torch import nn

class TinyDNABERT(nn.Module):
    """Toy encoder for masked-LM on k-mer tokenized DNA."""

    def __init__(self, vocab_size: int = 4**6 + 4, d_model: int = 128, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(2048, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        h = self.tok(x) + self.pos(pos)
        h = self.enc(h)
        return self.head(h)
```

A 5 M-parameter model trained on a few hundred MB of bacterial 16S sequence will already separate phyla in its `[CLS]` embedding space.

### 2.4a  Extended worked example — fine-tuning a DNA-LM for promoter strength

The worked example above trains a small masked LM on 16S. Augment it with a full fine-tuning pipeline for a regression task: predicting promoter strength from sequence (e.g., from the **J432** promoter library dataset).

**Steps:**

1. Obtain a frozen pre-trained model (e.g., DNABERT-2 or a small HyenaDNA checkpoint).
2. Replace the masked-LM head with a regression head (linear layer on the `[CLS]` token).
3. Fine-tune with MSE loss on 80% of promoters; test on 20%.
4. Evaluate the Spearman correlation between predicted and measured expression.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class PromoterStrengthPredictor(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.regressor(self.dropout(cls_emb)).squeeze(-1)

# Usage:
# tokenizer = AutoTokenizer.from_pretrained("hyenadna-tiny-1k")
# model = PromoterStrengthPredictor("hyenadna-tiny-1k")
# ... training loop with MSE loss
```

**Extension exercise (2.6e):** Compare the fine-tuned model against a simple baseline: GC content + TATA-box presence + linear regression. Which performs better with 100, 1000, 10,000 labeled promoters? Plot learning curves.

## 2.5  Evaluation pitfalls

- **Reverse-complement equivariance.** A good DNA LM should give (near-)identical predictions for a sequence and its reverse complement. Test it.
- **GC-content shortcut.** Many "function" labels are confounded with GC content; baseline against a GC-only classifier.
- **Train / test leakage by homology.** Split *by family*, not by random shuffle.

### 2.5a  Evaluation pitfalls deep dive — reverse-complement equivariance

Many DNA language models claim near-equivariance (output same for a sequence and its RC). But careful testing reveals failures.

**Why RC equivariance matters:** The physical DNA molecule has no orientation bias (except strand-specific regulatory elements like origins of replication). If a model gives different predictions for a sequence and its RC, it has learned a spurious strand bias, likely from training data that accidentally correlates strand with label.

**How to test properly:**

- Generate a test set of 10,000 random 500-bp sequences (non-coding, no known strand bias).
- For each, compute the prediction (e.g., a logit or embedding).
- Compute cosine similarity between original and RC embeddings (for encoder models) or absolute difference for scalar predictions.
- Report the distribution of similarity; a good model has median cosine > 0.99.

**Failure example:** DNABERT (original) had poor RC equivariance because its absolute positional encodings broke symmetry. Later versions (DNABERT-2) used relative position encodings to fix this.

**Remediation:** If your model is not naturally equivariant, you can enforce it by **data augmentation**: train on both the sequence and its RC, or modify the architecture (e.g., using RC-invariant tokenization and pooling).

```python
import numpy as np
import torch

def rc_consistency_check(model, tokenizer, seqs, device):
    diffs = []
    for seq in seqs:
        rc_seq = reverse_complement(seq)  # implement yourself or use BioPython
        ids_seq = tokenizer(seq, return_tensors="pt").to(device)
        ids_rc = tokenizer(rc_seq, return_tensors="pt").to(device)
        with torch.no_grad():
            emb_seq = model(**ids_seq).last_hidden_state.mean(dim=1)
            emb_rc = model(**ids_rc).last_hidden_state.mean(dim=1)
        cos = torch.nn.functional.cosine_similarity(emb_seq, emb_rc).item()
        diffs.append(1 - cos)
    return np.mean(diffs), np.std(diffs)
```

## 2.6  Exercises

1. **Tokenize and count.** Implement non-overlapping 6-mer tokenization. Plot the distribution of k-mer frequencies in the *E. coli* genome. How many of the `4096` possible 6-mers never appear?
2. **Train your own.** Train the `TinyDNABERT` above on the [Microbial Genome Atlas](https://www.mgnify.org/) 16S subset (1 epoch on Colab T4). Report perplexity.
3. **Variant priors.** Use the trained masked-LM to score each SNP in the GIAB high-confidence call set by its log-likelihood ratio (alt vs ref). Compare with CADD.
4. **Reverse-complement check.** Pass `seq` and `reverse_complement(seq)` through your model. Report the cosine similarity of their `[CLS]` embeddings.

## 2.7  Further reading

- Ji, Y. *DNABERT.* Bioinformatics 37 (2021).
- Dalla-Torre, H. *The Nucleotide Transformer.* bioRxiv (2023).
- Nguyen, E. *HyenaDNA.* NeurIPS (2023).
- Benegas, G. *DNA language models are powerful predictors of genome-wide variant effects.* PNAS (2023).

### Extended further reading

In addition to the works cited above:

- **Dalla-Torre, H. et al. (2024).** "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics." *Nature Genetics* – includes comprehensive RC equivariance benchmarks.
- **Benegas, G. et al. (2024).** "DNA language models are powerful predictors of genome-wide variant effects." *PNAS* – includes a comparison of tokenization strategies.
- **Nguyen, E. et al. (2024).** "HyenaDNA: Long-Range Genomic Foundation Model with Sub-quadratic Attention." *ICML* – demonstrates 1M context length.
- **Karollus, A. et al. (2023).** "Current pitfalls in the evaluation of DNA language models." *Bioinformatics* – critical review of homology leakage and benchmark design.

## See also

- [Chapter 3 — Attention in Genomics](chapter_03_attention_in_genomics.md)
- [Chapter 5 — Representation Learning](chapter_05_embeddings.md)
- [Genomics API](../api/genomics.md)
