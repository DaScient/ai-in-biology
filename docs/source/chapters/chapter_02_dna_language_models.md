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

## 2.2  Tokenization

| Strategy | Vocabulary | Typical use | Trade-off |
|----------|------------|-------------|-----------|
| Character (`A C G T N`) | 5 | Convolutional models, small transformers | Long sequences, simple |
| k-mer (k = 3…6) | 4ᵏ + special tokens | DNABERT-1, classical models | Captures local motif structure; explodes vocabulary |
| Byte-pair encoding | 4 k–32 k | Nucleotide Transformer, DNABERT-2 | Adaptive granularity |
| Single-nucleotide + RoPE | 5 | HyenaDNA, Caduceus | Million-base context windows |

A practical default: start with non-overlapping 6-mers for masked-LM pre-training, and switch to BPE if your downstream task involves variable-length functional elements.

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

## 2.5  Evaluation pitfalls

- **Reverse-complement equivariance.** A good DNA LM should give (near-)identical predictions for a sequence and its reverse complement. Test it.
- **GC-content shortcut.** Many "function" labels are confounded with GC content; baseline against a GC-only classifier.
- **Train / test leakage by homology.** Split *by family*, not by random shuffle.

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

## See also

- [Chapter 3 — Attention in Genomics](chapter_03_attention_in_genomics.md)
- [Chapter 5 — Representation Learning](chapter_05_embeddings.md)
- [Genomics API](../api/genomics.md)
