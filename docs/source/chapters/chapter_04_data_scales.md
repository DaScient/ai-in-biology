# Chapter 4 — Biological Data Across Scales

> *"There is no 'one' biological dataset; there is a hierarchy of measurements at different scales of space, time, and abstraction."*

## Learning objectives

- Place common biological data modalities on a space × time × abstraction grid.
- Understand the dominant file formats (FASTA, FASTQ, BAM, VCF, MTX, H5AD, OME-Zarr, NWB) and the situations where each is appropriate.
- Quantify the storage, I/O, and compute budgets that distinguish "laptop biology" from "cluster biology" from "warehouse biology".
- Build a minimal data pipeline that ingests raw data, validates schemas, and produces an analysis-ready tensor.

## 4.1  The biological data hierarchy

| Scale | Typical resolution | Modalities | Storage / sample |
|-------|--------------------|------------|-------------------|
| Atom | 0.1 nm, fs | Cryo-EM tomograms, MD trajectories | 10–500 GB |
| Molecule | 1 nm, μs | Mass spec, structures (PDB) | 10 KB–1 GB |
| Genome | 1 bp, generations | DNA-seq, RNA-seq, ATAC-seq | 1–100 GB |
| Cell | 1 µm, min | scRNA, scATAC, imaging | 100 MB–10 GB |
| Tissue | 100 µm, hours | Histology, spatial omics, MRI | 1–500 GB |
| Organism | 1 cm, days | Behavior video, EEG | 1 GB–10 TB |
| Population | 1 km, years | Camera traps, eDNA, GBIF | 1 MB–10 TB |
| Biosphere | 1000 km, decades | Satellite, climate reanalysis | 1 PB+ |

A practical heuristic: ML compute and data engineering budgets typically dwarf wet-lab budgets above the tissue scale.

## 4.2  File formats you must know

| Format | Purpose | Notes |
|--------|---------|-------|
| FASTA / FASTQ | Sequence + qualities | Text; consider `bgzip` + `tabix`. |
| BAM / CRAM | Aligned reads | CRAM is reference-based and ~30 % smaller. |
| VCF / BCF | Variant calls | Always validate with `bcftools norm`. |
| MTX | Sparse expression | Switch to `.h5ad` for anything > 100 k cells. |
| H5AD / Zarr | AnnData / chunked arrays | First-class for scanpy / squidpy. |
| OME-Zarr | Multi-resolution imaging | Stream from object storage. |
| NWB | Neurophysiology | Standard for DANDI archive. |
| Parquet | Tabular metadata | Columnar, predicate pushdown. |

## 4.3  Schema-aware ingestion

A reproducible pipeline has four layers:

1. **Raw landing zone** — immutable, content-addressed (e.g. SHA-256 of the file).
2. **Validated bronze** — checksum verified, format-validated, lightly enriched with metadata.
3. **Modeled silver** — normalized, joined, aligned to reference coordinates.
4. **Analysis gold** — feature tensors, splits, and labels ready for training.

```python
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa

def land(path: Path, metadata: dict) -> Path:
    """Move a file into a content-addressed landing zone."""
    import hashlib
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    dst = Path("data/raw") / h[:2] / f"{h}_{path.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    path.rename(dst)
    # write a side-car of metadata
    table = pa.table({k: [v] for k, v in {**metadata, "sha256": h}.items()})
    pq.write_table(table, dst.with_suffix(".meta.parquet"))
    return dst
```

## 4.4  Compute and memory back-of-envelope

- 30× whole-genome BAM ≈ 90 GB; CRAM ≈ 60 GB.
- 100 000 cells × 30 000 genes scRNA-seq ≈ 3 GB dense, ≈ 300 MB sparse.
- 1 hour of 1000-channel Neuropixels recording ≈ 60 GB.
- Streaming a 30 000 × 30 000 OME-Zarr image at 4× downsample costs ~1 GB per tile.

If a single sample exceeds laptop RAM, design for *streaming* (Dask, Webdataset) from day one — retrofitting is painful.

## 4.5  Worked example — building an analysis-ready scRNA tensor

```python
import scanpy as sc

adata = sc.read_10x_mtx("data/raw/pbmc3k", var_names="gene_symbols", cache=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
adata = adata[adata.obs["pct_counts_mt"] < 5].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.write_h5ad("data/silver/pbmc3k.h5ad")
```

This file is now ready to feed into any model in Chapters 5, 6, and 9.

## 4.6  Exercises

1. **Storage planner.** Estimate cost of storing one year of nightly whole-genome sequencing for a clinic doing 100 samples / day. Compare S3 Standard vs. Glacier Deep Archive.
2. **Bronze→silver pipeline.** Implement `land()`, `validate()`, and `normalize()` stages for the GTEx v8 RNA-seq dataset. Use Pydantic for schema validation.
3. **Format conversion.** Convert 10× MTX output to AnnData and to Parquet. Compare load times for 100 random gene queries.
4. **Streaming images.** Stream a level-3 view of an OME-Zarr from the IDR (idr.openmicroscopy.org) without downloading the full pyramid.

## 4.7  Further reading

- Marx, V. *The big challenges of big data.* Nature 498 (2013).
- Tarhan, L. *Single-cell portal.* Nat Genet (2023).
- Moore, J. *OME-NGFF.* Nat. Methods (2021).
- Rübel, O. *NWB: Neurodata Without Borders.* eLife (2022).

## See also

- [Chapter 5 — Representation Learning](chapter_05_embeddings.md)
- [Chapter 9 — Single-Cell Intelligence](chapter_09_single_cell.md)
- [Examples — Python](../examples/python.md)
