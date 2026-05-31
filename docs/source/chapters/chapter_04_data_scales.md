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

### 4.1a  The data hierarchy: a practical guide to scale transitions

The table above places modalities on a scale axis. The decision matrix below adds the
*engineering* dimension — which storage format and compute profile to choose based on
where your data live in the hierarchy.

| Scale | Typical size per sample | Recommended format | Compute profile | Key challenge |
|-------|------------------------|--------------------|-----------------|----------------|
| **Molecular (DNA/RNA)** | 100 MB – 100 GB (FASTQ/BAM) | CRAM (compressed BAM), Parquet for variants | High I/O, moderate compute | Alignment reference bias, indexing |
| **Cellular (scRNA-seq)** | 1 GB – 100 GB (10x MTX, H5AD) | AnnData (H5AD) + Zarr for very large | Moderate I/O, memory-intensive | Sparse matrix operations, batch correction |
| **Tissue (spatial transcriptomics)** | 10–100 GB per slide (image + expression) | OME-Zarr for images, AnnData for expression | GPU for image, CPU for expression | Multi-modal alignment, spot deconvolution |
| **Organism (imaging, MRI)** | 1–100 GB per scan (DICOM, NIfTI) | NIfTI + BIDS, Zarr for cloud | High memory, GPU 3D convs | 3D memory footprint |
| **Population (genomics cohort)** | 10–100 TB (thousands of genomes) | VCF/BCF + cohort-level Parquet | Distributed processing (Spark, Dask) | Privacy, federation |
| **Ecosystem (remote sensing)** | 10 GB – 1 PB per scene | Cloud-optimized GeoTIFF (COG), Zarr | Streaming, tile-based | Cloud masking, temporal gaps |

**Rule of thumb:** If a single sample exceeds laptop RAM (typically 16–32 GB), you must
design for streaming or cloud. Retrofitting streaming into a pipeline built for in-memory
processing is painful.

> See **Exercise 4.6e** to write one-line loaders for each row and identify which would
> fail on a 32 GB laptop.

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

### 4.2a  File format deep dive: CRAM vs. BAM vs. SAM

CRAM achieves higher compression than BAM by referencing a known reference genome and
storing only the differences (conceptually similar to a git delta). For human
whole-genome sequencing (WGS) at 30×, typical sizes are:

- FASTQ: ~90 GB per 30× sample
- BAM (uncompressed): ~90 GB; BAM (compressed, default): ~40 GB
- CRAM (with reference): ~20 GB

**Trade-offs:**

- **BAM:** De facto standard, supported everywhere, random access via indexing. Fixed
  compression ratio.
- **CRAM:** Better compression, lossless by default (can be lossy for quality scores),
  but requires the reference genome file at decompression time. Some older tools do not
  support it.
- **Recommendation:** Store master copies as CRAM to save cost; convert to BAM on the fly
  for compatibility.

```python
# Using pysam to convert and compare sizes
import pysam
import os

def compare_bam_cram(bam_path: str, ref_fasta: str, cram_out: str):
    """Convert BAM to CRAM and report the size ratio."""
    pysam.view("-C", "-T", ref_fasta, "-o", cram_out, bam_path, catch_stdout=False)
    bam_size = os.path.getsize(bam_path)
    cram_size = os.path.getsize(cram_out)
    print(f"BAM: {bam_size/1e9:.2f} GB, CRAM: {cram_size/1e9:.2f} GB, "
          f"ratio: {cram_size/bam_size:.2f}")
    return cram_size / bam_size

# Expected ratio ~0.5 for human WGS
```

**Pitfall:** CRAM files built against different reference genome builds (e.g. GRCh37 vs.
GRCh38) are not interchangeable. Always document the reference used.

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

### 4.3a  Schema-aware ingestion — validation with Pydantic

The `land()` function writes a sidecar metadata Parquet but does not check that the
metadata is well-formed. Adding a **validation layer** with Pydantic catches malformed or
mislabeled files before they enter the pipeline.

```python
import os
from pydantic import BaseModel, Field, field_validator, model_validator

class RNASeqSample(BaseModel):
    sample_id: str = Field(..., pattern=r'^[A-Z0-9]{6,12}$')
    condition: str = Field(..., pattern='^(control|treatment|disease)$')
    read_count: int = Field(..., gt=0, lt=1e9)
    genes_detected: int = Field(..., gt=0)
    percent_ribosomal: float = Field(..., ge=0, le=100)
    fastq_path: str
    metadata_schema_version: str = "1.0"

    @field_validator('fastq_path')
    @classmethod
    def file_exists(cls, v):
        if not os.path.exists(v):
            raise ValueError(f'FASTQ file {v} not found')
        return v

    @model_validator(mode='after')
    def genes_less_than_total(self):
        if self.genes_detected > self.read_count:
            raise ValueError('genes_detected cannot exceed read_count')
        return self
```

Integrate validation right after landing: load the sidecar metadata and reject any file
that fails the schema. This prevents downstream errors from corrupt or mislabeled data.

```python
from pathlib import Path
import pandas as pd

def validate_sample(landed_path: Path, schema_class=RNASeqSample) -> bool:
    """Load metadata from the .meta.parquet sidecar and validate it."""
    meta_path = landed_path.with_suffix(".meta.parquet")
    if not meta_path.exists():
        return False
    df = pd.read_parquet(meta_path)
    try:
        schema_class(**df.iloc[0].to_dict())
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False
```

> See **Exercise 4.6b** to write a Pydantic schema for a VCF row.

## 4.4  Compute and memory back-of-envelope

- 30× whole-genome BAM ≈ 90 GB; CRAM ≈ 60 GB.
- 100 000 cells × 30 000 genes scRNA-seq ≈ 3 GB dense, ≈ 300 MB sparse.
- 1 hour of 1000-channel Neuropixels recording ≈ 60 GB.
- Streaming a 30 000 × 30 000 OME-Zarr image at 4× downsample costs ~1 GB per tile.

If a single sample exceeds laptop RAM, design for *streaming* (Dask, Webdataset) from day one — retrofitting is painful.

### 4.4a  Streaming strategies — deciding when and how

The per-sample sizes above tell you *how big* the data are; the rule below tells you
*whether you need to stream*. Let \( R \) = available RAM (GB), \( S \) = single sample
size (GB):

- If \( S \leq 0.5 \times R \): in-memory is safe.
- If \( 0.5 \times R < S \leq R \): in-memory is possible but may cause swapping — use
  chunked processing.
- If \( S > R \): you must stream.

Tabular expression data larger than RAM can be processed lazily with Dask, or in chunks
with pandas:

```python
import dask.dataframe as dd
import pandas as pd

# Load a dataset larger than RAM (lazy)
df = dd.read_parquet('data/silver/expression_large/*.parquet')
mean_expr = df.groupby('gene_id')['tpm'].mean().compute()  # triggers computation

# Alternatively, iterate in chunks with pandas
chunk_size = 10000
for chunk in pd.read_csv('large_matrix.csv', chunksize=chunk_size):
    process(chunk)  # aggregate, write partial results
```

For imaging data, use chunked storage (Zarr) backed by `dask.array` and process tile by
tile:

```python
import zarr
import dask.array as da

zarr_array = zarr.open('data/raw/image.zarr', mode='r')
dask_array = da.from_zarr(zarr_array, chunks=(1024, 1024, 1))
# Apply a function to each chunk lazily, then trigger computation tile by tile.
result = dask_array.map_blocks(compute_on_tile)
result.compute()
```

> See **Exercise 4.6c** to compute a column-wise mean over a 50 GB `numpy.memmap` without
> loading the full array.

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

### 4.5a  Worked example extension — multi-modal data integration

The example above builds a single-modality scRNA tensor. A common next step is to
**integrate scRNA-seq and spatial transcriptomics** from the same tissue (e.g. mouse
brain):

1. Load Visium spatial data (gene expression per spot + H&E image).
2. Load a matched single-cell reference (scRNA-seq) from the same region.
3. Use a probabilistic model (e.g. Cell2location, RCTD) to deconvolve each Visium spot
   into cell type proportions.
4. Train a graph convolutional network (GCN) on the spatial graph (spots as nodes,
   neighbors as edges) to predict cell type composition from image patches.

```python
import scanpy as sc
import squidpy as sq

# Load spatial data and the matched single-cell reference
adata_spatial = sq.read.visium('path/to/spatial/')
adata_ref = sc.read_h5ad('path/to/scRNA.h5ad')

# Deconvolution using Cell2location (simplified)
from cell2location import run_cell2location
run_cell2location(adata_spatial, adata_ref)

# adata_spatial.obsm['q05_cell_abundance_w_sf'] now holds cell type proportions per spot.
# Build the spatial graph, then train a GNN (e.g. PyTorch Geometric) to predict
# proportions from image patches and neighbors.
sq.gr.spatial_neighbors(adata_spatial, coord_type='grid')
```

**Pitfall:** Spatial resolution is coarser than single-cell. A single Visium spot (55 µm)
contains ~10–50 cells, so deconvolution is inherently uncertain — report credible
intervals rather than point estimates.

## 4.6  Exercises

1. **Storage planner.** Estimate cost of storing one year of nightly whole-genome sequencing for a clinic doing 100 samples / day. Compare S3 Standard vs. Glacier Deep Archive.
2. **Bronze→silver pipeline.** Implement `land()`, `validate()`, and `normalize()` stages for the GTEx v8 RNA-seq dataset. Use Pydantic for schema validation.
3. **Format conversion.** Convert 10× MTX output to AnnData and to Parquet. Compare load times for 100 random gene queries.
4. **Streaming images.** Stream a level-3 view of an OME-Zarr from the IDR (idr.openmicroscopy.org) without downloading the full pyramid.

**Extensions**

- **4.6b. VCF row schema.** Write a Pydantic schema for a VCF row (CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO). Validate that ALT is not equal to REF and that POS is a positive integer.
- **4.6c. Out-of-core mean.** Simulate a 50 GB dataset using `numpy.memmap`. Write a function that computes the column-wise mean without loading the entire array into memory, and compare its time to an in-memory version on a smaller subset.
- **4.6e. Per-scale loaders.** For each row of the decision matrix in §4.1a, write a one-line Python pseudo-code using the appropriate library to load a single sample and compute a basic statistic (mean expression, total reads, etc.). Identify which would fail on a laptop with 32 GB RAM.
- **4.6h. Benchmark compressed formats.** Download a small BAM file (e.g. from ENA). Convert it to CRAM, SAM (uncompressed), and a block-compressed BAM (using `samtools collate`). Compare the file sizes and the `samtools idxstats` time.
- **4.6i. Data validation suite.** For a given dataset (choose GTEx RNA-seq, 10x scRNA, or a VCF), write a function that checks: required columns present; data types correct; no missing values in critical fields; values within plausible biological ranges (e.g. expression > 0, valid chromosome names); and no duplicate sample IDs.
- **4.6j. Memory bottleneck.** Generate a random matrix of size (200000, 100) (~160 MB) and compute its SVD with `numpy.linalg.svd` (which loads the full matrix). Then compute the same SVD with `scipy.sparse.linalg.svds` on a sparse random matrix of the same dimensions but 1% density. Report memory usage and time.

## 4.7  Further reading

- Marx, V. *The big challenges of big data.* Nature 498 (2013).
- Tarhan, L. *Single-cell portal.* Nat Genet (2023).
- Moore, J. *OME-NGFF.* Nat. Methods (2021).
- Rübel, O. *NWB: Neurodata Without Borders.* eLife (2022).
- Reiner, B. et al. *Zarr for large-scale biological imaging: a practical guide.* Nat. Methods (2023) — includes comparisons with HDF5.
- The HDF5 Group. *HDF5 vs. Zarr: a technical comparison* (2024) — independent benchmark.
- Lam, S. et al. *Cloud-optimized genomics: CRAM, index files, and object storage.* GigaScience (2022).
- Poldrack, R. A. *Data management for reproducible neuroimaging: BIDS and beyond.* Neuron (2023) — applicable to NWB.

## See also

- [Chapter 5 — Representation Learning](chapter_05_embeddings.md)
- [Chapter 9 — Single-Cell Intelligence](chapter_09_single_cell.md)
- [Examples — Python](../examples/python.md)
