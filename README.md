# AI in Biological Sciences

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://dascient.github.io/ai-in-biology)
[![Tests](https://github.com/DaScient/ai-in-biology/actions/workflows/tests.yml/badge.svg)](https://github.com/DaScient/ai-in-biology/actions/workflows/tests.yml)
[![API Status](https://img.shields.io/website?url=https%3A%2F%2Fapi.bioai.dascient.com%2Fhealth)](https://api.bioai.dascient.com/docs)

> *"Biology is fundamentally an information science. AI provides the computational syntax to decode life's language."* — Dr. Aris Thorne

## About This Repository

This repository is the living digital companion to the textbook **"AI in Biological Sciences: Theory, Applications, Practice, and Society"** (DaScient Press, 2025). It contains:

- **Interactive Jupyter Notebooks** for each chapter
- **RESTful API** for biological AI models
- **Pre-trained models** for protein structure, genomics, single-cell analysis
- **Benchmark datasets** and evaluation scripts
- **Docker containers** for reproducible execution
- **Cloud deployment** templates (AWS, GCP, Azure)

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/DaScient/ai-in-biology.git
cd ai-in-biology

# Set up conda environment
conda create -n ai-bio python=3.11
conda activate ai-bio

# Install package
pip install -e .

# Launch Jupyter
jupyter notebook
```

### Docker Quick Start

```bash
docker-compose up -d
# API available at http://localhost:8000
# Jupyter at http://localhost:8888
```

## Textbook Chapters

| Part | Chapter | Notebook |
|------|---------|----------|
| I | 1. Biology as an Information Science | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_01_bioinfo_basics.ipynb) |
| I | 2. AI: Concepts, Paradigms, Evolution | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_02_dna_language_models.ipynb) |
| I | 3. Mathematical Foundations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_03_attention_in_genomics.ipynb) |
| II | 4. Biological Data Across Scales | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_04_data_scales.ipynb) |
| III | 5. Representation Learning for Life | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_05_embeddings.ipynb) |
| III | 6. Modeling Living Systems | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_06_modeling.ipynb) |
| III | 7. Genomics & Gene Regulation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_07_genomics.ipynb) |
| III | 8. Protein Structure & Design | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_08_protein.ipynb) |
| III | 9. Cellular Systems & Single-Cell | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_09_single_cell.ipynb) |
| IV | 10. Development & Morphogenesis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_10_development.ipynb) |
| IV | 11. Neuroscience & Cognitive Bio | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_11_neuroscience.ipynb) |
| IV | 12. Behavior & Social Systems | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_12_ethology.ipynb) |
| V | 13. Evolutionary Dynamics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_13_evolution.ipynb) |
| V | 14. Ecology & Conservation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_14_ecology.ipynb) |
| V | 15. Earth Systems & Planetary Bio | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_15_earth.ipynb) |
| VI | 16. Medicine & Healthcare | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_16_medicine.ipynb) |
| VI | 17. Biotechnology & Bioengineering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_17_biotech.ipynb) |
| VI | 18. Experimental Design | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_18_experiments.ipynb) |
| VII | 19. Ethics of AI in Bio | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_19_ethics.ipynb) |
| VII | 20. Policy & Regulation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_20_policy.ipynb) |
| VII | 21. Societal Transformation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_21_society.ipynb) |
| VIII | 22. Co-Evolution of AI & Life | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_22_coevolution.ipynb) |
| VIII | 23. Limits & Open Questions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_23_limits.ipynb) |
| VIII | 24. A New Biology | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/notebooks/chapter_24_new_biology.ipynb) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/protein/fold` | POST | Predict protein structure |
| `/api/v1/protein/design` | POST | De novo protein design |
| `/api/v1/genomics/variant` | POST | Variant effect prediction |
| `/api/v1/genomics/regulatory` | POST | Predict regulatory elements |
| `/api/v1/scell/embed` | POST | Single-cell embedding |
| `/api/v1/scell/trajectory` | POST | Developmental trajectory |
| `/api/v1/ecology/sdm` | POST | Species distribution modeling |
| `/api/v1/ecology/tipping` | POST | Early warning signals |
| `/api/v1/clinical/diagnosis` | POST | Medical image diagnosis |
| `/api/v1/clinical/prognosis` | POST | Survival prediction |
| `/api/v1/drug/docking` | POST | Molecular docking |
| `/api/v1/drug/generate` | POST | De novo drug design |
| `/api/v1/literature/hypothesis` | POST | Hypothesis generation |
| `/api/v1/ethics/audit` | POST | Model bias auditing |

## Pre-trained Models Available

| Model | Description | Size | Download |
|-------|-------------|------|----------|
| BioBERT-base | Biomedical language model | 110M | [Link](https://huggingface.co/dascient/biobert-base) |
| ProtBERT | Protein language model | 420M | [Link](https://huggingface.co/dascient/protbert) |
| scGPT-base | Single-cell foundation model | 350M | [Link](https://huggingface.co/dascient/scgpt-base) |
| EcoGNN | Ecological network model | 85M | [Link](https://huggingface.co/dascient/ecognn) |
| Clinformer | Clinical prediction transformer | 250M | [Link](https://huggingface.co/dascient/clinformer) |

## Benchmark Datasets

- **Protein Folding**: CASP15, CAMEO, PDB
- **Variant Calling**: GIAB v4.2, Platinum Genomes
- **Single-Cell**: Human Cell Atlas, Tabula Sapiens
- **Ecology**: GBIF, iNaturalist 2023
- **Clinical**: MIMIC-IV, UK Biobank (subset)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

- **Report bugs**: Open GitHub issue
- **Suggest features**: Use feature request template
- **Submit code**: Fork → branch → PR
- **Improve docs**: Edit `/docs/source/`

## Citation

```bibtex
@book{thorne2025aibio,
    title = {AI in Biological Sciences: Theory, Applications, Practice, and Society},
    author = {Thorne, Aris and Chen, Wei and Adebayo, Marcus},
    year = {2025},
    publisher = {DaScient Press},
    series = {DaScient Intelligence Academy Textbook Series},
    isbn = {978-1-934293-74-9}
}
```

## License

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

**Commercial licensing** available. Contact commercial@dascient.com

## Acknowledgments

- **DaScient Intelligence Academy** for textbook development
- **OpenAI** for GPT API access
- **Meta** for ESM models
- **DeepMind** for AlphaFold2 open release
- **Contributors**: 20+ researchers, leaders, and educators worldwide

## Contact

- **Technical Support**: support@dascient.com
- **API Access**: api@dascient.com
- **Textbook Inquiries**: press@dascient.com
- **LinkedIn**: [DaScient](https://linkedin.com/company/dascient/)

---

*"The future of life on this planet will be shaped by how well you wield this understanding."* — Dr. Aris Thorne
