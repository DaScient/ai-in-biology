"""Unit tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.src.main import app

client = TestClient(app)


def test_health_check():
    """Health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_protein_fold_returns_200():
    """Protein folding endpoint returns a valid response."""
    payload = {"sequence": "MKTIIALSYIFCLVFA"}
    response = client.post("/api/v1/protein/fold", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "pdb_string" in data
    assert "plddt_scores" in data
    assert "mean_plddt" in data
    assert len(data["plddt_scores"]) == len(payload["sequence"])


def test_protein_design_returns_200():
    """Protein design endpoint returns a valid response."""
    payload = {"target_function": "bind ATP", "length": 50}
    response = client.post("/api/v1/protein/design", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "sequence" in data
    assert len(data["sequence"]) == 50
    assert 0.0 <= data["confidence"] <= 1.0


def test_variant_effect_returns_200():
    """Variant effect prediction endpoint returns a valid response."""
    payload = {
        "reference_sequence": "ATCGATCGATCG",
        "variant_position": 5,
        "reference_allele": "A",
        "alternate_allele": "T",
    }
    response = client.post("/api/v1/genomics/variant", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "pathogenicity_score" in data
    assert "effect_class" in data


def test_scell_embed_returns_200():
    """Single-cell embedding endpoint returns a valid response."""
    payload = {
        "expression_matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "gene_names": ["GENE1", "GENE2", "GENE3"],
    }
    response = client.post("/api/v1/scell/embed", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2


def test_ecology_sdm_returns_200():
    """Species distribution modeling endpoint returns a valid response."""
    payload = {
        "species_name": "Panthera leo",
        "occurrence_points": [{"lat": 1.0, "lon": 2.0}],
        "environmental_layers": ["temperature", "precipitation"],
    }
    response = client.post("/api/v1/ecology/sdm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "suitability_map" in data
    assert "auc_roc" in data


def test_drug_docking_returns_200():
    """Molecular docking endpoint returns a valid response."""
    payload = {
        "protein_pdb": "ATOM  ...",
        "ligand_smiles": "CCO",
    }
    response = client.post("/api/v1/drug/docking", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "binding_affinity" in data
    assert "confidence" in data
