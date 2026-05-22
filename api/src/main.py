"""Main FastAPI application entry point."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.src.endpoints import (
    alpha_fold_interface,
    drug_discovery,
    ecological_modeling,
    sequence_analysis,
    single_cell_embedding,
    structure_prediction,
)

app = FastAPI(
    title="AI in Biological Sciences API",
    description=(
        "RESTful API for biological AI models accompanying the textbook "
        "'AI in Biological Sciences: Theory, Applications, Practice, and Society'"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Register routers
app.include_router(alpha_fold_interface.router, prefix="/api/v1/protein", tags=["protein"])
app.include_router(structure_prediction.router, prefix="/api/v1/protein", tags=["protein"])
app.include_router(sequence_analysis.router, prefix="/api/v1/genomics", tags=["genomics"])
app.include_router(single_cell_embedding.router, prefix="/api/v1/scell", tags=["single-cell"])
app.include_router(drug_discovery.router, prefix="/api/v1/drug", tags=["drug-discovery"])
app.include_router(ecological_modeling.router, prefix="/api/v1/ecology", tags=["ecology"])


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Return API health status."""
    return {"status": "healthy", "version": "0.1.0"}
