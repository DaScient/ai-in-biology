"""Structure prediction endpoints (general, beyond AlphaFold)."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class RNAFoldRequest(BaseModel):
    """Request body for RNA secondary structure prediction."""

    sequence: str = Field(..., description="RNA sequence (A, U, G, C)")


class RNAFoldResponse(BaseModel):
    """Response body for RNA secondary structure prediction."""

    dot_bracket: str = Field(..., description="Secondary structure in dot-bracket notation")
    mfe: float = Field(..., description="Minimum free energy (kcal/mol)")


@router.post(
    "/rna-fold",
    response_model=RNAFoldResponse,
    summary="RNA secondary structure prediction",
)
async def predict_rna_structure(request: RNAFoldRequest) -> RNAFoldResponse:
    """Predict the secondary structure of an RNA sequence.

    Args:
        request: RNA sequence to fold.

    Returns:
        Secondary structure in dot-bracket notation with MFE.
    """
    # Placeholder implementation
    dot_bracket = "." * len(request.sequence)
    return RNAFoldResponse(dot_bracket=dot_bracket, mfe=-10.5)
