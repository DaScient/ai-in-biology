"""Ecological modeling endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class SDMRequest(BaseModel):
    """Request body for species distribution modeling."""

    species_name: str = Field(..., description="Scientific name of the species")
    occurrence_points: list[dict] = Field(
        ..., description="List of occurrence points with lat/lon keys"
    )
    environmental_layers: list[str] = Field(
        default_factory=list,
        description="Names of environmental predictor layers to use",
    )


class SDMResponse(BaseModel):
    """Response body for species distribution modeling."""

    suitability_map: list[list[float]] = Field(
        ..., description="Habitat suitability map as a 2-D grid"
    )
    auc_roc: float = Field(..., description="Area under the ROC curve")
    variable_importance: dict = Field(..., description="Importance of each environmental variable")


@router.post("/sdm", response_model=SDMResponse, summary="Species distribution modeling")
async def model_species_distribution(request: SDMRequest) -> SDMResponse:
    """Model the potential distribution of a species using occurrence data.

    Args:
        request: Species occurrence points and environmental layers.

    Returns:
        Predicted habitat suitability map with model performance metrics.
    """
    # Placeholder implementation
    return SDMResponse(
        suitability_map=[[0.5]],
        auc_roc=0.87,
        variable_importance={layer: 0.1 for layer in request.environmental_layers},
    )


class TippingPointRequest(BaseModel):
    """Request body for early warning signal detection."""

    time_series: list[float] = Field(..., description="Ecological time-series data")
    window_size: int = Field(default=50, ge=10, description="Rolling window size for statistics")


class TippingPointResponse(BaseModel):
    """Response body for early warning signal detection."""

    autocorrelation: list[float] = Field(..., description="Rolling autocorrelation values")
    variance: list[float] = Field(..., description="Rolling variance values")
    warning_probability: float = Field(
        ..., description="Probability that a tipping point is approaching (0–1)"
    )


@router.post(
    "/tipping",
    response_model=TippingPointResponse,
    summary="Early warning signal detection",
)
async def detect_tipping_point(request: TippingPointRequest) -> TippingPointResponse:
    """Detect early warning signals of ecological tipping points in a time series.

    Args:
        request: Ecological time-series data and window parameters.

    Returns:
        Rolling statistics and tipping-point warning probability.
    """
    n = len(request.time_series)
    # Placeholder implementation
    autocorrelation = [0.5] * n
    variance = [1.0] * n
    return TippingPointResponse(
        autocorrelation=autocorrelation,
        variance=variance,
        warning_probability=0.25,
    )
