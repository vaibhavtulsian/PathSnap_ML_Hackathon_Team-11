from typing import List, Optional
from pydantic import BaseModel, Field

class TripListResponse(BaseModel):
    trips: List[str] = Field(..., description="List of reserved evaluation trip IDs")
    count: int = Field(..., description="Total number of available evaluation trips")

class RoutePredictionResponse(BaseModel):
    trip_id: str
    status: str = "success"
    predicted_arcs: List[int] = Field(..., description="Chronological sequence of predicted arc IDs")
    accuracy_score: float = Field(..., description="Route segment match accuracy against ground truth (%)")
    pointwise_accuracy: float = Field(..., description="Point-level matching accuracy (%)")
    route_continuity: float = Field(..., description="Percentage of consecutive arcs that share a junction node (%)")
    inference_time_ms: float = Field(..., description="Inference and decoding execution time in milliseconds")
    gps_point_count: int = Field(..., description="Total number of raw GPS observations in trip")
    predicted_arc_count: int = Field(..., description="Total number of arcs in predicted reconstructed route")
    gps_coordinates: List[List[float]] = Field(..., description="List of [lat, lon] pairs for raw GPS points")
    predicted_path_coords: List[List[float]] = Field(..., description="List of [lat, lon] coordinates representing the reconstructed route polyline")

class HealthResponse(BaseModel):
    status: str
    version: str
    system: str
