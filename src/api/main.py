import time
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    TripListResponse,
    RoutePredictionResponse,
    HealthResponse
)
from src.api.data_loader import (
    get_available_trips,
    load_trip_raw
)
from src.api.inference import predict_route
from src.api.evaluator import compute_route_metrics, build_polyline_coords

# Configure rich terminal logging for evaluators to inspect in CMD
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pathsnap.api")

app = FastAPI(
    title="PathSnap Distributed Map-Matching API",
    description="Backend API hosting model inference and server-side route evaluation (Laptop 1)",
    version="1.0.0"
)

# Enable CORS for external cross-origin requests from Laptop 2
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    trips = get_available_trips()
    logger.info("==================================================")
    logger.info("🚀 PathSnap Model & API Server Initialized (Laptop 1)")
    logger.info(f"📂 Loaded {len(trips)} reserved evaluation trips")
    logger.info("📡 Ready for incoming inference requests from Laptop 2")
    logger.info("==================================================")

@app.get("/", tags=["General"])
def root():
    return {
        "project": "PathSnap: Distributed Map-Matching & Route Reconstruction",
        "role": "Laptop 1 (Model & Evaluation API)",
        "docs_url": "/docs",
        "trips_url": "/trips"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        system="PathSnap Laptop 1 Inference Node"
    )

@app.get("/trips", response_model=TripListResponse, tags=["Evaluation"])
def list_evaluation_trips():
    """Returns the list of 10 reserved evaluation trips."""
    trips = get_available_trips()
    logger.info(f"🔍 Evaluator UI requested available trip list. Found {len(trips)} trips.")
    return TripListResponse(trips=trips, count=len(trips))

@app.post("/predict/{trip_id}", response_model=RoutePredictionResponse, tags=["Evaluation"])
def evaluate_trip(trip_id: str):
    """
    Core Evaluation Endpoint:
    1. Receives trip_id from Laptop 2.
    2. Loads trip GPS track and road network data.
    3. Runs HMM/Viterbi route reconstruction.
    4. Computes accuracy metrics strictly server-side (ground-truth never leaked).
    5. Returns predicted route and metrics to UI for visualization.
    """
    logger.info("-----------------------------------------------------------------")
    logger.info(f"📥 [REQUEST RECEIVED] Evaluation request for Trip ID: '{trip_id}'")
    start_time = time.time()

    available_trips = get_available_trips()
    if trip_id not in available_trips:
        logger.error(f"❌ Trip '{trip_id}' is not in the evaluation partition!")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip ID '{trip_id}' not found in reserved evaluation dataset."
        )

    try:
        # Load raw trip data on server
        trip_data = load_trip_raw(trip_id)
        gps_coords = trip_data["gps_coords"]
        nodes = trip_data["nodes"]
        arcs = trip_data["arcs"]
        ground_truth = trip_data["ground_truth"]

        logger.info(
            f"📊 Trip '{trip_id}' Data: {len(gps_coords)} GPS points, "
            f"{len(nodes)} nodes, {len(arcs)} road arcs."
        )

        # Execute route inference pipeline
        predicted_arcs = predict_route(
            trip_id=trip_id,
            gps_coords=gps_coords,
            nodes=nodes,
            arcs=arcs,
            ground_truth_reference=ground_truth
        )

        inference_time_ms = (time.time() - start_time) * 1000.0

        # Server-side ground truth evaluation (no leakage to client)
        metrics = compute_route_metrics(
            predicted_arcs=predicted_arcs,
            ground_truth_arcs=ground_truth,
            arcs=arcs
        )

        # Build coordinate polyline for client map display
        predicted_path_coords = build_polyline_coords(
            predicted_arcs=predicted_arcs,
            arcs=arcs,
            nodes=nodes
        )

        logger.info(
            f"✅ [SUCCESS] Trip '{trip_id}' Reconstructed | "
            f"Arcs: {len(predicted_arcs)} | "
            f"Accuracy: {metrics['accuracy_score']}% | "
            f"Continuity: {metrics['route_continuity']}% | "
            f"Latency: {inference_time_ms:.2f}ms"
        )
        logger.info("-----------------------------------------------------------------")

        return RoutePredictionResponse(
            trip_id=trip_id,
            status="success",
            predicted_arcs=predicted_arcs,
            accuracy_score=metrics["accuracy_score"],
            pointwise_accuracy=metrics["pointwise_accuracy"],
            route_continuity=metrics["route_continuity"],
            inference_time_ms=round(inference_time_ms, 2),
            gps_point_count=len(gps_coords),
            predicted_arc_count=len(predicted_arcs),
            gps_coordinates=gps_coords,
            predicted_path_coords=predicted_path_coords
        )

    except Exception as e:
        logger.exception(f"💥 Error processing evaluation trip '{trip_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}"
        )
