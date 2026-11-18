import os
from typing import List, Tuple, Dict, Any

DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "student_test")
)

FALLBACK_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ML Hackathon", "Student folder", "student_test")
)

def get_data_dir() -> str:
    if os.path.exists(DEFAULT_DATA_DIR) and len(os.listdir(DEFAULT_DATA_DIR)) > 1:
        return DEFAULT_DATA_DIR
    elif os.path.exists(FALLBACK_DATA_DIR):
        return FALLBACK_DATA_DIR
    return DEFAULT_DATA_DIR

def get_available_trips() -> List[str]:
    data_dir = get_data_dir()
    if not os.path.exists(data_dir):
        return []
    trips = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ]
    return sorted(trips)

def load_trip_raw(trip_id: str) -> Dict[str, Any]:
    """
    Loads raw files for a given trip_id:
    - .track: GPS points (lat, lon, time)
    - .nodes: Road network nodes (lat, lon) indexed by line number
    - .arcs: Road network edges (from_node, to_node) indexed by line number
    - .route: Ground-truth sequence of arc IDs (server-side reference)
    """
    data_dir = get_data_dir()
    trip_path = os.path.join(data_dir, trip_id)
    if not os.path.exists(trip_path):
        raise FileNotFoundError(f"Trip directory not found: {trip_path}")

    # 1. Parse .track
    # Raw file format: col 0 = Longitude (X), col 1 = Latitude (Y), col 2 = Timestamp
    # Folium/Leaflet expects [latitude, longitude], so we store [lat, lon]
    track_file = os.path.join(trip_path, f"{trip_id}.track")
    gps_coords: List[List[float]] = []
    timestamps: List[float] = []
    if os.path.exists(track_file):
        with open(track_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    gps_coords.append([lat, lon])
                    if len(parts) >= 3:
                        timestamps.append(float(parts[2]))

    # 2. Parse .nodes (0-based row index is node_id)
    # Raw file format: col 0 = Longitude (X), col 1 = Latitude (Y)
    # Stored as [lat, lon] for Leaflet/Folium polyline rendering
    nodes_file = os.path.join(trip_path, f"{trip_id}.nodes")
    nodes: List[List[float]] = []
    if os.path.exists(nodes_file):
        with open(nodes_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    nodes.append([lat, lon])

    # 3. Parse .arcs (0-based row index is arc_id)
    arcs_file = os.path.join(trip_path, f"{trip_id}.arcs")
    arcs: List[Tuple[int, int]] = []
    if os.path.exists(arcs_file):
        with open(arcs_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    arcs.append((int(parts[0]), int(parts[1])))

    # 4. Parse .route (server-side ground truth)
    route_file = os.path.join(trip_path, f"{trip_id}.route")
    ground_truth: List[int] = []
    if os.path.exists(route_file):
        with open(route_file, "r") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    ground_truth.append(int(line_str))

    return {
        "trip_id": trip_id,
        "gps_coords": gps_coords,
        "timestamps": timestamps,
        "nodes": nodes,
        "arcs": arcs,
        "ground_truth": ground_truth,
    }
