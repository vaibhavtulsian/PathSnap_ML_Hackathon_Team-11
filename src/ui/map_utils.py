from typing import List, Optional
import folium

def generate_folium_map(
    gps_coords: List[List[float]],
    predicted_path_coords: List[List[float]],
    trip_id: str
) -> folium.Map:
    """
    Creates an interactive Folium map adhering to Section 6 of PathSnap guidelines:
    1. OpenStreetMap tiles (standard, free, no API key required).
    2. Distinguishable colors: Blue dots for GPS, Green polyline for Reconstructed route.
    3. Start/End green & red pin markers.
    4. Subsampled GPS dots if > 500 points to keep map fast and responsive.
    5. Clear layer control and auto-adjusted zoom bounds.
    """
    if not gps_coords and not predicted_path_coords:
        return folium.Map(location=[54.73, 55.97], zoom_start=13, tiles="OpenStreetMap")

    # Center location
    ref_coords = gps_coords if gps_coords else predicted_path_coords
    avg_lat = sum(p[0] for p in ref_coords) / len(ref_coords)
    avg_lon = sum(p[1] for p in ref_coords) / len(ref_coords)

    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    # 1. Raw GPS Observations Layer (subsample if large to prevent browser DOM lag)
    gps_layer = folium.FeatureGroup(name="Raw GPS Observations (Noisy)", show=True)
    step = max(1, len(gps_coords) // 300)  # Up to 300 crisp points for fast rendering
    subsampled_gps = gps_coords[::step]
    
    for idx, pt in enumerate(subsampled_gps):
        folium.CircleMarker(
            location=pt,
            radius=3.5,
            color="#1F77B4",
            fill=True,
            fill_color="#1F77B4",
            fill_opacity=0.75,
            tooltip=f"GPS Obs ({pt[0]:.5f}, {pt[1]:.5f})"
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    # 2. Predicted Reconstructed Route Layer
    if predicted_path_coords:
        route_layer = folium.FeatureGroup(name="Reconstructed Route (PathSnap)", show=True)
        folium.PolyLine(
            locations=predicted_path_coords,
            weight=5,
            color="#2CA02C",
            opacity=0.9,
            tooltip=f"Predicted Route for Trip {trip_id}"
        ).add_to(route_layer)
        route_layer.add_to(m)

        # Start Marker (Green Pin)
        folium.Marker(
            location=predicted_path_coords[0],
            popup=f"Trip {trip_id} Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)

        # End Marker (Red Pin)
        folium.Marker(
            location=predicted_path_coords[-1],
            popup=f"Trip {trip_id} Destination",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)

    # Add Layer Control
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # Fit Map to Bounds
    all_points = (gps_coords or []) + (predicted_path_coords or [])
    if all_points:
        min_lat = min(p[0] for p in all_points)
        max_lat = max(p[0] for p in all_points)
        min_lon = min(p[1] for p in all_points)
        max_lon = max(p[1] for p in all_points)
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(20, 20))

    return m
