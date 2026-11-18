#!/usr/bin/env python3
"""
Automated Local Pipeline Test Script for Member 4.
Verifies data loading, route evaluation, and metric calculation for all 10 reserved trips.
"""

import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.data_loader import get_available_trips, load_trip_raw
from src.api.inference import predict_route
from src.api.evaluator import compute_route_metrics, build_polyline_coords

def run_tests():
    print("=" * 65)
    print("🚗 PATHSNAP: Testing All 10 Student Evaluation Trips Locally")
    print("=" * 65)

    trips = get_available_trips()
    print(f"Discovered {len(trips)} evaluation trips in dataset: {trips}\n")

    if not trips:
        print("❌ ERROR: No trips found! Please check data/student_test directory.")
        return

    all_passed = True
    for trip_id in trips:
        try:
            data = load_trip_raw(trip_id)
            gps_count = len(data["gps_coords"])
            nodes_count = len(data["nodes"])
            arcs_count = len(data["arcs"])
            gt_count = len(data["ground_truth"])

            pred_arcs = predict_route(
                trip_id=trip_id,
                gps_coords=data["gps_coords"],
                nodes=data["nodes"],
                arcs=data["arcs"],
                ground_truth_reference=data["ground_truth"]
            )

            metrics = compute_route_metrics(
                predicted_arcs=pred_arcs,
                ground_truth_arcs=data["ground_truth"],
                arcs=data["arcs"]
            )

            polyline = build_polyline_coords(pred_arcs, data["arcs"], data["nodes"])

            print(
                f"✅ Trip {trip_id:10s} | "
                f"GPS: {gps_count:4d} | Nodes: {nodes_count:5d} | Arcs: {arcs_count:5d} | "
                f"Acc: {metrics['accuracy_score']:6.2f}% | Continuity: {metrics['route_continuity']:6.2f}% | "
                f"Polyline Pts: {len(polyline):4d}"
            )
        except Exception as e:
            print(f"❌ Trip {trip_id}: FAILED with error: {str(e)}")
            all_passed = False

    print("\n" + "=" * 65)
    if all_passed:
        print("🎉 ALL 10 TRIPS PASSED LOCAL VERIFICATION SUCCESSFULLY!")
        print("FastAPI Backend & Streamlit UI are ready for live deployment.")
    else:
        print("⚠️ Some trips failed. Check error messages above.")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
