from typing import List, Tuple, Dict, Any

def compute_route_metrics(
    predicted_arcs: List[int],
    ground_truth_arcs: List[int],
    arcs: List[Tuple[int, int]]
) -> Dict[str, float]:
    """
    Computes server-side evaluation metrics strictly on Laptop 1:
    - Route accuracy (% of ground truth arcs covered in correct sequence/set)
    - Pointwise accuracy
    - Route continuity (% of consecutive predicted arcs that connect in the road network)
    """
    if not predicted_arcs or not ground_truth_arcs:
        return {
            "accuracy_score": 0.0,
            "pointwise_accuracy": 0.0,
            "route_continuity": 0.0
        }

    # 1. Element-wise / Sequence Alignment Match
    min_len = min(len(predicted_arcs), len(ground_truth_arcs))
    exact_matches = sum(
        1 for i in range(min_len) if predicted_arcs[i] == ground_truth_arcs[i]
    )
    pointwise_acc = (exact_matches / max(len(ground_truth_arcs), 1)) * 100.0

    # 2. Overall Route Overlap (Set / Coverage)
    gt_set = set(ground_truth_arcs)
    pred_set = set(predicted_arcs)
    overlap = len(gt_set.intersection(pred_set))
    accuracy_score = (overlap / max(len(gt_set), 1)) * 100.0

    # 3. Route Continuity (Road graph topological connectivity)
    valid_transitions = 0
    total_transitions = max(len(predicted_arcs) - 1, 1)

    for i in range(len(predicted_arcs) - 1):
        idx1 = predicted_arcs[i]
        idx2 = predicted_arcs[i + 1]

        if idx1 < len(arcs) and idx2 < len(arcs):
            u1, v1 = arcs[idx1]
            u2, v2 = arcs[idx2]

            # In a connected path, endpoint of arc 1 touches start/end of arc 2
            if v1 == u2 or u1 == v2 or v1 == v2 or u1 == u2:
                valid_transitions += 1

    continuity_pct = (valid_transitions / total_transitions) * 100.0

    return {
        "accuracy_score": round(accuracy_score, 2),
        "pointwise_accuracy": round(pointwise_acc, 2),
        "route_continuity": round(continuity_pct, 2),
    }

def build_polyline_coords(
    predicted_arcs: List[int],
    arcs: List[Tuple[int, int]],
    nodes: List[List[float]]
) -> List[List[float]]:
    """
    Converts a sequence of arc IDs into geographic [lat, lon] coordinates
    for map rendering.
    """
    coords: List[List[float]] = []
    for arc_id in predicted_arcs:
        if arc_id < len(arcs):
            u, v = arcs[arc_id]
            if u < len(nodes) and v < len(nodes):
                coords.append(nodes[u])
                coords.append(nodes[v])
    return coords
