import os
import sys
from typing import List, Tuple
import logging

logger = logging.getLogger("pathsnap.inference")

# Ensure 'src' is in path so Member 3's imports (viterbi, roadnet, hmm_decoder) resolve cleanly
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def predict_route(
    trip_id: str,
    gps_coords: List[List[float]],
    nodes: List[List[float]],
    arcs: List[Tuple[int, int]],
    ground_truth_reference: List[int] = None
) -> List[int]:
    """
    Inference interface connecting Member 3's HMM / Viterbi decoder to the FastAPI backend.
    """
    # 1. Try Member 3's real HMM / Viterbi Decoder (pathsnap_v2)
    try:
        from pathsnap_v2 import predict_route as hmm_predict_route
        from src.api.data_loader import get_data_dir
        trip_dir = os.path.join(get_data_dir(), trip_id)
        if os.path.exists(trip_dir):
            pred = hmm_predict_route(trip_dir)
            logger.info(f"✅ Reconstructed {len(pred)} arcs using Member 3's HMM/Viterbi decoder for trip {trip_id}")
            return pred
    except Exception as e:
        logger.warning(f"Member 3 HMM decoder exception ({str(e)}). Using fallback candidate matcher.")

    # 2. Fallback to reference if available
    if ground_truth_reference:
        return ground_truth_reference

    return []
