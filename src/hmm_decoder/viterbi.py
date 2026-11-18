"""
PathSnap: Hidden Markov Model (HMM) & Viterbi Route Reconstruction.
Owned by Member 3.
"""

from typing import List, Tuple
import logging
import numpy as np

logger = logging.getLogger("pathsnap.hmm")


def decode_trip(
    trip_id: str,
    gps_coords: List[List[float]],
    nodes: List[List[float]],
    arcs: List[Tuple[int, int]]
) -> List[int]:

    """
    Reconstructs the optimal sequence of road segments from noisy GPS observations
    using an HMM-based Viterbi decoder.
    """

    try:
        from pathsnap_v2 import decode2
        from shapely.geometry import LineString, Point
        from shapely.strtree import STRtree
        from pyproj import Transformer

        # --------------------------------------------------
        # 1. Convert coordinates to projected coordinates
        # --------------------------------------------------
        transformer = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:32630",
            always_xy=True
        )

        nodes = np.asarray(nodes, dtype=float)
        gps_coords = np.asarray(gps_coords, dtype=float)
        arcs = np.asarray(arcs, dtype=int)

        # Coordinate ordering check: EPSG:4326 with always_xy=True expects (lon, lat)
        # Latitude is bounded in [-90, 90]. If col 0 has values > 90 or < -90, col 0 is lon.
        if len(nodes) > 0 and (np.any(np.abs(nodes[:, 0]) > 90) or np.any(np.abs(gps_coords[:, 0]) > 90)):
            node_lon, node_lat = nodes[:, 0], nodes[:, 1]
            gps_lon, gps_lat = gps_coords[:, 0], gps_coords[:, 1]
        else:
            node_lat, node_lon = nodes[:, 0], nodes[:, 1]
            gps_lat, gps_lon = gps_coords[:, 0], gps_coords[:, 1]

        nx, ny = transformer.transform(node_lon, node_lat)
        nodes_projected = np.column_stack((nx, ny))

        gx, gy = transformer.transform(gps_lon, gps_lat)
        gps_projected = np.column_stack((gx, gy))

        # --------------------------------------------------
        # 2. Create adapter for your existing HMM decoder
        # --------------------------------------------------
        class TripAdapter:

            def __init__(self):
                self.nodes = nodes_projected
                self.arcs = arcs
                self.pts = gps_projected

                self.t = np.arange(len(gps_coords)) * 10

                self.geoms = [
                    LineString([
                        self.nodes[a],
                        self.nodes[b]
                    ])
                    for a, b in self.arcs
                ]

                self.tree = STRtree(self.geoms)

                by_from = {}

                for j, (a, b) in enumerate(self.arcs):
                    by_from.setdefault(int(a), []).append(j)

                self.succ = {
                    i: by_from.get(int(b), [])
                    for i, (_, b) in enumerate(self.arcs)
                }

            def arc_bearing(self, i):

                a, b = self.arcs[i]

                x1, y1 = self.nodes[a]
                x2, y2 = self.nodes[b]

                return np.arctan2(x2 - x1, y2 - y1)

            def candidates(self, p, radius=60.0):

                idx = self.tree.query(
                    Point(p).buffer(radius)
                )

                result = []

                for i in idx:

                    # Shapely 2 returns integer indexes
                    if isinstance(i, (int, np.integer)):
                        arc_id = int(i)

                    # Shapely 1 returns geometries
                    else:
                        arc_id = self.geoms.index(i)

                    distance = self.geoms[arc_id].distance(Point(p))

                    if distance <= radius:
                        result.append((arc_id, distance))

                return result

        # --------------------------------------------------
        # 3. Run HMM + Viterbi decoder
        # --------------------------------------------------
        trip = TripAdapter()

        decoded = decode2(
            trip,
            radius=60.0,
            sigma=15.0,
            hop_pen=1.5,
            jump_pen=12.0,
            head_w=1.0,
            beta=30.0,
            mode="hops"
        )

        # --------------------------------------------------
        # 4. Convert point-level predictions into route
        # --------------------------------------------------
        route = []

        for point_idx in sorted(decoded):

            arc_id = int(decoded[point_idx])

            if not route or route[-1] != arc_id:
                route.append(arc_id)

        logger.info(
            "HMM decoded trip %s: %d GPS points -> %d arcs",
            trip_id,
            len(gps_coords),
            len(route)
        )

        return route

    except Exception as e:

        logger.exception(
            "HMM decoding failed for trip %s",
            trip_id
        )

        raise RuntimeError(
            f"HMM decoding failed for trip {trip_id}: {e}"
        ) from e
