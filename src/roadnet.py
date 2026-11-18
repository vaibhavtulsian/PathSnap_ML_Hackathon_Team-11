import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

_T = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)

class Trip:
    def __init__(self, trip_dir):
        d = Path(trip_dir)
        tid = d.name
        self.id = tid

        self.nodes_ll = np.loadtxt(d / f"{tid}.nodes")
        self.arcs = np.loadtxt(d / f"{tid}.arcs", dtype=int)

        rp = d / f"{tid}.route"
        self.route = np.loadtxt(rp, dtype=int) if rp.exists() else None

        tr = pd.read_csv(
            d / f"{tid}.track",
            sep=r"\s+",
            header=None,
            names=["lon", "lat", "t"],
            on_bad_lines="skip"
        )

        tr["t"] = tr["t"].interpolate().bfill().ffill()
        self.track_ll = tr[["lon", "lat"]].values
        self.t = tr["t"].values

        x, y = _T.transform(
            self.nodes_ll[:, 0],
            self.nodes_ll[:, 1]
        )
        self.nodes = np.c_[x, y]

        px, py = _T.transform(
            self.track_ll[:, 0],
            self.track_ll[:, 1]
        )
        self.pts = np.c_[px, py]

        self.geoms = [
            LineString([self.nodes[a], self.nodes[b]])
            for a, b in self.arcs
        ]

        self.tree = STRtree(self.geoms)

        by_from = {}
        for j, (a, _) in enumerate(self.arcs):
            by_from.setdefault(a, []).append(j)

        self.succ = {
            i: by_from.get(b, [])
            for i, (_, b) in enumerate(self.arcs)
        }

    def arc_bearing(self, i):
        (x1, y1), (x2, y2) = (
            self.nodes[self.arcs[i][0]],
            self.nodes[self.arcs[i][1]]
        )
        return np.arctan2(x2 - x1, y2 - y1)

    def candidates(self, p, radius=60.0):
        idx = self.tree.query(Point(p).buffer(radius))
        out = [
            (int(i), self.geoms[i].distance(Point(p)))
            for i in idx
        ]
        return [(i, d) for i, d in out if d <= radius]
