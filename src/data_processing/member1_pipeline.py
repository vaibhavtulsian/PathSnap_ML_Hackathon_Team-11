#!/usr/bin/env python
"""
PathSnap - Member 1 (Data and features)

Usage:
    python member1_pipeline.py --raw data/raw --out data/processed --radius 100

Expects, for every trip, files with the same stem: X.track X.nodes X.arcs [X.route]
(searched recursively under --raw). The partition (train / student_test /
evaluator_test) is inferred from the folder path: a folder name containing
"eval", "student" or "train". If your folders are named differently, edit
partition_of().

Rules from the brief that this script follows:
  * evaluator_test trips: candidates + features only, NEVER labels, never in folds.
  * folds are built over TRAIN trips only, grouped by trip.
  * no learned transforms are fitted here, so nothing leaks.

Outputs (in --out):
  candidates.parquet     trip_id, obs_idx, arc_id, distance          -> Member 3
  features.parquet       candidates + features + label               -> Member 2
  point_labels.parquet   per-point true arc (derived from .route)     -> Member 3
  folds.parquet          trip_id, partition, fold                     -> Members 2, 3
  trip_inventory.csv     per-trip EDA table
  miss_rate.csv          candidate miss rate vs radius
  figures/*.png, first_map.html
Requires: pandas numpy geopandas shapely>=2 scipy scikit-learn folium matplotlib pyarrow
"""
import argparse
import warnings
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import STRtree
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------- helpers
def read_ws(path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


def sibling(track, ext):
    p = track.with_suffix(ext)
    if p.exists():
        return p
    c = list(track.parent.glob(f"*{ext}"))
    return c[0] if len(c) == 1 else None


def partition_of(rel_path):
    s = "/".join(rel_path.parts).lower()
    if "eval" in s:
        return "evaluator_test"
    if "student" in s:
        return "student_test"
    if "train" in s:
        return "train"
    return "unknown"


def derive_point_labels(pts, lines, route):
    """Assign each GPS point to a route position, monotone non-decreasing
    (the vehicle moves forward along the route). Minimises total point-to-arc
    distance with a simple DP. Returns (route_pos, true_arc_id, true_dist)."""
    n, m = len(pts), len(route)
    cost = shapely.distance(pts[:, None], lines[route][None, :])  # n x m
    dp = cost[0].copy()
    back = np.zeros((n, m), dtype=np.int32)
    ar = np.arange(m)
    for i in range(1, n):
        pm = np.minimum.accumulate(dp)
        idx = np.maximum.accumulate(np.where(dp == pm, ar, 0))
        back[i] = idx
        dp = cost[i] + pm
    pos = np.zeros(n, dtype=np.int32)
    pos[-1] = int(np.argmin(dp))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return pos, route[pos], cost[np.arange(n), pos]


def angle_diff_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


# --------------------------------------------------------------- per trip
def process_trip(track, raw, radius, kmax, out, make_map):
    rel = track.relative_to(raw)
    tid = "__".join(rel.with_suffix("").parts)
    part = partition_of(rel)
    npath, apath, rpath = (sibling(track, e) for e in (".nodes", ".arcs", ".route"))
    info = dict(trip_id=tid, partition=part, has_nodes=npath is not None,
                has_arcs=apath is not None, has_route=rpath is not None,
                track_bytes=track.stat().st_size)

    trk = read_ws(track, ["lon", "lat", "t"])
    info["n_pts"] = len(trk)
    ok = (np.isfinite(trk[["lon", "lat", "t"]]).all(axis=1)
          & trk.lon.between(-180, 180) & trk.lat.between(-90, 90))
    info["n_invalid_pts"] = int((~ok).sum())
    dt_all = np.diff(trk.t.values)
    info.update(n_nonmonotonic=int((dt_all <= 0).sum()), n_gaps_gt1s=int((dt_all > 1).sum()),
                max_dt=float(dt_all.max()) if len(dt_all) else np.nan,
                duration_s=float(trk.t.iloc[-1] - trk.t.iloc[0]))
    if npath is None or apath is None:
        return None, None, info

    nodes = read_ws(npath, ["lon", "lat"])
    arcs = read_ws(apath, ["u", "v"]).astype(int)
    info.update(n_nodes=len(nodes), n_arcs=len(arcs),
                isolated_nodes=int(len(nodes) - len(np.unique(arcs.values))))
    g = coo_matrix((np.ones(len(arcs)), (arcs.u, arcs.v)), shape=(len(nodes),) * 2)
    info["n_components"] = int(connected_components(g, directed=False)[0])

    # ---- project to metres (UTM zone chosen from the data) ----
    crs = gpd.GeoSeries(gpd.points_from_xy(nodes.lon, nodes.lat), crs=4326).estimate_utm_crs()

    def proj(lon, lat):
        s = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(crs)
        return s.x.values, s.y.values

    nx_, ny_ = proj(nodes.lon, nodes.lat)
    lines = shapely.linestrings(np.stack([np.c_[nx_[arcs.u], ny_[arcs.u]],
                                          np.c_[nx_[arcs.v], ny_[arcs.v]]], axis=1))
    arc_len = shapely.length(lines)
    arc_brg = np.degrees(np.arctan2(nx_[arcs.v] - nx_[arcs.u], ny_[arcs.v] - ny_[arcs.u])) % 360

    v = trk[ok].copy()
    v["obs_idx"] = v.index
    v = v.reset_index(drop=True)
    px, py = proj(v.lon, v.lat)
    pts = shapely.points(px, py)
    n = len(v)
    # heading and speed from neighbouring points
    lo, hi = np.maximum(np.arange(n) - 1, 0), np.minimum(np.arange(n) + 1, n - 1)
    dx, dy = px[hi] - px[lo], py[hi] - py[lo]
    dts = np.maximum(v.t.values[hi] - v.t.values[lo], 1e-6)
    speed = np.hypot(dx, dy) / dts
    heading = np.degrees(np.arctan2(dx, dy)) % 360
    d_prev = np.r_[0, np.hypot(np.diff(px), np.diff(py))]
    d_next = np.r_[d_prev[1:], 0]
    ptab = pd.DataFrame(dict(trip_id=tid, obs_idx=v.obs_idx, lon=v.lon, lat=v.lat, t=v.t,
                             speed=speed, heading=heading))
    info["extent_m"] = float(np.hypot(np.ptp(px), np.ptp(py)))

    # ---- labels (never for evaluator trips) ----
    route = None
    if rpath is not None and part != "evaluator_test":
        route = pd.read_csv(rpath, header=None).iloc[:, 0].astype(int).values
        if route.max() >= len(arcs) or route.min() < 0:
            info["route_invalid"] = True
            route = None
    if route is not None:
        pos, true_arc, true_d = derive_point_labels(pts, lines, route)
        ptab["route_pos"], ptab["true_arc_id"], ptab["true_dist"] = pos, true_arc, true_d
        info["n_route_arcs"] = len(route)

    # ---- candidates ----
    qi, ai = STRtree(lines).query(pts, predicate="dwithin", distance=radius)
    c = pd.DataFrame(dict(qi=qi, arc_id=ai))
    c["distance"] = shapely.distance(pts[qi], lines[ai])
    c["rank"] = c.groupby("qi").distance.rank(method="first").astype(int)
    c = c[c["rank"] <= kmax].copy()
    qi, ai = c.qi.values, c.arc_id.values
    c["trip_id"], c["obs_idx"] = tid, v.obs_idx.values[qi]
    c["frac_along"] = shapely.line_locate_point(lines[ai], pts[qi], normalized=True)
    c["arc_length"] = arc_len[ai]
    c["dist_rel_min"] = c.distance - c.groupby("qi").distance.transform("min")
    c["n_cand"] = c.groupby("qi").distance.transform("size")
    hd = angle_diff_deg(heading[qi], arc_brg[ai])
    moving = speed[qi] > 0.5
    c["heading_diff"] = np.where(moving, hd, 90.0)            # 0 = same direction
    c["heading_diff_undirected"] = np.where(moving, np.minimum(hd, 180 - hd), 45.0)
    c["heading_valid"] = moving.astype(int)
    c["speed"] = speed[qi]
    c["d_prev"], c["d_next"] = d_prev[qi], d_next[qi]
    if route is not None:
        c["label"] = (c.arc_id.values == ptab.true_arc_id.values[qi]).astype(int)
    c = c.drop(columns="qi")

    if make_map and route is not None:
        m = folium.Map(location=[v.lat.mean(), v.lon.mean()], zoom_start=15)
        for a in range(0, len(arcs)):  # light background network
            folium.PolyLine([(nodes.lat[arcs.u[a]], nodes.lon[arcs.u[a]]),
                             (nodes.lat[arcs.v[a]], nodes.lon[arcs.v[a]])],
                            color="#bbbbbb", weight=1).add_to(m)
        for a in route:
            folium.PolyLine([(nodes.lat[arcs.u[a]], nodes.lon[arcs.u[a]]),
                             (nodes.lat[arcs.v[a]], nodes.lon[arcs.v[a]])],
                            color="green", weight=5, tooltip=f"true arc {a}").add_to(m)
        for i in range(0, n, max(n // 150, 1)):
            folium.CircleMarker((v.lat[i], v.lon[i]), radius=2, color="red").add_to(m)
        m.save(str(out / "first_map.html"))
    return c, ptab, info


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--radius", type=float, default=100.0, help="candidate radius, metres")
    ap.add_argument("--kmax", type=int, default=20, help="max candidates per point")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    tracks = sorted(raw.rglob("*.track"))
    print(f"found {len(tracks)} .track files")
    cands, ptabs, infos, mapped = [], [], [], False
    for k, t in enumerate(tracks):
        want_map = (not mapped) and partition_of(t.relative_to(raw)) == "train"
        c, p, info = process_trip(t, raw, a.radius, a.kmax, out, want_map)
        mapped = mapped or (want_map and c is not None and "label" in c)
        infos.append(info)
        if c is not None:
            cands.append(c)
            ptabs.append(p)
        print(f"[{k + 1}/{len(tracks)}] {info['trip_id']} ({info['partition']})")

    inv = pd.DataFrame(infos)
    inv.to_csv(out / "trip_inventory.csv", index=False)
    print(inv.groupby("partition").size().rename("trips"))

    cand = pd.concat(cands, ignore_index=True)
    pt = pd.concat(ptabs, ignore_index=True)
    key = ["trip_id", "obs_idx"]
    cand[key + ["arc_id", "distance"]].to_parquet(out / "candidates.parquet")
    cand.to_parquet(out / "features.parquet")
    pt.to_parquet(out / "point_labels.parquet")

    # ---- candidate miss rate (labelled points only: train + student_test) ----
    lab = pt.dropna(subset=["true_dist"]) if "true_dist" in pt else pt.iloc[0:0]
    rows = [dict(radius_m=r, miss_rate=float((lab.true_dist > r).mean()))
            for r in (25, 50, 75, 100, 150, 200)]
    if "label" in cand:
        hit = cand[cand.label.notna()].groupby(key).label.max()
        allp = lab.set_index(key).index
        rows.append(dict(radius_m=f"{a.radius:g} + top-{a.kmax}",
                         miss_rate=float(1 - hit.reindex(allp).fillna(0).mean())))
    miss = pd.DataFrame(rows)
    miss.to_csv(out / "miss_rate.csv", index=False)
    print("\ncandidate miss rate\n", miss)

    # ---- CV folds: TRAIN trips only, grouped by trip ----
    tr = inv.loc[inv.partition == "train", "trip_id"].values
    folds = inv[["trip_id", "partition"]].copy()
    folds["fold"] = -1
    if len(tr) >= a.folds:
        for f, (_, te) in enumerate(GroupKFold(a.folds).split(tr, groups=tr)):
            folds.loc[folds.trip_id.isin(tr[te]), "fold"] = f
    folds.to_parquet(out / "folds.parquet")

    # ---- EDA plots ----
    fig_specs = [
        ("points_per_trip", inv.n_pts, "GPS points per trip"),
        ("speed_hist", pt.speed.clip(upper=60), "speed (m/s, clipped at 60)"),
        ("max_time_gap", inv.max_dt.clip(upper=120), "max time gap per trip (s)"),
        ("trip_extent", inv.extent_m / 1000, "trip extent (km)"),
    ]
    for name, s, xl in fig_specs:
        plt.figure(figsize=(5, 3.5))
        plt.hist(s.dropna(), bins=40)
        plt.xlabel(xl)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(fig / f"{name}.png", dpi=150)
        plt.close()
    print(f"\ndone. outputs in {out}")


if __name__ == "__main__":
    main()