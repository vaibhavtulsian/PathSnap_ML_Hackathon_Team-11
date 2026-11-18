import heapq, numpy as np, shapely
from collections import deque
from viterbi import viterbi
from hmm_decoder import FLOOR, hops
from roadnet import Trip

# ---- 1. point labels: monotone alignment of GPS points to the reference route ----
def aligned_labels(trip, max_skip=4):
    route = [int(a) for a in trip.route]; m = len(route); n = len(trip.pts)
    geoms = np.array([trip.geoms[a] for a in route], dtype=object)
    cost = shapely.distance(shapely.points(trip.pts)[:, None], geoms[None, :])
    dp = np.full((n, m), np.inf); bp = np.zeros((n, m), int); dp[0] = cost[0]
    for i in range(1, n):
        best = np.full(m, np.inf); arg = np.zeros(m, int)
        for k in range(max_skip + 1):
            prev = np.full(m, np.inf); prev[k:] = dp[i - 1][:m - k] if k else dp[i - 1]
            better = prev < best; best[better] = prev[better]; arg[better] = (np.arange(m) - k)[better]
        dp[i] = cost[i] + best; bp[i] = arg
    pos = np.zeros(n, int); pos[-1] = int(np.argmin(dp[-1]))
    for i in range(n - 1, 0, -1): pos[i - 1] = bp[i][pos[i]]
    return np.array([route[p] for p in pos]), pos

def get_labels(trip):
    if not hasattr(trip, "_labels"): trip._labels, trip._pos = aligned_labels(trip)
    return trip._labels

def evaluate2(trip, pred):
    lab = get_labels(trip); ref = set(int(a) for a in trip.route)
    acc = np.mean([pred[i] == lab[i] for i in pred])
    seq = [pred[i] for i in sorted(pred)]
    cont = np.mean([(a == b) or (b in trip.succ[a]) for a, b in zip(seq, seq[1:])]) if len(seq) > 1 else 1.0
    pr = set(seq)
    return dict(point_acc=float(acc), continuity=float(cont),
                arc_precision=len(pr & ref) / len(pr), arc_recall=len(pr & ref) / len(ref))

# ---- 2. decoder with two transition models: 'hops' (default) and 'route' (route vs straight distance) ----
def _arc_len(trip):
    if not hasattr(trip, "_len"): trip._len = np.array([g.length for g in trip.geoms])
    return trip._len

def node_dists(trip, src_node, cutoff):
    if not hasattr(trip, "_out"):
        out = {}
        for j, (a, b) in enumerate(trip.arcs): out.setdefault(int(a), []).append((int(b), j))
        trip._out = out; trip._nd = {}
    key = (src_node, cutoff)
    if key in trip._nd: return trip._nd[key]
    L = _arc_len(trip); dist = {src_node: 0.0}; pq = [(0.0, src_node)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18) or d > cutoff: continue
        for v, j in trip._out.get(u, []):
            nd = d + L[j]
            if nd < dist.get(v, 1e18) and nd <= cutoff: dist[v] = nd; heapq.heappush(pq, (nd, v))
    trip._nd[key] = dist; return dist

def decode2(trip, radius=60.0, sigma=15.0, hop_pen=1.5, jump_pen=12.0, head_w=1.0,
            beta=30.0, mode="hops", classifier_logp=None, gap_s=10.0, max_route=600.0):
    n = len(trip.pts); cands, emis, keep = [], [], []
    for i in range(n):
        c = trip.candidates(trip.pts[i], radius)
        if not c: continue
        arcs = [a for a, _ in c]
        if classifier_logp is not None:
            e = [max(classifier_logp(i, a), FLOOR) for a in arcs]
        else:
            e = [-(d ** 2) / (2 * sigma ** 2) for _, d in c]
            if head_w and i + 1 < n:
                v = trip.pts[min(i + 1, n - 1)] - trip.pts[max(i - 1, 0)]
                if np.hypot(*v) > 2.0:
                    th = np.arctan2(v[0], v[1])
                    e = [x + head_w * (np.cos(th - trip.arc_bearing(a)) - 1) for x, a in zip(e, arcs)]
        cands.append(arcs); emis.append(np.array(e)); keep.append(i)
    L = _arc_len(trip); cache = {}
    def trans(t, a, b):
        i0, i1 = keep[t - 1], keep[t]
        gap = trip.t[i1] - trip.t[i0] > gap_s
        if mode == "hops":
            k = (a, b)
            if k not in cache: cache[k] = hops(trip, a, b)
            h = cache[k]; lp = -jump_pen if h is None else -hop_pen * h
        else:
            if a == b: lp = 0.0
            else:
                nd = node_dists(trip, int(trip.arcs[a][1]), max_route).get(int(trip.arcs[b][0]))
                if nd is None: lp = -jump_pen
                else:
                    straight = float(np.hypot(*(trip.pts[i1] - trip.pts[i0])))
                    lp = -abs(nd + 0.5 * (L[a] + L[b]) - straight) / beta - 0.5
        return lp * (0.3 if gap else 1.0)
    return dict(zip(keep, viterbi(cands, emis, trans)))

# ---- 3. classifier hook (stand-in logistic regression until Member 2's model is ready) ----
def cand_features(trip, i, radius=60.0):
    c = trip.candidates(trip.pts[i], radius); n = len(trip.pts)
    v = trip.pts[min(i + 1, n - 1)] - trip.pts[max(i - 1, 0)]; sp = float(np.hypot(*v)); th = np.arctan2(v[0], v[1])
    rows = []
    for a, d in c:
        cd = np.cos(th - trip.arc_bearing(a)) if sp > 2.0 else 0.0
        rows.append([d, d * d, cd, sp, len(trip.succ[a])])
    return [a for a, _ in c], np.array(rows)

def build_table(trip, radius=60.0, with_labels=True):
    X, y, key = [], [], []
    lab = get_labels(trip) if with_labels else None
    for i in range(len(trip.pts)):
        arcs, F = cand_features(trip, i, radius)
        for a, f in zip(arcs, F):
            X.append(f); key.append((i, a)); y.append(int(lab[i] == a) if with_labels else 0)
    return np.array(X), np.array(y), key

def classifier_logp_for(trip, model, radius=60.0):
    X, _, key = build_table(trip, radius, with_labels=False)
    if len(X) == 0: return (lambda i, a: FLOOR), {}
    p = np.clip(model.predict_proba(X)[:, 1], 1e-9, 1.0)
    d = {k: float(np.log(v)) for k, v in zip(key, p)}
    return (lambda i, a: d.get((i, a), FLOOR)), dict(zip(key, p))

def classifier_only(trip, probs):
    best = {}
    for (i, a), p in probs.items():
        if i not in best or p > best[i][1]: best[i] = (a, p)
    return {i: a for i, (a, _) in best.items()}

# ---- 4. function Member 4 calls from the API ----
def predict_route(trip_dir, params=None, classifier_logp=None):
    tr = Trip(trip_dir); pred = decode2(tr, classifier_logp=classifier_logp, **(params or {}))
    seq = []
    for i in sorted(pred):
        if not seq or seq[-1] != pred[i]: seq.append(int(pred[i]))
    return seq

# ---- 5. folium map ----
def make_map(trip, pred, path="trip_map.html"):
    import folium
    from pyproj import Transformer
    inv = Transformer.from_crs("EPSG:32630", "EPSG:4326", always_xy=True)
    arc_ll = lambda a: [(lat, lon) for lon, lat in (inv.transform(*trip.nodes[n]) for n in trip.arcs[a])]
    m = folium.Map(location=[trip.track_ll[0][1], trip.track_ll[0][0]], zoom_start=15)
    g = folium.FeatureGroup("GPS points (blue)")
    for lon, lat in trip.track_ll[::3]: folium.CircleMarker((lat, lon), radius=2, color="blue", fill=True).add_to(g)
    g.add_to(m)
    r = folium.FeatureGroup("Reference route (green)")
    for a in trip.route: folium.PolyLine(arc_ll(int(a)), color="green", weight=6, opacity=.6).add_to(r)
    r.add_to(m)
    p = folium.FeatureGroup("Predicted route (red)")
    for a in sorted(set(pred.values())): folium.PolyLine(arc_ll(a), color="red", weight=3).add_to(p)
    p.add_to(m)
    folium.LayerControl().add_to(m); m.save(path); return m

def complete_route(trip, seq, maxh=8):
    L = _arc_len(trip); out = [seq[0]] if seq else []
    for a, b in zip(seq, seq[1:]):
        if b in trip.succ[a] or a == b: out.append(b); continue
        dist = {a: 0.0}; prev = {}; pq = [(0.0, 0, a)]; found = False
        while pq:
            d, h, u = heapq.heappop(pq)
            if u == b: found = True; break
            if d > dist.get(u, 1e18) or h >= maxh: continue
            for v in trip.succ[u]:
                nd = d + L[v]
                if nd < dist.get(v, 1e18): dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, h + 1, v))
        if found:
            path = [b]
            while path[-1] != a: path.append(prev[path[-1]])
            out.extend(path[-2::-1])
        else: out.append(b)
    return out

def route_metrics(trip, pred, complete=True):
    seq = []
    for i in sorted(pred):
        if not seq or seq[-1] != pred[i]: seq.append(int(pred[i]))
    if complete: seq = complete_route(trip, seq)
    ref = set(int(a) for a in trip.route); pr = set(seq)
    cont = np.mean([(a == b) or (b in trip.succ[a]) for a, b in zip(seq, seq[1:])]) if len(seq) > 1 else 1.0
    p = len(pr & ref) / len(pr); r = len(pr & ref) / len(ref)
    return dict(route_precision=p, route_recall=r, route_f1=2 * p * r / (p + r + 1e-12), route_continuity=float(cont))
