import numpy as np
from collections import deque
from src.viterbi import viterbi

FLOOR = -30.0

def hops(trip, a, b, maxh=6):
    if a == b:
        return 0

    seen = {a}
    q = deque([(a, 0)])

    while q:
        u, h = q.popleft()
        if h >= maxh:
            continue

        for v in trip.succ[u]:
            if v == b:
                return h + 1
            if v not in seen:
                seen.add(v)
                q.append((v, h + 1))

    return None
