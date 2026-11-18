import numpy as np

def viterbi(cands, emis, trans):
    T = len(cands)
    score = [np.asarray(emis[0], float)]
    back = [None]

    for t in range(1, T):
        sc = np.full(len(cands[t]), -np.inf)
        bp = np.zeros(len(cands[t]), int)

        for j, sj in enumerate(cands[t]):
            tot = [
                score[-1][i] + trans(t, si, sj)
                for i, si in enumerate(cands[t - 1])
            ]
            k = int(np.argmax(tot))
            sc[j] = tot[k] + emis[t][j]
            bp[j] = k

        score.append(sc)
        back.append(bp)

    k = int(np.argmax(score[-1]))
    path = [k]

    for t in range(T - 1, 0, -1):
        k = int(back[t][k])
        path.append(k)

    path.reverse()
    return [cands[t][k] for t, k in enumerate(path)]
