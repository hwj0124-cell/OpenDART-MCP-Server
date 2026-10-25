"""기준 사양 순대출 반응의 누적합(h = 0…H−1)과 반감기: 국가 군집 부트스트랩 999회(fastboot).

누적 순대출 Σ_{j≤h} IRF_j: 충격 한 단위가 h년까지 만들어 낸 순대출(= 경상수지·순대외자산 기여)의 합.
반감기: IRF가 처음 0.5 아래로 내려가는 시점(선형 보간).
→ output/ep_cum.csv
"""
import numpy as np
import pandas as pd

import ep_final as ef
import ep_specs as sp
import fastboot as fb
from make_report import OUT

H, SEC = ef.H, ef.SEC


def half_life(irf):
    for h in range(1, len(irf)):
        if irf[h] < 0.5:
            return h - 1 + (irf[h - 1] - 0.5) / (irf[h - 1] - irf[h])
    return np.nan


def run(reps=999):
    q = sp.data()["B"]
    D = {}
    for s, m in SEC.items():
        q[f"w_{s}"] = sp.weights(q, s, ef.ctrl(s))
        for h in range(H):
            D[f"{s}_{h}"] = fb.Design(q, [m["x"], m["z"]], ef.ctrl(s), f"w_{s}", [f"{m['y']}_h{h}"])

    def stat(b):
        out = []
        for s, m in SEC.items():
            y = m["y"]
            irf = np.array([b[f"{s}_{h}"][f"{y}_h{h}"] / b[f"{s}_0"][f"{y}_h0"] for h in range(H)])
            out += list(np.cumsum(irf)) + [half_life(irf)]
        return np.array(out)
    pt, bs = fb.bootstrap(D, stat, reps=reps)
    pc = lambda p: np.nanpercentile(bs, p, axis=0)
    keys = [(s, f"cum{h}") for s in SEC for h in range(H)] + [(s, "half_life") for s in SEC]
    keys = [(s, f"cum{h}") for s in SEC for h in list(range(H)) + ["hl"]]
    r = pd.DataFrame(keys, columns=["sector", "stat"]).assign(b=pt, lo68=pc(16), hi68=pc(84), lo90=pc(5), hi90=pc(95))
    r.loc[r.stat == f"cum{'hl'}", "stat"] = "half_life"
    # 차이(가계 − 기업)의 누적합 분포
    nH = H + 1
    diff = bs[:, nH:nH + H] - bs[:, :H] if list(SEC)[0] == "corp" else bs[:, :H] - bs[:, nH:nH + H]
    r.to_csv(OUT / "ep_cum.csv", index=False)
    return r, diff


if __name__ == "__main__":
    import time
    t = time.time()
    r, diff = run()
    print(f"{time.time() - t:.0f}s", list(SEC))
    print(r.round(2).to_string())
    print("share of draws with hh cum > corp cum by h:", (diff > 0).mean(axis=0).round(3))
