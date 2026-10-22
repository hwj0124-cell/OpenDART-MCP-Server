"""균형 표본 점검: t+6까지 관측되는 사례만 남긴 점추정치(주 사양: FGLS, S−I 정규화, 시차 통제).

h6에서 자기 부문 S−I가 관측되지 않는 사례는 사건 더미를 0으로 돌려(비사건 관측으로) 모든 시점에서 같은 사례 집합을 쓴다.
가중치는 줄어든 사례로 1단계 잔차를 다시 구해 산출한다.
→ output/ep_balanced_lags.csv
"""
import numpy as np
import pandas as pd

import ep_report as rp
import lp_tools as lt
from make_report import OUT

rp.FGLS = rp.WSV = rp.LAGS = True
rp.NORM = "nl"


def weights(d, s):
    y0 = f"nl_{s}_h0"
    q = d[[y0, f"z_{s}", *rp.ctrl(s), "country", "year"]].dropna()
    X = np.hstack([q[[f"z_{s}", *rp.ctrl(s)]].to_numpy(float), lt._fe_matrix(q, ("country", "year"))])
    b = np.linalg.lstsq(X, q[y0].to_numpy(float), rcond=None)[0]
    e = pd.Series(q[y0].to_numpy(float) - X @ b, index=q.index)
    return 1 / d.country.map(e.groupby(q.country).var())


def run():
    P, d = rp.prepare()
    base = rp.stats(d)
    dropped = {}
    for s in ("hh", "corp"):
        m = (d[f"z_{s}"] != 0) & d[f"nl_{s}_h6"].isna()
        dropped[s] = d.loc[m, ["country", "year", f"z_{s}"]]
        d.loc[m, f"z_{s}"] = 0
        d[f"w_{s}"] = weights(d, s)
    bal = rp.stats(d)
    lab = [dict(sector=s, var=v, h=h) for s, v, h in rp.KEYS] + \
          [dict(sector="diff", var=n, h=h) for n, _, _ in rp.DIFFS for h in range(rp.K + 1)]
    r = pd.DataFrame(lab).assign(base=base, balanced=bal)
    r.to_csv(OUT / "ep_balanced_lags.csv", index=False)
    for s in ("hh", "corp"):
        n = int((d[f"z_{s}"] != 0).sum())
        print(f"[{s}] 남은 사례 {n}, 제외 {len(dropped[s])}:",
              ", ".join(f"{c}{y}({'+' if z > 0 else '−'})" for c, y, z in dropped[s].itertuples(index=False)))
    return r


if __name__ == "__main__":
    r = run()
    for (s, v), g in r.groupby(["sector", "var"], sort=False):
        print(f"{s:5s}{v:12s} base " + " ".join(f"{x:5.2f}" for x in g.base) + " | bal " + " ".join(f"{x:5.2f}" for x in g.balanced))
