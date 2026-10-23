"""원천 분해 국소투영 점검: (1) 통제변수 민감도, (2) 네 성분(이익·유보율·성향·소득) 모두의 지속성. 점추정.

주 사양과 같은 표본(모든 시점 고정)·2단계 가중최소제곱. 가중치는 사양마다 다시 계산.
→ output/ep_source_check.csv
"""
import numpy as np
import pandas as pd

import lp_tools as lt
import source_shocks as ss
from make_report import OUT

H = 7
SRC = ["x_P", "x_R", "x_H", "x_Y"]          # 이익, 유보율, 성향, 소득
OWN = {"x_P": "nl_corp", "x_R": "nl_corp", "x_H": "nl_hh", "x_Y": "nl_hh"}
LAG = lambda vs: [f"L{l}_{v}" for v in vs for l in (1, 2)]
LEV = [f"L1_{v}" for v in ["sv_corp", "sv_hh", "ca", "prof", "hh_rate"]]
SPECS = {
    "no controls": [],
    "lags of sources": LAG(SRC),
    "+ lags of CA change, growth": LAG(SRC + ["d_ca", "gdp_g"]),
    "+ lagged levels (main)": LAG(SRC + ["d_ca", "gdp_g"]) + LEV,
    "main, own S-I lags instead of CA": LAG(SRC + ["gdp_g"]) + LEV + ["dl1", "dl2"],
    "lags of sources, growth, own S-I (no levels)": LAG(SRC + ["gdp_g"]) + ["dl1", "dl2"],
}


def prepare():
    d = ss.prepare(pd.read_csv(OUT.parent / "data" / "panel.csv"))
    q = d.dropna(subset=[f"{y}_h{h}" for y in ("nl_corp", "nl_hh") for h in range(H)] + SRC + LAG(SRC + ["d_ca", "gdp_g"]) + LEV)
    return d, q[q.year <= 2019].copy()


def estimate(q, ctrl_fn):
    rows = []
    for y in ("nl_corp", "nl_hh"):
        ctrl = ctrl_fn(y)
        r = q.dropna(subset=ctrl).copy()
        M = np.hstack([r[SRC + ctrl].to_numpy(float), lt._fe_matrix(r, ("country", "year"))])
        e = r[f"{y}_h0"].to_numpy(float) - M @ np.linalg.lstsq(M, r[f"{y}_h0"].to_numpy(float), rcond=None)[0]
        r["w"] = 1 / r.country.map(pd.Series(e, index=r.index).groupby(r.country).var())
        B = np.array([lt.ols(r, f"{y}_h{h}", SRC, ctrl, se=False, w="w")[0] for h in range(H)])
        for k, x in enumerate(SRC):
            if OWN[x] == y:
                b = B[:, k]
                rows.append(dict(source=x, outcome=y, n=len(r), impact=b[0], **{f"h{h}": b[h] / b[0] for h in range(H)},
                                 short=b[1:3].mean() / b[0], medium=b[3:].mean() / b[0]))
    return rows


if __name__ == "__main__":
    d, q = prepare()
    g = q.groupby("country")
    for y, s in (("nl_corp", "c"), ("nl_hh", "h")):
        dy = d.groupby("country")[y].diff()
        d[f"dl1_{s}"], d[f"dl2_{s}"] = dy.groupby(d.country).shift(1), dy.groupby(d.country).shift(2)
    q = q.join(d[["dl1_c", "dl2_c", "dl1_h", "dl2_h"]])
    out = []
    for name, ctrl in SPECS.items():
        def fn(y, ctrl=ctrl):
            s = "c" if y == "nl_corp" else "h"
            return [c.replace("dl1", f"dl1_{s}").replace("dl2", f"dl2_{s}") if c in ("dl1", "dl2") else c for c in ctrl]
        out += [dict(spec=name, **r) for r in estimate(q, fn)]
    r = pd.DataFrame(out)
    r.to_csv(OUT / "ep_source_check.csv", index=False)
    lab = {"x_P": "corp: profit", "x_R": "corp: retention", "x_H": "hh: saving rate", "x_Y": "hh: income"}
    for name, g in r.groupby("spec", sort=False):
        print(f"== {name}  (n = {g.n.min()}–{g.n.max()})")
        for _, x in g.iterrows():
            print(f"   {lab[x.source]:16s} impact {x.impact:5.2f} |", " ".join(f"{x[f'h{h}']:5.2f}" for h in range(H)),
                  f"| 1-2y {x.short:5.2f}  3-6y {x.medium:5.2f}")
