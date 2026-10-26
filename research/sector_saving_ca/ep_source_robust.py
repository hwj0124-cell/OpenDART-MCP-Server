"""원천 분해 LP 기준 사양(자기 S−I 수준 2시차, 같은 해 자기 부문 성분만)의 강건성: 시점별 전체 경로.

(1) 가중치·코로나: 가중(기준) / 무가중 / 코로나 이전 창(시작 연도 ≤ 2013) / 결과 연도 2020–21 제외(시점마다 해당 시작 연도 제외).
    연도 FE는 시작 연도 t에 붙지만 h가 고정이면 t+h 연도 FE와 같아, 모든 나라에 공통인 코로나 충격은 이미 흡수됨.
(2) 사양 전체 경로(스파게티): 시차 1–4, 통제 추가(ΔCA, 다른 부문 ΔS−I, Δ영업잉여·Δ저축률·Δ가처분소득, Δ노년부양비, 전부, 교역조건),
    같은 해 다른 부문 성분 포함, 무가중, 코로나 두 가지. 지그재그 정도 = 1–6년 경로의 방향 전환 횟수와 총변동.
표본: 4시차까지 관측되는 국가·연도(시작 연도 ≤ 2019, 795개, 38개국). 기준 사양 구간은 국가 군집 부트스트랩(499회).
→ output/ep_source_robust.csv, output/ep_source_robust_bands.csv,
  output/paper/fig_source_robust_covid.(png|pdf), fig_source_robust_paths.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_source_controls as ec
import ep_source_fig as sf
import lp_tools as lt
from make_report import OUT

SRC, OWN, H = ec.SRC, ec.OWN, ec.H
ORDER = ["x_P", "x_Y", "x_R", "x_H"]
COVID = (2020, 2021)


def setup():
    ur = pd.read_csv(OUT / "ep_source_controls_unitroot.csv")
    tr = dict(zip(ur["var"], ur["transform"]))
    return ec.data(tr), {name: (fn, sub) for name, fn, sub in ec.specs(tr)}


def path(q, y, X, C, weighted=True, drop_covid=False):
    q = q.dropna(subset=X + C)
    if weighted:
        M = np.hstack([q[X + C].to_numpy(float), lt._fe_matrix(q, ("country", "year"))])
        e = q[f"{y}_h0"].to_numpy(float) - M @ np.linalg.lstsq(M, q[f"{y}_h0"].to_numpy(float), rcond=None)[0]
        q = q.assign(w=1 / q.country.map(pd.Series(e, index=q.index).groupby(q.country).var()))
    B = []
    for h in range(H):
        qh = q[~(q.year + h).isin(COVID)] if drop_covid else q
        B.append(lt.ols(qh, f"{y}_h{h}", X, C, se=False, w="w" if weighted else None)[0])
    B = np.array(B)
    return {x: B[:, k] / B[0, k] for k, x in enumerate(X) if x in OWN[y]}, len(q)


def variants(S):
    base = S["Baseline (own S–I in levels, 2 lags)"][0]
    V = [("Baseline", base, {}, None),
         ("Unweighted", base, dict(weighted=False), None),
         ("Windows ending before 2020", base, {}, "precovid"),
         ("Outcome years 2020–21 dropped", base, dict(drop_covid=True), None),
         ("Pre-2020 windows, unweighted", base, dict(weighted=False), "precovid")]
    for name, (fn, sub) in S.items():
        if name.startswith("Baseline (") or name.startswith("Ref.: own S–I in changes") or "real rate" in name or name.startswith("Ref.: state"):
            continue
        V.append((name, fn, {}, sub))
    return V


def run():
    q, S = setup()
    rows = []
    for name, fn, kw, sub in variants(S):
        qq = q[q.year <= 2013] if sub == "precovid" else (q.dropna(subset=[f"L1c_{sub}", f"L2c_{sub}"]) if sub else q)
        for y in OWN:
            X, C = fn(y)
            res, n = path(qq, y, X, C, **kw)
            rows += [dict(spec=name, source=x, h=h, ratio=v[h], n=n) for x, v in res.items() for h in range(H)]
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_source_robust.csv", index=False)

    base = S["Baseline (own S–I in levels, 2 lags)"][0]
    stat = lambda d: np.concatenate([np.concatenate(list(path(d, y, *base(y))[0].values())) for y in OWN])
    bs = lt.cluster_bootstrap(q, stat, reps=499)
    keys = [(x, h) for y in OWN for x in OWN[y] for h in range(H)]
    b = pd.DataFrame(keys, columns=["source", "h"]).assign(lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0),
                                                           lo90=np.nanpercentile(bs, 5, axis=0), hi90=np.nanpercentile(bs, 95, axis=0))
    b.to_csv(OUT / "ep_source_robust_bands.csv", index=False)
    return r, b


def zigzag(r):
    out = []
    for (s, x), g in r.groupby(["spec", "source"], sort=False):
        v = g.sort_values("h").ratio.to_numpy()
        d = np.diff(v[1:])
        out.append(dict(spec=s, source=x, turns=int((np.sign(d[1:]) != np.sign(d[:-1])).sum()), tv=np.abs(d).sum(),
                        short=v[1:3].mean(), medium=v[3:].mean(), h6=v[6]))
    return pd.DataFrame(out)


def fig_covid(r, b):
    import fig_paper as fp
    plt = fp.plt
    col = {"x_P": "#c98a1b", "x_Y": "#c98a1b", "x_R": "#2b6c8f", "x_H": "#2b6c8f"}
    sty = {"Baseline": ("-", 1.9, "o"), "Unweighted": ("--", 1.3, None), "Windows ending before 2020": ("-.", 1.3, None),
           "Outcome years 2020–21 dropped": (":", 1.6, None), "Pre-2020 windows, unweighted": ((0, (1, 1, 4, 1)), 1.0, None)}
    gray = {"Baseline": None, "Unweighted": "0.15", "Windows ending before 2020": "0.35", "Outcome years 2020–21 dropped": "0.1",
            "Pre-2020 windows, unweighted": "0.55"}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
    for ax, x in zip(axes.flat, ORDER):
        bb = b[b.source == x].sort_values("h")
        ax.fill_between(bb.h, bb.lo90, bb.hi90, color=col[x], alpha=0.10, lw=0)
        ax.fill_between(bb.h, bb.lo68, bb.hi68, color=col[x], alpha=0.18, lw=0)
        for s, (ls, lw, mk) in sty.items():
            g = r[(r.spec == s) & (r.source == x)].sort_values("h")
            n = int(g.n.iloc[0])
            ax.plot(g.h, g.ratio, ls=ls, lw=lw, marker=mk, ms=3, color=gray[s] or col[x], label=f"{s} (N = {n})" if x == "x_P" else s)
        fp.finish(ax, "Own-sector S–I\n(impact = 1)" if x in ("x_P", "x_R") else "", h0=0)
        ax.set_xlabel("Years after the saving change ($h$)" if x in ("x_R", "x_H") else "")
        ax.set_title(sf.LAB[x], loc="left", fontsize=9.5)
        ax.set_ylim(-1.0, 1.8)
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=7.6, frameon=False)
    fig.suptitle("Weighting and the COVID-19 years", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.0, -0.08, "Notes: Baseline: own-sector balance in levels, the four components and GDP growth (two lags each), the own sector's two components at t, "
             "country and year fixed effects, two-step weighted least squares; shaded areas are its 68 and 90 percent country-cluster bootstrap bands (499 "
             "replications). Year fixed effects at t equal year effects at t+h for a given horizon, so the common COVID-19 shock is absorbed in every "
             "specification. 'Outcome years 2020–21 dropped' removes, at each horizon, start years whose outcome falls in 2020 or 2021.",
             fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_source_robust_covid", sfx="")


def fig_paths(r, b):
    import fig_paper as fp
    plt = fp.plt
    col = {"x_P": "#c98a1b", "x_Y": "#c98a1b", "x_R": "#2b6c8f", "x_H": "#2b6c8f"}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
    specs = [s for s in dict.fromkeys(r.spec) if s != "Baseline"]
    for ax, x in zip(axes.flat, ORDER):
        bb = b[b.source == x].sort_values("h")
        ax.fill_between(bb.h, bb.lo68, bb.hi68, color=col[x], alpha=0.18, lw=0, label="Baseline, 68% band")
        for i, s in enumerate(specs):
            g = r[(r.spec == s) & (r.source == x)].sort_values("h")
            ax.plot(g.h, g.ratio, color="0.45", lw=0.8, alpha=0.7, label="Other specifications" if i == 0 else None)
        g = r[(r.spec == "Baseline") & (r.source == x)].sort_values("h")
        ax.plot(g.h, g.ratio, color=col[x], lw=2.2, marker="o", ms=3.5, label="Baseline")
        fp.finish(ax, "Own-sector S–I\n(impact = 1)" if x in ("x_P", "x_R") else "", h0=0)
        ax.set_xlabel("Years after the saving change ($h$)" if x in ("x_R", "x_H") else "")
        ax.set_title(sf.LAB[x], loc="left", fontsize=9.5)
        ax.set_ylim(-1.0, 1.8)
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3, fontsize=7.8, frameon=False)
    fig.suptitle(f"Full response paths across {len(specs) + 1} specifications", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.0, -0.05, "Notes: Gray lines: lag length 1, 3 and 4; added controls (ΔCA, change in the other sector's balance, Δoperating surplus, Δsaving rate, "
             "Δdisposable income, Δold-age dependency, all of these, ToT growth); the other sector's components at t; unweighted; windows ending before 2020; "
             "outcome years 2020–21 dropped. All keep the own-sector balance in levels.", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_source_robust_paths", sfx="")


if __name__ == "__main__":
    if "--render" in sys.argv:
        r, b = pd.read_csv(OUT / "ep_source_robust.csv"), pd.read_csv(OUT / "ep_source_robust_bands.csv")
    else:
        r, b = run()
    fig_covid(r, b)
    fig_paths(r, b)
    pd.set_option("display.width", 220)
    print(r.pivot_table(index=["source", "spec"], columns="h", values="ratio", sort=False).loc[ORDER].round(2).to_string())
    z = zigzag(r)
    z.to_csv(OUT / "ep_source_robust_zigzag.csv", index=False)
    print(z.pivot_table(index="spec", columns="source", values="turns", sort=False)[ORDER].to_string())
