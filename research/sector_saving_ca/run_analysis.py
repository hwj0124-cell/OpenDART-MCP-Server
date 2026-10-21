"""기업저축 vs 가계저축 충격에 대한 순자금운용·경상수지 반응 비교.

1) 패널 LP: 두 충격을 동시에 넣은 국소투영 (국가·연도 FE, 국가 군집 SE)
2) 시점별 전가율 LP-IV: Δ_h CA 를 Δ_h(해당 부문 저축)에 회귀, 충격을 도구로
3) 에피소드 분석: 부문저축 급증 에피소드 전후 평균 경로 + 에피소드 LP
4) 단기(h=0~1) 대비 중기(h=4~6) 전가율 하락폭의 부문 간 차이 — 국가 블록 부트스트랩

    python run_analysis.py [--panel data/panel.csv] [--reps 499]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lp_tools import add_lp_columns, balance, cluster_bootstrap, iv, lagged, ols

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
H = 6
LAGS = 2
SHORT, MEDIUM = (0, 1), (4, 5, 6)
Z90 = 1.645

# 충격 정의: 부문 → (충격변수, 정규화 기준이 되는 부문저축)
SHOCKS = {
    "baseline": {"corp": "d_sv_corp", "hh": "d_sv_hh"},           # 부문 총저축 변화 (GDP 대비 %p)
    "profit_propensity": {"corp": "d_prof", "hh": "d_hh_rate"},  # 기업이익 vs 가계 저축률
}
OWN = {"corp": "sv_corp", "hh": "sv_hh"}
OWN_NL = {"corp": "nl_corp", "hh": "nl_hh"}
OUTCOMES = ["ca", "sv_corp", "sv_hh", "nl_corp", "nl_hh", "nl_gov", "nl_fin",
            "inv_corp", "inv_hh", "cons_hh", "div_paid", "tax_paid", "prof", "sv_gov"]
LABEL = {"corp": "Corporate (NFC) saving shock", "hh": "Household saving shock"}


def prepare(panel):
    d = balance(panel)
    g = d.groupby("country")
    for v in ["sv_corp", "sv_hh", "prof", "hh_rate", "ca", "nl_corp", "nl_hh"]:
        d[f"d_{v}"] = g[v].diff()
    base_ctrl = ["d_sv_corp", "d_sv_hh", "d_ca", "gdp_g"]
    levels = ["sv_corp", "sv_hh", "ca", "prof", "hh_rate"]
    d = pd.concat([d, lagged(d, base_ctrl + ["d_prof", "d_hh_rate"] + levels, LAGS)], axis=1)
    lp = {f"{y}_h{h}": add_lp_columns(d, y, h)
          for y in sorted(set(OUTCOMES) | set(OWN.values()) | set(OWN_NL.values())) if y in d
          for h in range(H + 1)}
    # 누적 반응 Σ_{k≤h}(y_{t+k} − y_{t−1}): 누적 전가율(Ramey–Zubairy 방식) 계산용
    for y in {k.rsplit("_h", 1)[0] for k in lp}:
        lp.update({f"{y}_c{h}": sum(lp[f"{y}_h{k}"] for k in range(h + 1)) for h in range(H + 1)})
    return pd.concat([d, pd.DataFrame(lp, index=d.index)], axis=1)


def controls_for(shock_set, extra=()):
    lagvars = ["d_sv_corp", "d_sv_hh", "d_ca", "gdp_g"]
    levels = ["sv_corp", "sv_hh", "ca"]  # 충격 직전 수준: 부문저축·CA의 평균회귀 통제
    if shock_set == "profit_propensity":
        lagvars += ["d_prof", "d_hh_rate"]
        levels += ["prof", "hh_rate"]
    return ([f"L{l}_{v}" for v in lagvars for l in range(1, LAGS + 1)]
            + [f"L1_{v}" for v in levels] + list(extra))


def sample(d, drop_years=()):
    return d[~d["year"].isin(drop_years)] if drop_years else d


# ---------------------------------------------------------------- 1) 결합 LP
def joint_lp(d, shock_set, extra=()):
    sx = SHOCKS[shock_set]
    ctrl = controls_for(shock_set, extra)
    rows = []
    for y in OUTCOMES:
        if y not in d or d[y].notna().sum() < 100:
            continue
        for h in range(H + 1):
            b, V, n = ols(d, f"{y}_h{h}", [sx["corp"], sx["hh"]], ctrl)
            diff = b[0] - b[1]
            se_d = np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1])
            rows.append(dict(set=shock_set, outcome=y, h=h, n=n,
                             b_corp=b[0], se_corp=np.sqrt(V[0, 0]),
                             b_hh=b[1], se_hh=np.sqrt(V[1, 1]),
                             diff=diff, se_diff=se_d))
    return pd.DataFrame(rows)


# ------------------------------------------------- 2) 전가율 LP-IV (정규화 반응)
def passthrough(d, shock_set, extra=(), se=True, outcome="ca"):
    """impact: 충격 시점 부문저축 1%p 증가당 h기 CA 반응 (충격 지속성+전가 포함)
       horizon: 0~h기 누적 부문저축 증가 1%p당 0~h기 누적 CA 증가 (누적 전가율)"""
    sx = SHOCKS[shock_set]
    ctrl = controls_for(shock_set, extra)
    rows = []
    for s in ("corp", "hh"):
        other = sx["hh" if s == "corp" else "corp"]
        c = ctrl + [other]
        for h in range(H + 1):
            bi, sei, n, Fi = iv(d, f"{outcome}_h{h}", f"{OWN[s]}_h0", sx[s], c, se=se)
            bh, seh, _, Fh = iv(d, f"{outcome}_c{h}", f"{OWN[s]}_c{h}", sx[s], c, se=se)
            rows.append(dict(set=shock_set, sector=s, h=h, n=n,
                             impact=bi, se_impact=sei, F_impact=Fi,
                             horizon=bh, se_horizon=seh, F_horizon=Fh))
    return pd.DataFrame(rows)


def pt_contrasts(pt):
    """단기·중기 평균 전가율과 하락폭, 부문 간 차이(이중차분)."""
    out = {}
    for col in ("impact", "horizon"):
        for s in ("corp", "hh"):
            p = pt[pt.sector == s].set_index("h")[col]
            out[f"{col}_{s}_short"] = p.loc[list(SHORT)].mean()
            out[f"{col}_{s}_medium"] = p.loc[list(MEDIUM)].mean()
            out[f"{col}_{s}_decline"] = out[f"{col}_{s}_short"] - out[f"{col}_{s}_medium"]
        out[f"{col}_dd"] = out[f"{col}_corp_decline"] - out[f"{col}_hh_decline"]
        for h in range(H + 1):
            out[f"{col}_diff_h{h}"] = (pt[(pt.sector == "corp") & (pt.h == h)][col].iloc[0]
                                       - pt[(pt.sector == "hh") & (pt.h == h)][col].iloc[0])
    return out


def bootstrap_contrasts(d, shock_set, reps, extra=()):
    point = pt_contrasts(passthrough(d, shock_set, extra, se=False))
    keys = list(point)
    def stat(bd):
        c = pt_contrasts(passthrough(bd, shock_set, extra, se=False))
        return [c[k] for k in keys]

    draws = cluster_bootstrap(d, stat, reps=reps)
    res = {}
    for i, k in enumerate(keys):
        x = draws[:, i]
        res[k] = dict(point=point[k], se=np.nanstd(x), lo90=np.nanpercentile(x, 5),
                      hi90=np.nanpercentile(x, 95),
                      p_le0=float(np.mean(x <= 0)))  # 단측: H0 차이 ≤ 0
    return res


# ------------------------------------------------------------- 3) 에피소드
def define_episodes(d, shock_set="baseline", q=0.9, gap=3):
    """부문저축이 상위 (1−q) 분위로 급증하고 다른 부문은 중위수 이하인 해를 에피소드 시작으로."""
    sx = SHOCKS[shock_set]
    c, h = d[sx["corp"]], d[sx["hh"]]
    thr_c, thr_h = c.quantile(q), h.quantile(q)
    med_c, med_h = c.median(), h.median()
    flags = {"corp": (c >= thr_c) & (h <= med_h), "hh": (h >= thr_h) & (c <= med_c)}
    ep = {}
    for s, f in flags.items():
        f = f.fillna(False).to_numpy()
        e = np.zeros(len(d), bool)
        last = {}
        for i, (ctry, yr) in enumerate(zip(d["country"], d["year"])):
            if f[i] and yr - last.get(ctry, -99) > gap:
                e[i] = True
                last[ctry] = yr
        ep[s] = pd.Series(e.astype(float), index=d.index)
    return ep, dict(thr_corp=thr_c, thr_hh=thr_h, med_corp=med_c, med_hh=med_h)


def event_study(d, ep, vars_, pre=3):
    """연도별 국가평균을 뺀 값으로 y_{t+k}−y_{t−1} 평균 경로 (k=−pre..H)."""
    rows = []
    dm = d.copy()
    for v in vars_:
        dm[v] = d[v] - d.groupby("year")[v].transform("mean")
    g = dm.groupby("country")
    for s, e in ep.items():
        idx = np.flatnonzero(e.to_numpy() == 1)
        for v in vars_:
            base = g[v].shift(1).to_numpy()
            for k in range(-pre, H + 1):
                yk = g[v].shift(-k).to_numpy()
                x = (yk - base)[idx]
                x = x[~np.isnan(x)]
                if len(x) > 2:
                    rows.append(dict(sector=s, var=v, k=k, mean=x.mean(),
                                     se=x.std(ddof=1) / np.sqrt(len(x)), n=len(x)))
    return pd.DataFrame(rows)


def episode_lp(d, ep, shock_set="baseline"):
    """에피소드 더미 LP (두 더미 동시 투입) + 충격시점 부문저축 1%p로 정규화한 CA 전가율."""
    dd = d.assign(ep_corp=ep["corp"], ep_hh=ep["hh"])
    ctrl = controls_for(shock_set)
    rows = []
    for y in ["ca", "sv_corp", "sv_hh", "nl_corp", "nl_hh", "nl_gov", "inv_corp", "div_paid"]:
        if y not in dd:
            continue
        for h in range(H + 1):
            b, V, n = ols(dd, f"{y}_h{h}", ["ep_corp", "ep_hh"], ctrl)
            rows.append(dict(outcome=y, h=h, n=n, b_corp=b[0], se_corp=np.sqrt(V[0, 0]),
                             b_hh=b[1], se_hh=np.sqrt(V[1, 1]),
                             se_diff=np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1])))
    norm = []
    for s in ("corp", "hh"):
        other = "ep_hh" if s == "corp" else "ep_corp"
        for h in range(H + 1):
            b, se, n, F = iv(dd, f"ca_h{h}", f"{OWN[s]}_h0", f"ep_{s}", ctrl + [other])
            norm.append(dict(sector=s, h=h, impact=b, se_impact=se, F=F, n=n))
    return pd.DataFrame(rows), pd.DataFrame(norm)


# ---------------------------------------------------------------- 그림
def plot_band(ax, x, b, se, color, label):
    ax.plot(x, b, color=color, lw=2, label=label)
    ax.fill_between(x, b - Z90 * se, b + Z90 * se, color=color, alpha=0.15, lw=0)


C = {"corp": "#c0392b", "hh": "#2c6fbb"}


def fig_passthrough(pt, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for s in ("corp", "hh"):
        p = pt[pt.sector == s]
        plot_band(axes[0], p.h, p.impact, p.se_impact, C[s], LABEL[s])
        plot_band(axes[1], p.h, p.horizon, p.se_horizon, C[s], LABEL[s])
    axes[0].set_title("CA response to 1%p sector-saving rise on impact")
    axes[1].set_title("Cumulative pass-through: ΣΔCA / ΣΔ(sector saving)")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("years after shock")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_joint(jl, outcomes, path, title):
    n = len(outcomes)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for ax, y in zip(axes.flat, outcomes):
        j = jl[jl.outcome == y]
        plot_band(ax, j.h, j.b_corp, j.se_corp, C["corp"], "corp")
        plot_band(ax, j.h, j.b_hh, j.se_hh, C["hh"], "hh")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(y, fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_events(es, vars_, path, title):
    fig, axes = plt.subplots(1, len(vars_), figsize=(3.2 * len(vars_), 3), squeeze=False)
    for ax, v in zip(axes[0], vars_):
        for s in ("corp", "hh"):
            e = es[(es.sector == s) & (es["var"] == v)]
            plot_band(ax, e.k, e["mean"], e.se, C[s], f"{s} (n={e.n.max() if len(e) else 0})")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(-0.5, color="grey", lw=0.6, ls=":")
        ax.set_title(v, fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main(panel_path, reps):
    OUT.mkdir(exist_ok=True)
    d = prepare(pd.read_csv(panel_path))
    summary = {"countries": int(d.dropna(subset=["ca_h0", "d_sv_corp", "d_sv_hh"])["country"].nunique()),
               "years": [int(d.year.min()), int(d.year.max())]}

    variants = {
        "baseline": dict(shock_set="baseline", drop=()),
        "baseline_exCOVID": dict(shock_set="baseline", drop=(2020, 2021)),
        "baseline_growthctrl": dict(shock_set="baseline", drop=(), extra=("gdp_g",)),
        "profit_propensity": dict(shock_set="profit_propensity", drop=()),
        "profit_propensity_exCOVID": dict(shock_set="profit_propensity", drop=(2020, 2021)),
    }
    all_jl, all_pt = [], []
    for name, v in variants.items():
        ds = sample(d, v["drop"])
        extra = v.get("extra", ())
        jl = joint_lp(ds, v["shock_set"], extra).assign(variant=name)
        pt = passthrough(ds, v["shock_set"], extra).assign(variant=name)
        all_jl.append(jl)
        all_pt.append(pt)
        fig_passthrough(pt, OUT / f"passthrough_{name}.png", f"Pass-through — {name}")
        fig_joint(jl, ["ca", "sv_corp", "sv_hh", "nl_corp", "nl_hh", "nl_gov",
                       "inv_corp", "inv_hh", "div_paid", "tax_paid", "cons_hh", "prof"],
                  OUT / f"joint_lp_{name}.png", f"Joint LP responses (pp of GDP) — {name}")
        if reps:
            summary[f"contrasts_{name}"] = bootstrap_contrasts(ds, v["shock_set"], reps, extra)
        summary[f"pt_{name}"] = pt.to_dict("records")
    pd.concat(all_jl).to_csv(OUT / "joint_lp.csv", index=False)
    pd.concat(all_pt).to_csv(OUT / "passthrough.csv", index=False)

    for name, ss, drop in [("episodes_baseline", "baseline", ()),
                           ("episodes_baseline_exCOVID", "baseline", (2020, 2021)),
                           ("episodes_propensity", "profit_propensity", ())]:
        ds = sample(d, drop)
        ep, thr = define_episodes(ds, ss)
        lst = {s: ds.loc[e == 1, ["country", "year"]].astype({"year": int}).values.tolist()
               for s, e in ep.items()}
        vars_ = ["ca", "sv_corp", "sv_hh", "nl_corp", "nl_hh", "nl_gov"]
        es = event_study(ds, ep, [v for v in vars_ if v in ds])
        es.to_csv(OUT / f"{name}_event.csv", index=False)
        fig_events(es, ["ca", "sv_corp", "sv_hh", "nl_corp", "nl_hh"], OUT / f"{name}.png",
                   f"Event study (year-demeaned, change vs t−1) — {name}")
        elp, enorm = episode_lp(ds, ep, ss)
        elp.to_csv(OUT / f"{name}_lp.csv", index=False)
        enorm.to_csv(OUT / f"{name}_lp_norm.csv", index=False)
        summary[name] = dict(thresholds=thr, episodes=lst, norm=enorm.to_dict("records"))

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("contrasts")},
                     indent=1, default=float)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(HERE / "data" / "panel.csv"))
    ap.add_argument("--reps", type=int, default=499)
    a = ap.parse_args()
    main(a.panel, a.reps)
