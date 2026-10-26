"""이익 횡재(기업) vs 저축 행태 변화(가계): 자기 부문 S−I 지속성, 패널 VAR, 구성 요소 분해.

충격
  이익 횡재  x_P = ρ_{t−1}Δπ_t   (유보율을 그대로 둔 영업잉여 변화가 만든 기업저축 변화)
  행태 변화  x_H = y_{t−1}Δσ_t/100 (소득을 그대로 둔 저축률 변화가 만든 가계저축 변화)
LP (부문별 식, 각 부문 블록만):
  n^j_{t+h} − n^j_{t−1} = β_h x^j_t + γ_h z^j_t + Σ_{l=1,2}[ n^j_{t−l}, x^j_{t−l}, z^j_{t−l}, g_{t−l} ] + 국가·연도 FE
  기업: x = x_P, z = x_R(유보율 성분)   가계: x = x_H, z = x_Y(소득 성분)
  - z_t: 같은 부문 저축 변화의 나머지 성분. 넣어야 '원천'이 분리됨.
  - 다른 부문 같은 해 성분은 넣지 않음(충격의 결과일 수 있는 나쁜 통제). 다른 부문 시차는 예측력 검정 후 제외.
  - n^j는 수준 시차(평균회귀, 시차 보강), 성장률 시차(경기), 연도 FE(공통 충격·코로나), 국가 FE.
  - 2단계 WLS(시작 연도 식 잔차의 국가별 분산 역수). 표본: 두 부문 모두 h = 0…6 관측, 시작 연도 ≤ 2019.
패널 VAR(부문별 블록): 기업 [x_R, x_P, g, n_C], 가계 [x_Y, x_H, g, n_H], 2시차, 국가·연도 FE, 촐레스키(순서 = LP 통제 구조).
구성 요소(누적 변화, 시작 연도 S−I 반응 = 1):
  기업 n_C = 영업잉여 − 순배당 − 법인세 − 이자 지급 − 총자본형성 + 기타
  가계 n_H = 가처분소득 − 소비 − 총자본형성 + 기타
구간: 국가 군집 부트스트랩 499회(LP·VAR·분해 같은 추출).
→ output/ep_wb_paths.csv, output/ep_wb_decomp.csv, output/ep_wb_avg.csv,
  output/paper/fig_wb_persistence.(png|pdf), fig_wb_decomp.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import lp_tools as lt
import source_shocks as ss
from make_report import OUT

H = 7
SEC = {"corp": dict(y="nl_corp", x="x_P", z="x_R", comps=["prof", "div_net", "tax_paid", "int_paid", "gcf_corp"]),
       "hh": dict(y="nl_hh", x="x_H", z="x_Y", comps=["yd_hh", "cons_hh", "gcf_hh"])}
SIGN = {"prof": 1, "div_net": -1, "tax_paid": -1, "int_paid": -1, "gcf_corp": -1, "yd_hh": 1, "cons_hh": -1, "gcf_hh": -1}
LAB = {"prof": "Operating surplus", "div_net": "Net dividends paid (−)", "tax_paid": "Income taxes (−)", "int_paid": "Interest paid (−)",
       "gcf_corp": "Investment (−)", "yd_hh": "Disposable income", "cons_hh": "Consumption (−)", "gcf_hh": "Investment (−)", "other": "Other"}


def data():
    p = pd.read_csv(OUT.parent / "data" / "panel.csv")
    d = ss.prepare(p)
    d["div_net"] = d.div_paid - d.div_recv_corp
    g = d.groupby("country")
    for v in ["div_net", "tax_paid", "int_paid", "gcf_corp", "yd_hh", "cons_hh", "gcf_hh", "prof"]:
        for h in range(H):
            d[f"{v}_h{h}"] = lt.add_lp_columns(d, v, h)
    for v in ["nl_corp", "nl_hh", "x_P", "x_R", "x_H", "x_Y", "gdp_g"]:
        for l in (1, 2):
            d[f"L{l}_{v}"] = g[v].shift(l)
    for s, m in SEC.items():
        oth = [f"{y}_h{h}" for h in range(H) for y in [m["y"]] + m["comps"]]
        d[f"other_{s}_ok"] = d[oth].notna().all(axis=1)
    need = [f"{m['y']}_h{h}" for m in SEC.values() for h in range(H)] + [c for s in SEC for c in [SEC[s]["x"], SEC[s]["z"]] + ctrl(s)]
    q = d.dropna(subset=need)
    return d, q[q.year <= 2019].copy()


def ctrl(s):
    m = SEC[s]
    return [f"L{l}_{v}" for v in (m["y"], m["x"], m["z"], "gdp_g") for l in (1, 2)]


def weights(q):
    for s, m in SEC.items():
        X, C = [m["x"], m["z"]], ctrl(s)
        M = np.hstack([q[X + C].to_numpy(float), lt._fe_matrix(q, ("country", "year"))])
        e = q[f"{m['y']}_h0"].to_numpy(float) - M @ np.linalg.lstsq(M, q[f"{m['y']}_h0"].to_numpy(float), rcond=None)[0]
        q[f"w_{s}"] = 1 / q.country.map(pd.Series(e, index=q.index).groupby(q.country).var())
    return q


def lp(q, s, var):
    m = SEC[s]
    return np.array([lt.ols(q, f"{var}_h{h}", [m["x"], m["z"]], ctrl(s), se=False, w=f"w_{s}")[0][0] for h in range(H)])


# ---- 패널 VAR (부문 블록) ----
VORD = {"corp": ["x_R", "x_P", "gdp_g", "nl_corp"], "hh": ["x_Y", "x_H", "gdp_g", "nl_hh"]}


def var_irf(est, s):
    Z = VORD[s]
    lags = [f"L{l}_{v}" for l in (1, 2) for v in Z]
    e = est.dropna(subset=Z + lags)
    X = np.hstack([e[lags].to_numpy(float), lt._fe_matrix(e, ("country", "year"))])
    Y = e[Z].to_numpy(float)
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    U = Y - X @ B
    S = U.T @ U / (len(U) - X.shape[1])
    k = len(Z)
    F = np.zeros((2 * k, 2 * k))
    F[:k] = np.hstack([B[:k].T, B[k:2 * k].T])
    F[k:, :k] = np.eye(k)
    b = np.linalg.cholesky(S)[:, 1]
    st, out = np.concatenate([b, np.zeros(k)]), []
    for _ in range(H):
        out.append(st[k - 1])
        st = F @ st
    out = np.array(out)
    return out / out[0]


def stats(q, est):
    q = weights(q.copy())
    res = {}
    for s, m in SEC.items():
        base = lp(q, s, m["y"])
        res[f"lp_{s}"] = base / base[0]
        res[f"var_{s}"] = var_irf(est, s)
        for v in m["comps"]:
            res[f"dc_{s}_{v}"] = SIGN[v] * lp(q, s, v) / base[0]
        res[f"dc_{s}_other"] = res[f"lp_{s}"] - sum(res[f"dc_{s}_{v}"] for v in m["comps"])
    res["lp_diff"] = res["lp_hh"] - res["lp_corp"]
    res["var_diff"] = res["var_hh"] - res["var_corp"]
    return res


def run(reps=499):
    d, q = data()
    est = d[d.country.isin(q.country.unique())]
    pt = stats(q, est)
    keys = list(pt)
    flat = lambda r: np.concatenate([r[k] for k in keys])
    est_by = est.copy()
    # 같은 국가 추출을 LP 표본과 VAR 표본에 함께 적용
    both = pd.concat([q.assign(_lp=True), est_by.assign(_lp=False)], ignore_index=True)
    bs = lt.cluster_bootstrap(both, lambda b: flat(stats(b[b._lp.eq(True)], b[b._lp.eq(False)])), reps=reps)
    rows = []
    for i, k in enumerate(keys):
        sl = bs[:, i * H:(i + 1) * H]
        for h in range(H):
            rows.append(dict(key=k, h=h, b=pt[k][h], lo68=np.nanpercentile(sl[:, h], 16), hi68=np.nanpercentile(sl[:, h], 84),
                             lo90=np.nanpercentile(sl[:, h], 5), hi90=np.nanpercentile(sl[:, h], 95)))
    r = pd.DataFrame(rows).assign(n=len(q), countries=q.country.nunique(), var_n=len(est.dropna(subset=VORD["corp"])))
    avg = []
    for k in ("lp_corp", "lp_hh", "lp_diff", "var_corp", "var_hh", "var_diff"):
        i = keys.index(k)
        sl = bs[:, i * H:(i + 1) * H]
        for lab, lo, hi in (("1-2", 1, 3), ("3-6", 3, 7)):
            a = sl[:, lo:hi].mean(axis=1)
            avg.append(dict(key=k, years=lab, b=pt[k][lo:hi].mean(), lo68=np.percentile(a, 16), hi68=np.percentile(a, 84),
                            lo90=np.percentile(a, 5), hi90=np.percentile(a, 95), share_le0=np.mean(a <= 0)))
    r.to_csv(OUT / "ep_wb_paths.csv", index=False)
    pd.DataFrame(avg).to_csv(OUT / "ep_wb_avg.csv", index=False)
    return r, pd.DataFrame(avg)


def checks():
    """다른 부문 시차의 예측력(시작 연도·h3 결과식, 충격식), 군집 강건 F."""
    import ep_iv_industry as e
    _, q = data()
    q = weights(q)
    out = []
    for s, m in SEC.items():
        o = "hh" if s == "corp" else "corp"
        Z = [f"L{l}_{v}" for v in (SEC[o]["y"], SEC[o]["x"], SEC[o]["z"]) for l in (1, 2)]
        for dep in (f"{m['y']}_h0", f"{m['y']}_h3", m["x"]):
            C = ctrl(s) + ([m["x"], m["z"]] if dep != m["x"] else [m["z"]])
            F, n, k = e.first_stage(q.assign(w_nl_corp=q[f"w_{s}"]), Z, C, dep)
            out.append(dict(sector=s, dependent=dep, F_other_sector_lags=F, n=n))
    return pd.DataFrame(out)


def fig_persist(r, a):
    import fig_paper as fp
    plt = fp.plt
    col = {"corp": fp.CO, "hh": fp.HH}
    lab = {"corp": "Profit windfall (corporate S–I)", "hh": "Shift in household saving rate (household S–I)"}
    g = lambda k: r[r.key == k].sort_values("h")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True)
    for ax, meth, title in ((axes[0], "lp", "(a) Local projections"), (axes[1], "var", "(b) Panel VAR")):
        for s in ("corp", "hh"):
            q = g(f"{meth}_{s}")
            ax.fill_between(q.h, q.lo68, q.hi68, color=col[s], alpha=0.16, lw=0)
            ax.plot(q.h, q.b, color=col[s], lw=1.9, marker="o", ms=3.3, label=lab[s])
        fp.finish(ax, "Own-sector S–I (impact = 1)" if meth == "lp" else "", h0=0)
        ax.set_xlabel("Years after the saving change ($h$)")
        ax.set_title(title, loc="left", fontsize=9.5)
    ax = axes[2]
    for meth, cl, ls in (("lp", "0.15", "-"), ("var", "0.45", "--")):
        q = g(f"{meth}_diff")
        ax.fill_between(q.h, q.lo90, q.hi90, color=cl, alpha=0.12 if meth == "lp" else 0.0, lw=0)
        if meth == "var":
            ax.plot(q.h, q.lo90, color=cl, lw=0.7, ls=":"); ax.plot(q.h, q.hi90, color=cl, lw=0.7, ls=":")
        ax.plot(q.h, q.b, color=cl, lw=1.9, ls=ls, marker="o" if meth == "lp" else "s", ms=3.3,
                label="Local projections (90% band shaded)" if meth == "lp" else "Panel VAR (90% band dotted)")
    fp.finish(ax, "", h0=0)
    ax.set_xlabel("Years after the saving change ($h$)")
    ax.set_title("(c) Household minus corporate", loc="left", fontsize=9.5)
    ax.set_ylim(-0.6, 1.1)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.8, frameon=False)
    fig.suptitle("A profit windfall fades faster than a shift in household saving behavior", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    n, k, vn = int(r.n.iloc[0]), int(r.countries.iloc[0]), int(r.var_n.iloc[0])
    fig.text(0.0, -0.13, f"Notes: Local projections (N = {n}, {k} economies): each sector's balance on the windfall or behavior component of its saving change and "
             "the remaining component, with two lags of the sector's balance (levels), both components and GDP growth, and country and year fixed effects; "
             f"two-step weighted least squares. Panel VAR ({vn} country-years, through 2025): the same four variables per sector, two lags, country and year "
             "fixed effects, recursive ordering [remaining component, shock, growth, balance]. Responses normalized by the impact response. Shaded areas in "
             "(a)–(b): 68 percent bands. Country-cluster bootstrap (499 replications).", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_wb_persistence", sfx="")


def fig_decomp(r):
    import fig_paper as fp
    plt = fp.plt
    g = lambda k: r[r.key == k].sort_values("h")
    pal = {"prof": "#c98a1b", "div_net": "#7a5195", "tax_paid": "#bc5090", "int_paid": "#ef8e9c", "gcf_corp": "#2b6c8f",
           "yd_hh": "#c98a1b", "cons_hh": "#7a5195", "gcf_hh": "#2b6c8f", "other": "0.75"}
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.0), gridspec_kw=dict(height_ratios=[1.15, 1]))
    for j, (s, title) in enumerate((("corp", "Corporations after a profit windfall"), ("hh", "Households after a shift in the saving rate"))):
        ax = axes[0, j]
        comps = SEC[s]["comps"] + ["other"]
        hs = np.arange(H)
        pos, neg = np.zeros(H), np.zeros(H)
        for v in comps:
            b = g(f"dc_{s}_{v}").b.to_numpy()
            base = np.where(b >= 0, pos, neg)
            ax.bar(hs, b, bottom=base, color=pal[v], width=0.7, label=LAB[v], edgecolor="white", lw=0.3)
            pos, neg = pos + np.clip(b, 0, None), neg + np.clip(b, None, 0)
        ax.plot(hs, g(f"lp_{s}").b, color="k", lw=1.8, marker="o", ms=3.5, label="Net lending (S–I)")
        fp.finish(ax, "Cumulative change\n(impact S–I = 1)" if j == 0 else "", h0=0)
        ax.set_title(("(a) " if j == 0 else "(b) ") + title, loc="left", fontsize=9.2)
        ax.set_xlabel("")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=6.8, frameon=False, ncol=3)
        ax = axes[1, j]
        keys = ["prof", "div_net", "gcf_corp"] if s == "corp" else ["yd_hh", "cons_hh", "gcf_hh"]
        for v in keys:
            q = g(f"dc_{s}_{v}")
            sgn = SIGN[v]
            ax.fill_between(q.h, sgn * q.lo68, sgn * q.hi68, color=pal[v], alpha=0.16, lw=0)
            ax.plot(q.h, sgn * q.b, color=pal[v], lw=1.8, marker="o", ms=3, label=LAB[v].replace(" (−)", ""))
        fp.finish(ax, "Cumulative change, level\n(impact S–I = 1)" if j == 0 else "", h0=0)
        ax.set_xlabel("Years after the saving change ($h$)")
        ax.set_title(("(c) " if j == 0 else "(d) ") + ("Profits, dividends and investment" if s == "corp" else "Income, consumption and investment"),
                     loc="left", fontsize=9.2)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle("Where the extra saving goes", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.text(0.0, -0.03, "Notes: Same local projections as the persistence figure, with each item as the outcome (cumulative change from year −1, percent "
             "of GDP), scaled by the impact response of the sector's balance. (a)–(b): contributions to net lending, with outlays entered with a negative sign; "
             "'Other' is the remainder (interest and rent received, capital transfers, statistical differences). (c)–(d): levels of selected items "
             "with 68 percent country-cluster bootstrap bands.", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_wb_decomp", sfx="")


if __name__ == "__main__":
    if "--render" in sys.argv:
        r, a = pd.read_csv(OUT / "ep_wb_paths.csv"), pd.read_csv(OUT / "ep_wb_avg.csv")
    else:
        print(checks().round(2).to_string(index=False))
        r, a = run()
    fig_persist(r, a)
    fig_decomp(r)
    pd.set_option("display.width", 200)
    print(r.pivot_table(index="key", columns="h", values="b", sort=False).round(2).to_string())
    print(a.round(2).to_string(index=False))
