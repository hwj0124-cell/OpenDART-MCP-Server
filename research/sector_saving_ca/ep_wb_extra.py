"""이익 횡재 vs 저축 행태 변화: 추가 점검 세 가지 (ep_windfall_behavior.py와 같은 표본·통제·가중치).

(1) 다른 부문 성분을 같은 해에 통제: 기업 식 + x_H, x_Y / 가계 식 + x_P, x_R.
(2) 이익 횡재 뒤 다른 부문과 경상수지: 기업 식 설계로 기업·가계·정부 S−I와 경상수지의 누적 변화(기업 S−I 시작 연도 반응 = 1).
    배당·세금이 가계·정부로 넘어가 거기서 일부 저축되면 기업 S−I가 0이 되어도 경상수지는 남을 수 있음 → 한국 BVAR(경상수지 기준)과 비교.
(3) 상태의존: 교역조건(상품) 개선 해 D_t = 1[ToT 증가율 > 0]. x_P·D, x_P·(1−D), D를 함께 넣음(x_R·통제는 그대로).
    결과: 기업 S−I, 투자, 영업잉여, 순배당. 교역조건 자료가 있는 국가·연도만.
구간: 국가 군집 부트스트랩 499회.
→ output/ep_wb_extra.csv, output/paper/fig_wb_cross.(png|pdf), fig_wb_state.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_windfall_behavior as wb
import lp_tools as lt
from make_report import OUT

H = wb.H
SEC = wb.SEC


def path(q, s, var, X, C, k=0, w=None):
    return np.array([lt.ols(q, f"{var}_h{h}", X, C, se=False, w=w or f"w_{s}")[0][k] for h in range(H)])


def cross(q):
    out = {}
    for s, o in (("corp", "hh"), ("hh", "corp")):
        m, mo = SEC[s], SEC[o]
        for lab, X in (("own", [m["x"], m["z"]]), ("cross", [m["x"], m["z"], mo["x"], mo["z"]])):
            b = path(q, s, m["y"], X, wb.ctrl(s))
            out[f"cross_{s}_{lab}"] = b / b[0]
    return out


def spill(q):
    X, C = [SEC["corp"]["x"], SEC["corp"]["z"]], wb.ctrl("corp")
    base = path(q, "corp", "nl_corp", X, C)
    out = {f"spill_{v}": path(q, "corp", v, X, C) / base[0] for v in ("nl_corp", "nl_hh", "nl_gov", "ca", "prof")}
    return out


def state(q):
    q = q.dropna(subset=["tot_goods_g"]).copy()
    q = wb.weights(q)
    q["D"] = (q.tot_goods_g > 0).astype(float)
    q["xP_up"], q["xP_dn"] = q.x_P * q.D, q.x_P * (1 - q.D)
    X, C = ["xP_up", "xP_dn", "x_R", "D"], wb.ctrl("corp")
    out = {}
    for k, st in ((0, "up"), (1, "dn")):
        base = path(q, "corp", "nl_corp", X, C, k)
        out[f"state_{st}_nl_corp"] = base / base[0]
        for v in ("gcf_corp", "prof", "div_net"):
            out[f"state_{st}_{v}"] = path(q, "corp", v, X, C, k) / base[0]
    return out


def stats(q):
    q = wb.weights(q.copy())
    return {**cross(q), **spill(q), **state(q)}


def run(reps=499):
    _, q = wb.data()
    pt = stats(q)
    keys = list(pt)
    bs = lt.cluster_bootstrap(q, lambda b: np.concatenate([stats(b)[k] for k in keys]), reps=reps)
    rows = []
    for i, k in enumerate(keys):
        sl = bs[:, i * H:(i + 1) * H]
        for h in range(H):
            rows.append(dict(key=k, h=h, b=pt[k][h], lo68=np.nanpercentile(sl[:, h], 16), hi68=np.nanpercentile(sl[:, h], 84),
                             lo90=np.nanpercentile(sl[:, h], 5), hi90=np.nanpercentile(sl[:, h], 95)))
    qs = q.dropna(subset=["tot_goods_g"])
    r = pd.DataFrame(rows).assign(n=len(q), n_state=len(qs), k_state=qs.country.nunique(), share_up=(qs.tot_goods_g > 0).mean())
    r.to_csv(OUT / "ep_wb_extra.csv", index=False)
    return r


def figs(r):
    import fig_paper as fp
    plt = fp.plt
    g = lambda k: r[r.key == k].sort_values("h")
    base = pd.read_csv(OUT / "ep_wb_paths.csv")
    gb = lambda k: base[base.key == k].sort_values("h")

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.3))
    for ax, s, title in ((axes[0], "corp", "(a) Profit windfall: corporate S–I"), (axes[1], "hh", "(b) Saving-rate shift: household S–I")):
        cl = fp.CO if s == "corp" else fp.HH
        b = gb(f"lp_{s}")
        ax.fill_between(b.h, b.lo68, b.hi68, color=cl, alpha=0.16, lw=0)
        ax.plot(b.h, b.b, color=cl, lw=1.9, marker="o", ms=3.3, label="Own sector's components at t (baseline)")
        c = g(f"cross_{s}_cross")
        ax.plot(c.h, c.b, color="0.15", lw=1.4, ls="--", marker="s", ms=3, label="+ other sector's components at t")
        fp.finish(ax, "Own-sector S–I (impact = 1)" if s == "corp" else "", h0=0)
        ax.set_xlabel("Years after the saving change ($h$)")
        ax.set_title(title, loc="left", fontsize=9.3)
        ax.set_ylim(-0.6, 1.1)
        ax.legend(loc="upper right", fontsize=6.9, frameon=False)
    ax = axes[2]
    for v, cl, lab in (("nl_corp", fp.CO, "Corporations"), ("nl_hh", fp.HH, "Households"), ("nl_gov", "#6a8f3c", "Government"),
                       ("ca", "0.1", "Current account")):
        q = g(f"spill_{v}")
        if v == "ca":
            ax.fill_between(q.h, q.lo68, q.hi68, color="0.4", alpha=0.18, lw=0)
        ax.plot(q.h, q.b, color=cl, lw=2.0 if v == "ca" else 1.4, marker="o" if v == "ca" else None, ms=3.3, label=lab)
    fp.finish(ax, "", h0=0)
    ax.set_xlabel("Years after the saving change ($h$)")
    ax.set_title("(c) Profit windfall: sector balances and CA", loc="left", fontsize=9.3)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_ylim(-0.6, 1.1)
    fig.suptitle("Other sectors' components at t, and where the windfall ends up", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.text(0.0, -0.06, "Notes: Same local projections as the main persistence figure. (a)–(b): the dashed line adds the other sector's two components at t. "
             "(c): cumulative change in each sector's balance and in the current account after a profit windfall, from the corporate equation, scaled by "
             "the impact response of the corporate balance; the shaded area is the 68 percent band for the current account. Balances need not add up to "
             "the current account (financial corporations, statistical discrepancies).", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_wb_cross", sfx="")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
    up = r.share_up.iloc[0]
    for ax, v, title in zip(axes.flat, ("nl_corp", "gcf_corp", "prof", "div_net"),
                            ("(a) Corporate S–I", "(b) Corporate investment", "(c) Operating surplus", "(d) Net dividends paid")):
        for st, cl, lab in (("up", "#c98a1b", "Terms of trade improving"), ("dn", "0.35", "Terms of trade not improving")):
            q = g(f"state_{st}_{v}")
            ax.fill_between(q.h, q.lo68, q.hi68, color=cl, alpha=0.16, lw=0)
            ax.plot(q.h, q.b, color=cl, lw=1.9, marker="o", ms=3.2, label=lab)
        fp.finish(ax, "Cumulative change\n(impact corporate S–I = 1)" if v in ("nl_corp", "prof") else "", h0=0)
        ax.set_xlabel("Years after the saving change ($h$)" if v in ("prof", "div_net") else "")
        ax.set_title(title, loc="left", fontsize=9.3)
    axes[0, 0].legend(loc="upper right", fontsize=7.4, frameon=False)
    fig.suptitle("Profit windfalls in years of improving and non-improving terms of trade", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.text(0.0, -0.04, f"Notes: State-dependent local projections on {int(r.n_state.iloc[0])} country-years ({int(r.k_state.iloc[0])} economies) with goods "
             f"terms-of-trade data; the terms of trade improve in {up:.0%} of them. The profit-windfall component is interacted with an indicator for "
             "a rise in the goods terms of trade at t and its complement; the indicator, the retention component and the baseline controls are included. "
             "Each response is scaled by the impact response of the corporate balance in that state. 68 percent country-cluster bootstrap bands.",
             fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_wb_state", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_wb_extra.csv") if "--render" in sys.argv else run()
    figs(r)
    pd.set_option("display.width", 200)
    print(r.pivot_table(index="key", columns="h", values="b", sort=False).round(2).to_string())
    a = r.assign(per=np.where(r.h.between(1, 2), "1-2", np.where(r.h >= 3, "3-6", "0")))
    print(a[a.per != "0"].pivot_table(index="key", columns="per", values="b", sort=False).round(2).to_string())
