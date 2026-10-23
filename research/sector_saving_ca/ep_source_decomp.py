"""원천 분해 국소투영(제안 사양)의 순대출 구성 분해와 성향 지표.

네 성분(이익·유보율·저축률·소득)마다 자기 부문 순대출의 변화를 구성 요소로 나눔(점추정, 자기 부문 시작 연도 S−I 반응 = 1):
  기업: 영업잉여 − 순배당 − 총자본형성 + 기타      가계: 가처분소득 − 소비 − 총자본형성 + 기타
성향 지표(원단위, 원천 저축 1%p당, ±1 국가 군집 표준오차): 가계 저축률(%p), 기업 재투자율·순배당 성향(영업잉여 대비 %p)
→ output/ep_source_decomp.csv, output/paper/fig_source_decomp.(png|pdf), fig_source_propensity.(png|pdf)
"""
import numpy as np
import pandas as pd

import ep_source_check as c
import ep_source_fig as sf
import lp_tools as lt
from make_report import OUT

H = c.H
EXTRA = ["prof", "div_net", "gcf_corp", "sv_corp", "yd_hh", "cons_hh", "gcf_hh", "hh_rate", "reinv", "payout"]
GROUPS = {"x_P": "nl_corp", "x_R": "nl_corp", "x_Y": "nl_hh", "x_H": "nl_hh"}


def sample():
    q, ctrl = sf.sample()
    p = pd.read_csv(OUT.parent / "data" / "panel.csv")
    p["div_net"] = p.div_paid - p.div_recv_corp
    p["reinv"], p["payout"] = 100 * p.gcf_corp / p.prof, 100 * p.div_net / p.prof
    p = lt.balance(p)
    for v in EXTRA:
        for h in range(H):
            p[f"{v}_h{h}"] = lt.add_lp_columns(p, v, h)
    q = q.drop(columns=[x for x in q.columns if any(x.startswith(v + "_h") for v in EXTRA)], errors="ignore")
    return q.merge(p[["country", "year"] + [f"{v}_h{h}" for v in EXTRA for h in range(H)]], on=["country", "year"], how="left"), ctrl


def coef(q, ctrl, y, src, outcome_sector):
    k = c.SRC.index(src)
    out = []
    for h in range(H):
        b, V, n = lt.ols(q, f"{y}_h{h}", c.SRC, ctrl[outcome_sector], w=f"w_{outcome_sector}")
        out.append((b[k], np.sqrt(V[k, k])))
    return np.array(out)


def run():
    q, ctrl = sample()
    rows = []
    for src, sec in GROUPS.items():
        base = coef(q, ctrl, sec, src, sec)[0, 0]
        ys = ["prof", "div_net", "gcf_corp", "sv_corp", "reinv", "payout"] if sec == "nl_corp" else ["yd_hh", "cons_hh", "gcf_hh", "hh_rate"]
        for y in [sec] + ys:
            r = coef(q, ctrl, y, src, sec)
            for h in range(H):
                rows.append(dict(source=src, var=y, h=h, b=r[h, 0], se=r[h, 1], norm=r[h, 0] / base))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_source_decomp.csv", index=False)
    return r


TITLE = {"x_P": "(a) Corporate saving from higher profits", "x_Y": "(b) Household saving from higher income",
         "x_R": "(c) Corporate saving from higher retention", "x_H": "(d) Household saving from a higher saving rate"}


def figures(r):
    import fig_paper as fp
    plt = fp.plt
    g = lambda s, v: r[(r.source == s) & (r["var"] == v)].sort_values("h").norm.to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4), sharey="row")
    for ax, s in zip(axes.ravel(), ["x_P", "x_Y", "x_R", "x_H"]):
        if GROUPS[s] == "nl_corp":
            nl = g(s, "nl_corp")
            comps = [("Operating surplus", g(s, "prof"), "#1f3b73"), ("Net dividends paid (−)", -g(s, "div_net"), "#7a6aa8"),
                     ("Capital formation (−)", -g(s, "gcf_corp"), "#b5541c")]
        else:
            nl = g(s, "nl_hh")
            comps = [("Disposable income", g(s, "yd_hh"), "#1f3b73"), ("Consumption (−)", -g(s, "cons_hh"), "#7a6aa8"),
                     ("Capital formation (−)", -g(s, "gcf_hh"), "#b5541c")]
        comps.append(("Other", nl - sum(v for _, v, _ in comps), "#a6a6a6"))
        hs, up, dn = np.arange(H), np.zeros(H), np.zeros(H)
        for lab, v, col in comps:
            ax.bar(hs, v, bottom=np.where(v >= 0, up, dn), width=0.6, color=col, alpha=0.85, edgecolor="white", lw=0.4, label=lab)
            up, dn = up + np.clip(v, 0, None), dn + np.clip(v, None, 0)
        ax.plot(hs, nl, color="black", lw=1.4, marker="o", ms=3, mfc="white", label="Saving–investment balance")
        ax.axhline(0, color=".3", lw=.6); ax.set_xticks(hs)
        ax.set_title(TITLE[s], loc="left", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Years after the saving change ($h$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Contribution\n(impact S–I = 1)")
    fig.suptitle("Where the extra saving goes: income-driven (top) and behavior-driven (bottom)", x=0.0, ha="left", fontweight="bold", y=1.01)
    fig.tight_layout()
    for ax, x0 in ((axes[1, 0], 0.27), (axes[1, 1], 0.77)):
        h_, l_ = ax.get_legend_handles_labels()
        fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(x0, 0.0), ncol=2, fontsize=7)
    fig.text(0.0, -0.1, "Notes: Point estimates from the source local projections (proposed specification), normalized by the impact response of the own-sector "
             "saving–investment balance. Bars sum to that balance; 'Other' collects interest, taxes, other property income, transfers and non-produced assets.",
             fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_source_decomp", sfx="")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    for ax, s in zip(axes.ravel(), ["x_P", "x_Y", "x_R", "x_H"]):
        vs = [("reinv", "Reinvestment ratio", "#b5541c", "-"), ("payout", "Net payout ratio", "#7a6aa8", "--")] \
            if GROUPS[s] == "nl_corp" else [("hh_rate", "Household saving rate", "#1f3b73", "-")]
        for v, lab, col, ls in vs:
            q = r[(r.source == s) & (r["var"] == v)].sort_values("h")
            ax.fill_between(q.h, q.b - q.se, q.b + q.se, color=col, alpha=0.14, lw=0)
            ax.plot(q.h, q.b, color=col, ls=ls, lw=1.6, marker="o", ms=3, label=lab)
        ax.axhline(0, color=".3", lw=.6); ax.set_xticks(range(H))
        ax.set_title(TITLE[s], loc="left", fontsize=9)
        ax.set_ylabel("pp of operating surplus" if GROUPS[s] == "nl_corp" else "pp of disposable income")
        ax.legend(loc="best", fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("Years after the saving change ($h$)")
    fig.suptitle("Saving propensities after each type of saving change", x=0.0, ha="left", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.text(0.0, -0.02, "Notes: Responses per 1 pp of GDP of the saving change, not normalized; shaded areas are ±1 country-clustered standard error. "
             "Same specification as the decomposition.", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_source_propensity", sfx="")


if __name__ == "__main__":
    import sys
    r = pd.read_csv(OUT / "ep_source_decomp.csv") if "--render" in sys.argv else run()
    figures(r)
    t = r.pivot_table(index=["source", "var"], columns="h", values="norm", sort=False).round(2)
    print(t.to_string())
    print(r[r["var"].isin(["hh_rate", "reinv", "payout", "yd_hh"])].pivot_table(index=["source", "var"], columns="h", values="b", sort=False).round(2).to_string())
