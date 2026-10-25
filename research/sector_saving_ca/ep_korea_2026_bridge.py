"""동료 방식(1023 드래프트 각주 7·Figure 5(b))의 재현과 세 방법의 표본 외 검증, 이익/유보 분해까지의 다리.

방법 C (동료, 분기·GDP 대비): 기준연도 B의 연간값으로
  I_C(t) = [GCF(t) − 정부 GFCF(t)] × I_NFC(B) / [GCF(B) − 정부 GFCF(B)]
  S_C(t) = I_C(t) + NL_NFC^FoF(t) + GDP(t) × D_NFC(B)/GDP(B),   D_NFC(B) = S_NFC(B) − I_NFC(B) − NL_NFC^FoF(B)
경로 I·II: ep_korea_2026.estimate (전년 같은 분기 기준).
검증: 2025년 각 분기를 B = 2024(경로 I·II는 2024년 같은 분기)로 추정해 공표된 비금융법인 총저축(DT_200Y162)과 비교.
→ output/ep_korea_2026_bridge.csv, output/paper/fig_korea_2026_bridge.(png|pdf)
"""
import numpy as np
import pandas as pd

import ep_korea_2026 as k
from make_report import OUT

QS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"]


def load():
    q, a, fa = k.load()
    g = pd.read_csv(k.RAW / "bok_gdp_exp.csv", comment="#").set_index("period")
    q["gdp"] = g.gdp
    a = a.join(g.gdp.rename(lambda x: int(x)).loc[lambda s: s.index.isin(a.index)]) if False else a
    ga = g[~g.index.str.contains("Q")].copy()
    ga.index = ga.index.astype(int)
    a["gdp"], a["gov_gfcf_x"] = ga.gdp, ga.gov_gfcf
    return q, a, fa


def method_c(q, a, fa, t, B):
    r = q.loc[t]
    inv = (r.gcf - r.gov_gfcf) * a.loc[B, "i_nfc"] / (a.loc[B, "s_nfc"] * 0 + (q.filter(like=str(B), axis=0).gcf.sum() - a.loc[B, "gov_gfcf_x"]))
    d = a.loc[B, "s_nfc"] - a.loc[B, "i_nfc"] - fa.loc[B, "nfc"]
    return inv + r.nl_nfc + r.gdp * d / a.loc[B, "gdp"], inv


def run():
    q, a, fa = load()
    rows = []
    for t in QS:
        y = int(t[:4])
        pq = f"{y - 1}{t[4:]}"
        sc, ic = method_c(q, a, fa, t, y - 1)
        b, n, e = k.estimate(q, a, [pq], [t], [pq])
        rows.append(dict(quarter=t, gdp=q.loc[t, "gdp"], actual=q.loc[t, "nfc_saving"], C=sc, I=e["nfc_I"], II=e["nfc_II"], inv_C=ic,
                         prev_actual=q.loc[pq, "nfc_saving"], prev_gdp=q.loc[pq, "gdp"]))
    r = pd.DataFrame(rows).set_index("quarter")
    pct = r[["actual", "C", "I", "II", "inv_C"]].div(r.gdp, axis=0) * 100
    pct["prev"] = r.prev_actual / r.prev_gdp * 100
    return q, a, fa, r, pct


def figure(a, pct, dec, info):
    import fig_paper as fp
    plt = fp.plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    yrs = [2013, 2014, 2015]
    x0 = np.arange(3)
    ax.bar(x0, 100 * a.loc[yrs, "s_nfc"] / a.loc[yrs, "gdp"], color="0.75", width=0.6, label="Published (annual / quarterly)")
    x1 = np.arange(len(pct)) + 4
    pub = pct.actual.to_numpy()
    ax.bar(x1[:4], pub[:4], color="0.75", width=0.6)
    mk = {"C": ("s", fp.CO, "Colleague's method (C)"), "II": ("^", "#7a5195", "Route II"), "I": ("o", fp.HH, "Route I")}
    for j, (m, (s, c, l)) in enumerate(mk.items()):
        ax.scatter(x1 + (j - 1) * 0.18, pct[m], marker=s, color=c, s=22, zorder=5, label=l)
    ax.axvline(7.5, color="0.5", lw=0.7, ls=":")
    ax.axvspan(7.5, 9.6, color="0.95", lw=0, zorder=0)
    ax.set_xticks(list(x0) + list(x1))
    ax.set_xticklabels([str(y) for y in yrs] + [f"{p[4:]}\n{p[:4]}" for p in pct.index], fontsize=7.6)
    ax.set_xlim(-0.6, 9.6)
    ax.set_ylim(12, 35)
    ax.set_ylabel("Percent of GDP")
    ax.set_title("(a) Nonfinancial corporate gross saving", loc="left", fontsize=9.3)
    ax.text(4.6, 34.2, "2025: out-of-sample test\n(base year 2024)", fontsize=6.8, color="0.35", va="top")
    ax.text(8.55, 34.2, "2026:\nestimates", fontsize=6.8, color="0.35", va="top", ha="center")
    ax.legend(loc="upper left", fontsize=6.8, frameon=False)
    ax.annotate("29.5 in the\nearlier draft", (x1[-1] - 0.18, pct.C.iloc[-1]), (x1[-1] - 2.6, 29.0), fontsize=6.5, color="0.3",
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
    ax = axes[1]
    cols = {"Corporate profits": fp.CO, "Corporate retention": "#7a5195", "Unallocated": "0.82"}
    for x, (m, c) in enumerate(dec.items()):
        pos = neg = 0.0
        for kk, v in c.items():
            ax.bar(x, v, bottom=pos if v >= 0 else neg, color=cols[kk], width=0.55, edgecolor="white", lw=0.3, label=kk if x == 0 else None)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        tot = c["Corporate profits"] + c["Corporate retention"]
        ax.scatter(x, tot, color="k", s=20, zorder=5, label="Change in corporate saving" if x == 0 else None)
    ax.axhline(0, color=".3", lw=.6)
    ax.set_xticks(range(len(dec)))
    ax.set_xticklabels(list(dec), fontsize=7.8)
    ax.set_ylabel("Percentage points of GNI")
    ax.set_title("(b) Corporate saving, H1 2025→H1 2026", loc="left", fontsize=9.3)
    ax.legend(loc="upper right", fontsize=6.8, frameon=False)
    ax.set_ylim(-2, 10)
    fig.suptitle("Three ways to estimate corporate saving agree that profits drove the 2026 rise", x=0.0, ha="left", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.text(0.0, -0.02,
             "Notes: (a) Grey bars: published gross saving of nonfinancial corporations (annual 2013–15; quarterly 2025). Markers: estimates. "
             "Method C (the earlier draft's footnote): investment = total gross capital formation less government fixed capital formation, scaled "
             "to the base year's corporate investment; saving = that investment + flow-of-funds net lending + the base year's capital-account "
             "adjustments and discrepancy as a share of GDP. Route I: national saving less published household saving, government saving "
             "(flow-of-funds net lending + investment + gap at the base share) and financial-corporation saving (base share). Route II: as C, "
             "but with base values from the same quarter a year earlier. 2025 quarters use 2024 as the base, so they test each method out of "
             f"sample. (b) Profit component = change in operating surplus (implied growth {info['nfc_gos_growth']:.0f}%) at the H1 2025 retention "
             "ratio, the same for all three; retention = the rest. Source: Bank of Korea; authors' calculations.",
             fontsize=6.6, wrap=True, va="top")
    fp.save(fig, "fig_korea_2026_bridge", sfx="")


if __name__ == "__main__":
    q, a, fa, r, pct = run()
    err = pct.loc["2025Q1":"2025Q4", ["C", "I", "II"]].sub(pct.loc["2025Q1":"2025Q4", "actual"], axis=0)
    derr = err.copy()   # 수준 오차 = 전년동기 대비 변화 오차(전년 값은 공표치)
    pd.set_option("display.width", 200)
    print("levels, % GDP"); print(pct.round(2).to_string())
    print("2025 out-of-sample errors (pp GDP)"); print(err.round(2).to_string())
    print("RMSE", np.sqrt((err ** 2).mean()).round(2).to_dict(), "mean", err.mean().round(2).to_dict())
    # H1 2025→H1 2026, GNI 대비: 세 방법의 기업 저축 변화와 이익/유보 분해
    b, n, e = k.estimate(q, a, ["2025Q1", "2025Q2"], ["2026Q1", "2026Q2"], ["2024Q1", "2024Q2"])
    sC = sum(method_c(q, a, fa, t, 2025)[0] for t in ("2026Q1", "2026Q2"))
    cI, dnat, info = k.shares(b, n, e, "nfc_I")
    P = lambda x, g: 100 * x / g
    prof = cI["Corporate profits"]
    dec = {}
    for lab, s_ in (("Route I\n(baseline)", e["nfc_I"]), ("Route II", e["nfc_II"]), ("Colleague's\nmethod (C)", sC)):
        ds = P(s_, n.gni) - P(b.nfc_saving, b.gni)
        dec[lab] = {"Corporate profits": prof, "Corporate retention": ds - prof, "Unallocated": 0.0}
    for kk, v in dec.items():
        print(kk.replace("\n", " "), {x: round(y, 2) for x, y in v.items()})
    out = pct.round(3).join(r[["actual", "C", "I", "II"]].add_suffix("_bn"))
    out.to_csv(OUT / "ep_korea_2026_bridge.csv")
    for kk in dec:
        dec[kk].pop("Unallocated")
    figure(a, pct, dec, info)
