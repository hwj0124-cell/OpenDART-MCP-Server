"""2026 상반기: 실물–금융 통계 불일치(D)의 변화를 부문에 명시적으로 배분하는 추정.

부문 j(GNI 대비 %p 변화):  ΔS_j = Δ(I_j + NL_j^FoF) + ΔD_j,   Σ_j ΔD_j = ΔD (국민저축 − 총투자 − 국내 순자금운용, 관측)
가계: S_HH 공표 → ΔD_HH는 공표치에 맞춰 정해짐(가계 투자는 민간 총자본형성의 2025년 연간 비중, 가정 A4).
나머지 R = ΔD − ΔD_HH를 비금융법인·금융법인·정부에 배분하는 규칙:
  equal   : 1/3씩
  history : 연간 2011–2025 ΔD_j를 ΔD에 회귀한 적재치(원점 통과)를 세 부문 안에서 재정규화
  all_nfc : 전부 비금융법인(이전 경로 I과 같은 귀속 방향)
  none    : 배분하지 않음(ΔD_j = 0, 순자금운용만 신뢰) → 국민저축과 R만큼 어긋남
D_j의 수준은 부문마다 크므로(비금융법인 GNI의 약 5%, 가계 약 −6%) D=0을 수준으로 둘 수는 없고, 변화만 0으로 둘 수 있음(none).
분해: 이익 성분 = 2025 상반기 유보율 × Δ(영업잉여/GNI) (모든 규칙 공통), 유보 성분 = ΔS_NFC − 이익 성분.
→ output/ep_korea_2026_alloc.csv, output/paper/fig_korea_2026_alloc.(png|pdf)
"""
import numpy as np
import pandas as pd

import ep_korea_2026 as k
from make_report import OUT

RULES = {"equal": "Equal thirds", "history": "Historical loadings", "all_nfc": "All to corporations", "none": "Not allocated"}


def loadings(a, fa):
    P = lambda x: 100 * x / a.gndi
    D = pd.DataFrame({s: P(a[f"s_{s}"] - a[f"i_{s}"] - fa[s]) for s in k.SEC})
    d = D.diff().dropna()
    tot = d.sum(axis=1)
    beta = {s: float((d[s] * tot).sum() / (tot ** 2).sum()) for s in k.SEC}
    return pd.Series(beta)


def pieces(q, a):
    b, n, e = k.estimate(q, a, ["2025Q1", "2025Q2"], ["2026Q1", "2026Q2"], ["2024Q1", "2024Q2"])
    S = lambda x, g: 100 * x / g
    w = a.loc[2025, ["i_nfc", "i_fc", "i_hh"]]
    w = w / w.sum()
    ipb, ipn = b.gcf - b.gov_gfcf, n.gcf - n.gov_gfcf
    inv = {"nfc": (w.i_nfc * ipb, w.i_nfc * ipn), "fc": (w.i_fc * ipb, w.i_fc * ipn), "hh": (w.i_hh * ipb, w.i_hh * ipn),
           "gov": (b.gov_gfcf, n.gov_gfcf)}
    dIN = {s: S(n[f"nl_{s}"] + inv[s][1], n.gni) - S(b[f"nl_{s}"] + inv[s][0], b.gni) for s in k.SEC}
    dD = S(e["D"], n.gni) - S(e["D_base"], b.gni)
    dS_hh = S(n.hh_saving, n.gni) - S(b.hh_saving, b.gni)
    dD_hh = dS_hh - dIN["hh"]
    return b, n, e, pd.Series(dIN), dD, dD_hh


def allocate(rule, R, beta):
    if rule == "equal":
        w = pd.Series(1 / 3, index=["nfc", "fc", "gov"])
    elif rule == "history":
        w = beta[["nfc", "fc", "gov"]] / beta[["nfc", "fc", "gov"]].sum()
    elif rule == "all_nfc":
        w = pd.Series({"nfc": 1.0, "fc": 0.0, "gov": 0.0})
    else:
        w = pd.Series(0.0, index=["nfc", "fc", "gov"])
    return R * w


def decompose(b, n, e, dIN, dD_alloc, dD_hh):
    S = lambda x, g: 100 * x / g
    dS = dIN + dD_alloc.reindex(dIN.index).fillna(0)
    dS["hh"] = dIN["hh"] + dD_hh
    ret0 = b.nfc_saving / b.nfc_gos
    prof = ret0 * (S(e["nfc_gos"], n.gni) - S(b.nfc_gos, b.gni))
    sig0, sig1 = b.hh_saving / b.hh_gndi, n.hh_saving / n.hh_gndi
    spend = -(S(n.gov_cons_x, n.gni) - S(b.gov_cons, b.gni))
    c = {"Corporate profits": prof, "Corporate retention": dS["nfc"] - prof,
         "Household saving rate": S(b.hh_gndi, b.gni) * (sig1 - sig0), "Household income": sig1 * (S(n.hh_gndi, n.gni) - S(b.hh_gndi, b.gni)),
         "Government revenue": dS["gov"] - spend, "Government spending": spend, "Financial corporations": dS["fc"]}
    dnat = S(e["nat"], n.gni) - S(b.nat_saving, b.gni)
    c["Unallocated"] = dnat - sum(c.values())
    nfc1 = b.nfc_saving / b.gni + dS["nfc"] / 100
    ret1 = nfc1 * n.gni / e["nfc_gos"]
    return pd.Series(c), dS, dnat, ret0, ret1


def figure(res, alloc, dD, dD_hh, beta):
    import fig_paper as fp
    plt = fp.plt
    pal = {"Corporate profits": fp.CO, "Corporate retention": "#7a5195", "Household saving rate": fp.HH, "Household income": "#7f9cc9",
           "Government revenue": "#6a8f3c", "Government spending": "#b3cf8f", "Financial corporations": "0.55", "Unallocated": "0.85"}
    lab = {"Government revenue": "Government disposable income", "Government spending": "Government consumption (−)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    for x, (r, (c, dnat)) in enumerate(res.items()):
        pos = neg = 0.0
        for kk, v in c.items():
            if abs(v) < 1e-9:
                continue
            ax.bar(x, v, bottom=pos if v >= 0 else neg, color=pal[kk], width=0.58, edgecolor="white", lw=0.3,
                   label=lab.get(kk, kk) if x == 0 or (kk == "Unallocated" and r == "none") else None)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        ax.scatter(x, dnat, color="k", s=22, zorder=5, label="Change in national saving" if x == 0 else None)
    ax.axhline(0, color=".3", lw=.6)
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels([RULES[r].replace(" ", "\n", 1) + f"\n(retention {res[r][0]['Corporate retention']:+.1f})" for r in res], fontsize=7.6)
    ax.set_ylim(-3, 17.5)
    ax.set_ylabel("Percentage points of GNI")
    ax.set_title("(a) Sources of the change in national saving,\nH1 2025→H1 2026, by allocation rule", loc="left", fontsize=9.2)
    ax.legend(loc="upper center", fontsize=6.3, frameon=False, ncol=3)
    ax = axes[1]
    cols = {"hh": fp.HH, "nfc": fp.CO, "fc": "0.55", "gov": "#6a8f3c"}
    names = {"hh": "Households (implied by published saving)", "nfc": "Nonfinancial corporations", "fc": "Financial corporations", "gov": "Government"}
    for x, (r, al) in enumerate(alloc.items()):
        pos = 0.0
        for s in ("hh", "nfc", "fc", "gov"):
            v = dD_hh if s == "hh" else al.get(s, 0.0)
            if abs(v) < 1e-9:
                continue
            ax.bar(x, v, bottom=pos, color=cols[s], width=0.58, edgecolor="white", lw=0.3, label=names[s] if r == "equal" else None)
            pos += v
    ax.axhline(dD, color="k", lw=0.9, ls="--", label=f"Observed change in discrepancy ({dD:+.1f})")
    ax.set_xticks(range(len(alloc)))
    ax.set_xticklabels([RULES[r].replace(" ", "\n", 1) for r in alloc], fontsize=7.8)
    ax.set_ylabel("Percentage points of GNI")
    ax.set_ylim(0, 4.2)
    ax.set_title("(b) Who absorbs the change in the\nreal–financial discrepancy", loc="left", fontsize=9.2)
    ax.legend(loc="upper right", fontsize=6.4, frameon=False)
    fig.tight_layout()
    fig.text(0.0, -0.03,
             "Notes: Each sector's saving = its investment + its flow-of-funds net lending + D, where D collects capital transfers, non-produced "
             "assets and the gap between the real and financial accounts. The total change in D is published (national saving − investment − "
             "domestic net lending). Households' share follows from their published saving (their investment is private investment at its 2025 "
             "share). The rest is split across nonfinancial corporations, financial corporations and government: equally; by historical loadings "
             f"(regression of each sector's annual change in D on the total, 2011–2025: corporations {beta['nfc']:.2f}, financial "
             f"{beta['fc']:.2f}, government {beta['gov']:.2f}, households {beta['hh']:.2f}); entirely to corporations; or not at all. Profit "
             "component = change in operating surplus at the H1 2025 retention ratio, common to all rules; retention component = the rest of the "
             "change in corporate saving. Source: Bank of Korea; authors' calculations.", fontsize=6.6, wrap=True, va="top")
    fp.save(fig, "fig_korea_2026_alloc", sfx="")


if __name__ == "__main__":
    q, a, fa = k.load()
    beta = loadings(a, fa)
    b, n, e, dIN, dD, dD_hh = pieces(q, a)
    R = dD - dD_hh
    res, alloc, rows = {}, {}, {}
    for r in RULES:
        al = allocate(r, R, beta)
        c, dS, dnat, ret0, ret1 = decompose(b, n, e, dIN, al, dD_hh)
        res[r], alloc[r] = (c, dnat), al
        rows[r] = pd.concat([c, dS.add_prefix("dS_"), al.add_prefix("dD_"), pd.Series({"ret0": ret0, "ret1": ret1, "dnat": dnat})])
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ep_korea_2026_alloc.csv")
    pd.set_option("display.width", 200)
    print("loadings", beta.round(2).to_dict())
    print("Δ(I+NL)", dIN.round(2).to_dict(), "ΔD", round(dD, 2), "ΔD_hh", round(dD_hh, 2), "R", round(R, 2))
    print(out.round(2).to_string())
    figure(res, alloc, dD, dD_hh, beta)
