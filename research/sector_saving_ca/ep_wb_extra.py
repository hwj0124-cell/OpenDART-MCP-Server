"""이익 횡재 vs 저축 행태 변화: 추가 점검 세 가지 (ep_windfall_behavior.py와 같은 표본·통제·가중치).

(1) 다른 부문 성분을 같은 해에 통제: 기업 식 + x_H, x_Y / 가계 식 + x_P, x_R.
(2) 이익 횡재 뒤 다른 부문과 경상수지: 기업 식 설계로 기업·가계·정부 S−I와 경상수지의 누적 변화(기업 S−I 시작 연도 반응 = 1).
    배당·세금이 가계·정부로 넘어가 거기서 일부 저축되면 기업 S−I가 0이 되어도 경상수지는 남을 수 있음 → 한국 BVAR(경상수지 기준)과 비교.
(3) 상태의존: 교역조건(상품) 개선 해 D_t = 1[ToT 증가율 > 0]. x_P·D, x_P·(1−D), D를 함께 넣음(x_R·통제는 그대로).
    결과: 기업 S−I, 투자, 영업잉여, 순배당. 교역조건 자료가 있는 국가·연도만.
구간: 국가 군집 델타 방법(시점·결과변수별 계수의 국가 단위 영향함수로 비율·차이의 공분산 계산). 결과변수별로
    모든 시점이 관측되는 행만 써서, 시점마다 결측을 따로 빼는 본 추정과 표본이 약간 다를 수 있음.
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


def design(q, X, C, w, ys):
    """같은 우변으로 여러 좌변을 한 번에. 반환: 첫 설명변수 계수(m), 국가별 영향함수(G×m, 국가 라벨)."""
    r = q.dropna(subset=X + C + ys + [w])
    M = np.hstack([r[X + C].to_numpy(float), lt._fe_matrix(r, ("country", "year"))])
    W = r[w].to_numpy(float)
    A = np.linalg.pinv((M * W[:, None]).T @ M)
    Y = r[ys].to_numpy(float)
    B = A @ (M * W[:, None]).T @ Y
    U = Y - M @ B
    a = (M @ A[0]) * W                                   # 첫 계수의 관측치별 가중
    psi = pd.DataFrame(a[:, None] * U, columns=ys).groupby(r.country.to_numpy()).sum()
    return pd.Series(B[0], index=ys), psi


class Delta:
    """여러 설계의 계수를 모아 비율·차이의 델타 방법 구간을 계산."""
    def __init__(self):
        self.b, self.psi = {}, []

    def add(self, tag, b, psi):
        for k, v in b.items():
            self.b[f"{tag}|{k}"] = v
        self.psi.append(psi.add_prefix(f"{tag}|"))

    def finish(self):
        P = pd.concat(self.psi, axis=1).fillna(0.0)
        G = len(P)
        self.keys = list(P.columns)
        self.V = P.to_numpy().T @ P.to_numpy() * G / (G - 1)
        self.bv = np.array([self.b[k] for k in self.keys])

    def fn(self, f):
        """f: 계수 dict → 스칼라. 수치 미분으로 기울기."""
        d = dict(zip(self.keys, self.bv))
        v0 = f(d)
        g = np.zeros(len(self.keys))
        for i, k in enumerate(self.keys):
            e = 1e-6 * max(1.0, abs(d[k]))
            dd = dict(d); dd[k] += e
            g[i] = (f(dd) - v0) / e
        return v0, np.sqrt(g @ self.V @ g)


def ycols(v):
    return [f"{v}_h{h}" for h in range(H)]


def run():
    _, q = wb.data()
    q = wb.weights(q)
    D = Delta()
    for s, o in (("corp", "hh"), ("hh", "corp")):
        m, mo = SEC[s], SEC[o]
        for lab, X in (("own", [m["x"], m["z"]]), ("cross", [m["x"], m["z"], mo["x"], mo["z"]])):
            D.add(f"cross_{s}_{lab}", *design(q, X, wb.ctrl(s), f"w_{s}", ycols(m["y"])))
    sp = sum((ycols(v) for v in ("nl_corp", "nl_hh", "nl_gov", "ca", "prof")), [])
    D.add("spill", *design(q, [SEC["corp"]["x"], SEC["corp"]["z"]], wb.ctrl("corp"), "w_corp", sp))
    qs = wb.weights(q.dropna(subset=["tot_goods_g"]).copy())
    qs["D"] = (qs.tot_goods_g > 0).astype(float)
    qs["xP_up"], qs["xP_dn"] = qs.x_P * qs.D, qs.x_P * (1 - qs.D)
    stv = sum((ycols(v) for v in ("nl_corp", "gcf_corp", "prof", "div_net")), [])
    for st, X in (("up", ["xP_up", "xP_dn", "x_R", "D"]), ("dn", ["xP_dn", "xP_up", "x_R", "D"])):
        D.add(f"state_{st}", *design(qs, X, wb.ctrl("corp"), "w_corp", stv))
    D.finish()

    rows = []
    def emit(key, f):
        for h in range(H):
            b, se = D.fn(lambda d, h=h: f(d, h))
            rows.append(dict(key=key, h=h, b=b, se=se, lo68=b - se, hi68=b + se, lo90=b - 1.645 * se, hi90=b + 1.645 * se))
    for s in ("corp", "hh"):
        y = SEC[s]["y"]
        for lab in ("own", "cross"):
            t = f"cross_{s}_{lab}"
            emit(t, lambda d, h, t=t, y=y: d[f"{t}|{y}_h{h}"] / d[f"{t}|{y}_h0"])
        emit(f"cross_{s}_diff", lambda d, h, y=y: d[f"cross_{s}_cross|{y}_h{h}"] / d[f"cross_{s}_cross|{y}_h0"]
             - d[f"cross_{s}_own|{y}_h{h}"] / d[f"cross_{s}_own|{y}_h0"])
    for v in ("nl_corp", "nl_hh", "nl_gov", "ca", "prof"):
        emit(f"spill_{v}", lambda d, h, v=v: d[f"spill|{v}_h{h}"] / d["spill|nl_corp_h0"])
    for st in ("up", "dn"):
        for v in ("nl_corp", "gcf_corp", "prof", "div_net"):
            emit(f"state_{st}_{v}", lambda d, h, st=st, v=v: d[f"state_{st}|{v}_h{h}"] / d[f"state_{st}|nl_corp_h0"])
    emit("state_diff_nl_corp", lambda d, h: d[f"state_up|nl_corp_h{h}"] / d["state_up|nl_corp_h0"]
         - d[f"state_dn|nl_corp_h{h}"] / d["state_dn|nl_corp_h0"])
    # 기간 평균
    avg = []
    for key, f in (("spill_ca", lambda d, lo, hi: np.mean([d[f"spill|ca_h{h}"] for h in range(lo, hi)]) / d["spill|nl_corp_h0"]),
                   ("spill_nl_gov", lambda d, lo, hi: np.mean([d[f"spill|nl_gov_h{h}"] for h in range(lo, hi)]) / d["spill|nl_corp_h0"]),
                   ("spill_nl_hh", lambda d, lo, hi: np.mean([d[f"spill|nl_hh_h{h}"] for h in range(lo, hi)]) / d["spill|nl_corp_h0"]),
                   ("state_diff_nl_corp", lambda d, lo, hi: np.mean([d[f"state_up|nl_corp_h{h}"] / d["state_up|nl_corp_h0"]
                                                                     - d[f"state_dn|nl_corp_h{h}"] / d["state_dn|nl_corp_h0"] for h in range(lo, hi)]))):
        for lab, lo, hi in (("1-2", 1, 3), ("3-6", 3, 7)):
            b, se = D.fn(lambda d: f(d, lo, hi))
            avg.append(dict(key=key, years=lab, b=b, se=se, lo90=b - 1.645 * se, hi90=b + 1.645 * se))
    r = pd.DataFrame(rows).assign(n=len(q), n_state=len(qs), k_state=qs.country.nunique(), share_up=qs.D.mean())
    r.to_csv(OUT / "ep_wb_extra.csv", index=False)
    pd.DataFrame(avg).to_csv(OUT / "ep_wb_extra_avg.csv", index=False)
    print(pd.DataFrame(avg).round(2).to_string(index=False))
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
    ax.legend(loc="lower center", fontsize=7, frameon=False, ncol=2)
    ax.set_ylim(-0.95, 1.1)
    fig.suptitle("Other sectors' components at t, and where the windfall ends up", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.text(0.0, -0.06, "Notes: Same local projections as the main persistence figure; bands from the country-cluster delta method. (a)–(b): the dashed line adds the other sector's two components at t. "
             "(c): cumulative change in each sector's balance and in the current account after a profit windfall, from the corporate equation, scaled by "
             "the impact response of the corporate balance; the shaded area is the 68 percent band for the current account (country-cluster delta method). Balances need not add up to "
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
             "Each response is scaled by the impact response of the corporate balance in that state. 68 percent bands, country-cluster delta method.",
             fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_wb_state", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_wb_extra.csv") if "--render" in sys.argv else run()
    figs(r)
    pd.set_option("display.width", 200)
    print(r.pivot_table(index="key", columns="h", values="b", sort=False).round(2).to_string())
    a = r.assign(per=np.where(r.h.between(1, 2), "1-2", np.where(r.h >= 3, "3-6", "0")))
    print(a[a.per != "0"].pivot_table(index="key", columns="per", values="b", sort=False).round(2).to_string())
