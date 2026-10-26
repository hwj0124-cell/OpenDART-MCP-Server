"""모형(동료 원고 4장)에서 두 가지 일시성 식을 도출한 노트(output/notes/model_two_sources_derivation.pdf)의 수치 검증.

1) 닫힌 해 (5)–(9)를 수치 최적화로 다시 풀어 확인(제약 없는 경제 u, 제약 있는 경제 c).
2) 횡재의 효과: ΔCA^u ≈ a·T + b·P (식 14, 1차 근사), ΔCA^c = (1−θ)W (식 17, 정확).
3) 중기: 지속적 횡재 P(ρ = 1)에 대한 제약 없는 경제의 반응 = b·P (식 18).
4) 일시적 부분 = ΔCA^c − ΔCA^med = φ_s·T + (φ_s − φ_m)·P (식 20), 동료 식 (4.5)와의 관계 (식 24).
근사 오차는 ε에 대해 2차(ε를 절반으로 줄이면 절대 오차는 약 1/4, W 대비 오차는 절반).
→ output/model_check.csv, output/paper/fig_model_check.(png|pdf)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from make_report import OUT

P0 = dict(TT=1.0, Q=1.0, R=1.05, beta=0.95, theta=0.12, Dbar=0.30, bbar=0.05)
P0["A2"] = P0["R"]                     # I0 = (TT·A2/2R)^2 = 0.25 → 투자/소득 0.25
P0["Ibar"] = 0.8 * (P0["TT"] * P0["A2"] / (2 * P0["R"])) ** 2


def frictionless(eps, rho, p=P0, numeric=False):
    TT1, TT2 = p["TT"] + eps, p["TT"] + rho * eps
    Y1 = TT1 * p["Q"]
    if not numeric:
        I = (TT2 * p["A2"] / (2 * p["R"])) ** 2
        C1 = (Y1 + I) / (1 + p["beta"])
        return dict(Y1=Y1, I=I, C1=C1, CA=Y1 - C1 - I)
    # 수치: 기업은 Y2 − R·I 최대화, 가계는 자유 차입으로 평생 효용 최대화
    I = minimize_scalar(lambda i: -(TT2 * p["A2"] * np.sqrt(i) - p["R"] * i), bounds=(1e-9, 10), method="bounded",
                        options=dict(xatol=1e-12)).x
    Vres = Y1 - I + TT2 * p["A2"] * np.sqrt(I) / p["R"]
    C1 = minimize_scalar(lambda c: -(np.log(c) + p["beta"] * np.log(p["R"] * (Vres - c))), bounds=(1e-9, Vres - 1e-9),
                         method="bounded", options=dict(xatol=1e-12)).x
    return dict(Y1=Y1, I=I, C1=C1, CA=Y1 - C1 - I)


def constrained(eps, rho, p=P0, numeric=False):
    TT1, TT2 = p["TT"] + eps, p["TT"] + rho * eps
    Y1, W = TT1 * p["Q"], eps * p["Q"]
    Iu = (TT2 * p["A2"] / (2 * p["R"])) ** 2
    I = min(Iu, p["Ibar"])
    D1 = p["Dbar"] + p["theta"] * W
    aF = Y1 - I - D1
    D2 = TT2 * p["A2"] * np.sqrt(I) + p["R"] * aF
    if not numeric:
        Chat = (D1 + D2 / p["R"]) / (1 + p["beta"])
        C1 = min(Chat, D1 + p["bbar"])
    else:
        # 가계: C1 + b = D1, C2 = D2 + R·b, b ≥ −b̄
        f = lambda b: -(np.log(D1 - b) + p["beta"] * np.log(D2 + p["R"] * b))
        lo, hi = -p["bbar"], D1 - 1e-9
        b = minimize_scalar(f, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-12)).x
        C1 = D1 - b
    return dict(Y1=Y1, I=I, C1=C1, CA=Y1 - C1 - I, binds_I=Iu > p["Ibar"], binds_L=C1 >= D1 + p["bbar"] - 1e-9)


def coefs(p=P0):
    I0 = (p["TT"] * p["A2"] / (2 * p["R"])) ** 2
    kappa = 2 * I0 / (p["TT"] * p["Q"])
    a = p["beta"] / (1 + p["beta"])
    b = (p["beta"] - (2 + p["beta"]) * kappa) / (1 + p["beta"])
    return a, b, 1 - p["theta"], kappa


def check(eps, rho, p=P0):
    a, b, phis, kappa = coefs(p)
    W = eps * p["Q"]
    T, Pp = (1 - rho) * W, rho * W
    u0, u1 = frictionless(0, rho, p), frictionless(eps, rho, p)
    c0, c1 = constrained(0, rho, p), constrained(eps, rho, p)
    dCAu, dCAc = u1["CA"] - u0["CA"], c1["CA"] - c0["CA"]
    # 중기: 프리미엄이 ρε로 줄어 지속(ρ' = 1), 제약 없음
    med = frictionless(rho * eps, 1.0, p)["CA"] - u0["CA"]
    trans = dCAc - med
    return dict(eps=eps, rho=rho, W=W, T=T, P=Pp,
                dCAu_exact=dCAu, dCAu_formula=a * T + b * Pp,
                dCAc_exact=dCAc, dCAc_formula=phis * W,
                med_exact=med, med_formula=b * Pp,
                trans_exact=trans, trans_formula=phis * T + (phis - b) * Pp,
                gap45_exact=dCAc - dCAu, gap45_formula=(phis - a) * T + (phis - b) * Pp,
                binds=bool(c0["binds_I"] and c0["binds_L"] and c1["binds_I"] and c1["binds_L"]))


def closed_vs_numeric(p=P0):
    rows = []
    for eps, rho in ((0.0, 0.5), (0.2, 0.0), (0.2, 0.5), (0.2, 1.0)):
        for name, f in (("frictionless", frictionless), ("constrained", constrained)):
            x, y = f(eps, rho, p), f(eps, rho, p, numeric=True)
            rows.append(dict(economy=name, eps=eps, rho=rho, CA_closed=x["CA"], CA_numeric=y["CA"], diff=x["CA"] - y["CA"]))
    return pd.DataFrame(rows)


def figure(r):
    import fig_paper as fp
    plt = fp.plt
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    q = r[r.eps == 0.1].sort_values("rho")
    for ax, (ex, fo, t) in zip(axes, (("dCAu_exact", "dCAu_formula", "(a) Frictionless response, eq. (14)"),
                                       ("med_exact", "med_formula", "(b) Medium run, eq. (18)"),
                                       ("trans_exact", "trans_formula", "(c) Transitory part, eq. (20)"))):
        ax.plot(q.rho, q[ex] / q.W, color=fp.HH, lw=2.2, label="Exact model solution")
        ax.plot(q.rho, q[fo] / q.W, color=fp.CO, lw=1.4, ls="--", label="Formula")
        ax.set_xlabel(r"Persistence of the price gain $\rho$")
        ax.set_title(t, loc="left", fontsize=9.2)
        ax.axhline(0, color=".3", lw=.6)
    axes[0].set_ylabel("Change in CA per unit of windfall $W$")
    axes[0].legend(fontsize=7.5, frameon=False)
    a, b, phis, kappa = coefs()
    fig.suptitle("The two-sources formula matches the exact model solution", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.text(0.0, -0.03,
             f"Notes: Windfall ε = 0.1 (10 percent of export income). Parameters: β = {P0['beta']}, R = {P0['R']}, θ = {P0['theta']}, "
             f"investment/income = 0.25 (κ = {kappa:.2f}), Ī = 0.8 × frictionless investment, so both constraints bind. Formula coefficients: "
             f"a = {a:.3f}, b = φ_m = {b:.3f}, φ_s = {phis:.2f}. Differences are second order in ε.", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_model_check", sfx="")


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print("closed form vs numerical optimization")
    print(closed_vs_numeric().round(6).to_string(index=False))
    a, b, phis, kappa = coefs()
    print(f"\na={a:.4f}  b=phi_m={b:.4f}  phi_s={phis:.4f}  kappa={kappa:.4f}")
    rows = [check(e, rho) for e in (0.2, 0.1, 0.05) for rho in np.linspace(0, 1, 11)]
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "model_check.csv", index=False)
    print("all cases bind:", r.binds.all())
    for e in (0.2, 0.1, 0.05):
        q = r[r.eps == e]
        err = {k: float((q[f"{k}_exact"] - q[f"{k}_formula"]).abs().max() / e) for k in ("dCAu", "dCAc", "med", "trans", "gap45")}
        print(f"eps={e}: max |exact − formula| / W:", {k: round(v, 5) for k, v in err.items()})
    print(r[(r.eps == 0.1) & r.rho.isin([0.0, 0.5, 1.0])][["rho", "T", "P", "dCAc_exact", "med_exact", "trans_exact", "trans_formula"]]
          .round(5).to_string(index=False))
    figure(r)
