"""식 (1)을 ρ로 쓴 조정액: transitory = φ_s(1−ρ)X₀ + (φ_s − φ_m)ρX₀ = (φ_s − φ_m ρ)X₀, underlying = φ_m ρ X₀.
BVAR 전가율(φ_s = 0.88, φ_m = 0.26)과 2026년 반도체 수지(기준선 대비 17.2%p)로 그리고, 가격 시나리오(2031년 P = 2.4, 4.5, 9.6)를 점으로 표시.
→ output/paper/fig_rho_paths.(png|pdf)
"""
import numpy as np

PHI_S, PHI_M, X0 = 0.88, 0.26, 17.2
SCEN = {"Faster": 2.4, "Baseline": 4.5, "Slower": 9.6}


def parts(rho):
    return dict(price=PHI_S * (1 - rho) * X0, absorb=(PHI_S - PHI_M) * rho * X0, under=PHI_M * rho * X0)


if __name__ == "__main__":
    import fig_paper as fp
    plt = fp.plt
    rho = np.linspace(0, 1, 101)
    p = parts(rho)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(rho, p["price"], color=fp.CO, lw=1.8, label="Price reversal  φ_s(1−ρ)X₀")
    ax.plot(rho, p["absorb"], color="#7a5195", lw=1.8, label="Absorption  (φ_s−φ_m)ρX₀")
    ax.plot(rho, p["price"] + p["absorb"], color="k", lw=2.2, label="Transitory, total  (φ_s−φ_m ρ)X₀")
    ax.plot(rho, p["under"], color=fp.HH, lw=1.8, ls="--", label="Underlying  φ_m ρ X₀")
    for name, P in SCEN.items():
        r = P / X0
        q = parts(r)
        for v, cl in ((q["price"], fp.CO), (q["absorb"], "#7a5195"), (q["price"] + q["absorb"], "k"), (q["under"], fp.HH)):
            ax.scatter(r, v, color=cl, s=20, zorder=5)
        ax.axvline(r, color="0.8", lw=0.6, ls=":")
        ax.text(r, 16.6, f"{name}\nρ={r:.2f}", ha="center", fontsize=6.8, color="0.35")
        print(name, f"rho={r:.3f}", {k: round(float(v), 2) for k, v in q.items()}, "total", round(float(q['price'] + q['absorb']), 2))
    ax.set_xlabel("Share of the 2026 chip balance still present in 2031, ρ = P / X₀")
    ax.set_ylabel("Points of GDP")
    ax.set_ylim(0, 18)
    ax.legend(loc="center left", fontsize=7, frameon=False, bbox_to_anchor=(0.0, 0.45))
    ax.set_title("Equation (1) in terms of persistence", loc="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.text(0.0, -0.02, "Notes: X₀ = 17.2 points of GDP (chip balance above baseline in 2026); φ_s = 0.88 and φ_m = 0.26 (BVAR, scheme 1). Dots: "
             "the three price scenarios, whose 2031 chip balances (2.4, 4.5 and 9.6 points) imply ρ.", fontsize=6.8, wrap=True, va="top")
    fp.save(fig, "fig_rho_paths", sfx="")
