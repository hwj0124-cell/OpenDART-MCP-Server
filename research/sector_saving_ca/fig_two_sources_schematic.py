"""두 가지 일시성 개념도: 상태(단기·중기·이후)를 가로축에 두고, (a) 반도체 수지(횡재) W → P, (b) 경상수지 효과
φ_s·W → φ_m·P와 그 사이의 가격 회귀 φ_s·T, 흡수 (φ_s−φ_m)·P, underlying φ_m·P를 수식으로 표시. 크기는 예시(축척 아님).
→ output/paper/fig_two_sources_schematic.(png|pdf)
"""
import numpy as np

import fig_paper as fp

W, RHO, PS, PM = 1.0, 0.45, 0.88, 0.40      # 개념도용 값(축척 아님)
P = RHO * W
X = [0, 1, 2]
LAB = ["Short run\n(constraints bind,\npremium $\\varepsilon$)", "Medium run\n(constraints relaxed,\npremium $\\rho\\varepsilon$)",
       "Beyond\n(premium seen as\nnearly permanent)"]


def brace(ax, x, y0, y1, text, color, side=1, dx=0.06):
    ax.annotate("", xy=(x, y0), xytext=(x, y1), arrowprops=dict(arrowstyle="<->", color=color, lw=1.2, shrinkA=0, shrinkB=0))
    ax.text(x + side * dx, (y0 + y1) / 2, text, color=color, fontsize=8.6, va="center", ha="left" if side > 0 else "right")


def main():
    plt = fp.plt
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    ax = axes[0]
    ax.plot(X, [W, P, P], color="k", lw=2.2, marker="o", ms=5)
    ax.fill_between([0, 1], [W, P], [P, P], color=fp.CO, alpha=0.12, lw=0)
    ax.fill_between(X, [P, P, P], 0, color=fp.HH, alpha=0.10, lw=0)
    brace(ax, 0.0, P, W, r"$T=(1-\rho)W$", fp.CO, side=1)
    brace(ax, 1.55, 0, P, r"$P=\rho W$", fp.HH, side=1)
    ax.text(0.0, W + 0.05, r"$W=\varepsilon\bar Q$", ha="center", fontsize=9)
    ax.set_title("(a) Windfall: chip balance relative to baseline", loc="left", fontsize=9.5)
    ax = axes[1]
    ax.plot(X, [PS * W, PM * P, PM * P], color="k", lw=2.2, marker="o", ms=5, label="Effect on the current account")
    ax.plot([0, 1], [PS * W, PS * P], color="0.45", lw=1.2, ls="--", label=r"Pass-through held at $\varphi_s$")
    brace(ax, 1.0, PS * P, PS * W, r"Price reversal  $\varphi_s T$", fp.CO, side=1)
    brace(ax, 1.0, PM * P, PS * P, r"Absorption  $(\varphi_s-\varphi_m)P$", "#7a5195", side=1)
    brace(ax, 2.0, 0, PM * P, r"Underlying  $\varphi_m P$", fp.HH, side=-1)
    ax.text(0.0, PS * W + 0.05, r"$\varphi_s W$", ha="center", fontsize=9)
    ax.plot([0, 1], [PS * W, PS * W], color="0.75", lw=0.8, ls=":")
    ax.text(0.93, PS * P - 0.01, r"$\varphi_s P$", ha="right", va="top", fontsize=8.5, color="0.35")
    ax.set_title("(b) Effect on the current account", loc="left", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    for ax in axes:
        ax.axhline(0, color=".3", lw=.6)
        ax.set_xticks(X)
        ax.set_xticklabels(LAB, fontsize=7.8)
        ax.set_xlim(-0.35, 2.5)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    fig.suptitle("Two sources of temporariness", x=0.0, ha="left", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.text(0.0, -0.04,
             r"Notes: Schematic, not to scale. $\varphi_s=1-\theta$: short-run pass-through while investment and household spending are "
             r"constrained ($\theta$: share of the windfall paid out at once). $\varphi_m=\delta/(r+\delta)$: medium-run pass-through once the "
             r"constraints are relaxed, when the remaining premium is expected to erode at rate $\delta$. Transitory $=\varphi_s T+(\varphi_s-\varphi_m)P$; "
             r"underlying $=\varphi_m P$.", fontsize=7.2, wrap=True, va="top")
    fp.save(fig, "fig_two_sources_schematic", sfx="")


if __name__ == "__main__":
    main()
