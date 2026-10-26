"""국가 군집 부트스트랩을 빠르게: 국가별 정규방정식 블록을 미리 계산해 두고 반복마다 뽑힌 횟수로 가중 합산.

y = X b + (국가 FE) + (연도 FE) + e, 가중치는 국가 단위 상수(2단계 WLS 가중치, 반복에서 고정).
- 국가 FE: 국가 안에서 평균을 빼면 제거됨(FWL). 같은 국가가 여러 번 뽑혀도 각 사본의 평균 제거 블록은 원래 블록과 같음
  → 사본을 별개 국가로 취급하는 통상의 군집 부트스트랩과 정확히 같은 추정치.
- 연도 FE: 국가 내 평균을 뺀 연도 더미를 설명변수로 그대로 둠(사본 간 공유).
반복마다: A* = Σ_c n_c A_c, B* = Σ_c n_c B_c, b* = A*⁻¹ B* (작은 행렬 하나).
"""
import numpy as np
import pandas as pd


class Design:
    def __init__(self, q, X, C, w, ys, unit="country", time="year"):
        r = q.dropna(subset=X + C + ys + [w])
        yrs = pd.get_dummies(r[time], dtype=float)
        Z = np.hstack([r[X + C].to_numpy(float), yrs.to_numpy()[:, 1:]])
        Y = r[ys].to_numpy(float)
        g = r[unit].to_numpy()
        Zd = Z - pd.DataFrame(Z).groupby(g).transform("mean").to_numpy()
        Yd = Y - pd.DataFrame(Y).groupby(g).transform("mean").to_numpy()
        W = r[w].to_numpy(float)
        self.units = pd.unique(g)
        idx = {u: np.where(g == u)[0] for u in self.units}
        self.A = np.stack([(Zd[i] * W[i, None]).T @ Zd[i] for i in idx.values()])
        self.B = np.stack([(Zd[i] * W[i, None]).T @ Yd[i] for i in idx.values()])
        self.ys = ys
        self.n = len(r)

    def coef(self, counts=None):
        """counts: 국가별 뽑힌 횟수(self.units 순서). 반환: 첫 설명변수 계수(ys 순서)."""
        if counts is None:
            A, B = self.A.sum(0), self.B.sum(0)
        else:
            A, B = np.tensordot(counts, self.A, 1), np.tensordot(counts, self.B, 1)
        sol = np.linalg.lstsq(A, B, rcond=None)[0]
        return pd.Series(sol[0], index=self.ys)


def bootstrap(designs, stat, reps=999, seed=0):
    """designs: {이름: Design}. stat: {이름: 계수 Series} → 1차원 배열. 반환: (점추정, reps×k 배열)."""
    units = pd.unique(np.concatenate([d.units for d in designs.values()]))
    pos = {k: pd.Index(units).get_indexer(d.units) for k, d in designs.items()}
    pt = stat({k: d.coef() for k, d in designs.items()})
    rng = np.random.default_rng(seed)
    out = np.empty((reps, len(pt)))
    for b in range(reps):
        cnt = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)).astype(float)
        out[b] = stat({k: d.coef(cnt[pos[k]]) for k, d in designs.items()})
    return pt, out
