"""패널 국소투영(LP) 도구: 이원 고정효과 OLS/2SLS, 국가 군집 표준오차, 국가 블록 부트스트랩."""
import numpy as np
import pandas as pd


def balance(df, unit="country", time="year"):
    """국가별 연도 격자를 채워 shift 가 실제 연도 차이를 따르도록 한다."""
    parts = []
    for c, g in df.groupby(unit):
        yrs = pd.RangeIndex(g[time].min(), g[time].max() + 1, name=time)
        parts.append(g.set_index(time).reindex(yrs).assign(**{unit: c}).reset_index())
    return pd.concat(parts, ignore_index=True).sort_values([unit, time])


def add_lp_columns(df, y, h, unit="country"):
    """y_{t+h} − y_{t−1}"""
    g = df.groupby(unit)[y]
    return g.shift(-h) - g.shift(1)


def lagged(df, cols, lags, unit="country"):
    out = {}
    for c in cols:
        g = df.groupby(unit)[c]
        for l in range(1, lags + 1):
            out[f"L{l}_{c}"] = g.shift(l)
    return pd.DataFrame(out, index=df.index)


def _fe_matrix(d, fe):
    mats = [np.ones((len(d), 1))]
    for f in fe:
        mats.append(pd.get_dummies(d[f], drop_first=True, dtype=float).to_numpy())
    return np.hstack(mats)


def _cluster_meat(Xu, groups):
    meat = np.zeros((Xu.shape[1], Xu.shape[1]))
    for gidx in groups:
        s = Xu[gidx].sum(axis=0)
        meat += np.outer(s, s)
    return meat


def _inv(A):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def _groups(cl):
    codes, uniq = pd.factorize(cl)
    return [np.flatnonzero(codes == k) for k in range(len(uniq))]


def ols(d, y, x, controls, fe=("country", "year"), cluster="country", se=True):
    """y on x (+controls, FE). 반환: (coef[x], cov[x], nobs)"""
    cols = [y] + list(x) + list(controls)
    d = d[cols + list(dict.fromkeys([*fe, cluster]))].dropna(subset=cols)
    X = np.hstack([d[list(x) + list(controls)].to_numpy(float), _fe_matrix(d, fe)])
    Y = d[y].to_numpy(float)
    XtX_inv = _inv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    k = len(x)
    if not se:
        return b[:k], None, len(d)
    u = Y - X @ b
    G = d[cluster].nunique()
    n, K = X.shape
    meat = _cluster_meat(X * u[:, None], _groups(d[cluster]))
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - K))
    return b[:k], V[:k, :k], len(d)


def iv(d, y, endog, instr, controls, fe=("country", "year"), cluster="country", se=True):
    """단일 내생변수 2SLS. 반환: (beta, se, nobs, 1단계 F)"""
    cols = list(dict.fromkeys([y, endog, instr] + list(controls)))
    d = d[cols + list(dict.fromkeys([*fe, cluster]))].dropna(subset=cols)
    W = np.hstack([d[list(controls)].to_numpy(float), _fe_matrix(d, fe)])
    Z = np.hstack([d[[instr]].to_numpy(float), W])
    X = np.hstack([d[[endog]].to_numpy(float), W])
    Y = d[y].to_numpy(float)
    ZtZ_inv = _inv(Z.T @ Z)
    pi = ZtZ_inv @ Z.T @ X[:, 0]
    Xhat = X.copy()
    Xhat[:, 0] = Z @ pi
    A = _inv(Xhat.T @ X)
    b = A @ Xhat.T @ Y
    if not se:
        return b[0], None, len(d), None
    u = Y - X @ b
    groups = _groups(d[cluster])
    G, n, K = len(groups), *X.shape
    B = _inv(Xhat.T @ Xhat)
    V = B @ _cluster_meat(Xhat * u[:, None], groups) @ B * (G / (G - 1)) * ((n - 1) / (n - K))
    # 1단계 F (군집강건, 단일 도구)
    v = X[:, 0] - Z @ pi
    Vz = ZtZ_inv @ _cluster_meat(Z * v[:, None], groups) @ ZtZ_inv * (G / (G - 1))
    F = pi[0] ** 2 / Vz[0, 0]
    return b[0], np.sqrt(V[0, 0]), len(d), F


def cluster_bootstrap(d, fn, reps=499, unit="country", seed=0):
    """국가 단위 복원추출. fn(df) → 1차원 배열. 반환: (reps × k) 배열"""
    rng = np.random.default_rng(seed)
    units = d[unit].unique()
    by = {u: g for u, g in d.groupby(unit)}
    out = []
    for _ in range(reps):
        draw = rng.choice(units, size=len(units), replace=True)
        # 같은 국가가 여러 번 뽑히면 별개 단위로 취급(고유 ID 부여)
        bd = pd.concat([by[u].assign(**{unit: f"{u}#{i}"}) for i, u in enumerate(draw)],
                       ignore_index=True)
        try:
            out.append(np.asarray(fn(bd), float))
        except np.linalg.LinAlgError:
            continue
    return np.vstack(out)
