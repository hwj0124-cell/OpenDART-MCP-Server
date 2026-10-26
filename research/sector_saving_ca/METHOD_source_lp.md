# Cross-country evidence: persistence of sector balances by source of the saving change

Draft text for the paper. Script: `ep_source_lp.py`; results: `output/ep_source_lp.csv`; figure: `output/paper/fig_source_lp.png`.

## Data

Annual OECD non-financial sector accounts (Table 14) for non-financial corporations and households (including non-profit institutions serving households), 1970–2025, in percent of nominal GDP. Net lending (B9) measures each sector's saving–investment balance. The estimation sample covers 38 economies.

## Separating the source of a change in sector saving

Let $s^C_{it}$ be gross saving of non-financial corporations, $\pi_{it}$ their gross operating surplus and $\rho_{it} = s^C_{it}/\pi_{it}$ their retention ratio. Let $s^H_{it}$ be gross household saving, $y_{it}$ gross household disposable income and $\sigma_{it}$ the household saving rate (percent of disposable income). All levels are in percent of GDP. The change in each sector's saving splits exactly into two parts:

$$
\Delta s^C_{it} = \underbrace{\rho_{i,t-1}\,\Delta\pi_{it}}_{x^P_{it}} + \underbrace{\pi_{it}\,\Delta\rho_{it}}_{x^R_{it}},
\qquad
\Delta s^H_{it} = \underbrace{y_{i,t-1}\,\Delta\sigma_{it}/100}_{x^H_{it}} + \underbrace{\sigma_{it}\,\Delta y_{it}/100}_{x^Y_{it}} .
$$

$x^P$ is the change in corporate saving that would occur if firms kept the change in profits at last year's retention ratio (profit-driven). $x^H$ is the change in household saving that comes from a change in the saving rate at last year's income (saving-rate-driven). $x^R$ and $x^Y$ are the remaining parts.

## Specification

For sector $j \in \{C, H\}$ with net lending $n^j$, and horizons $h = 0, \dots, 6$:

$$
n^j_{i,t+h} - n^j_{i,t-1} = \beta^{j,P}_h\, x^P_{it} + \beta^{j,H}_h\, x^H_{it} + \gamma^{j\prime}_h W_{it} + \alpha^j_{ih} + \delta^j_{th} + \varepsilon^j_{i,t+h},
$$

where $W_{it}$ contains $x^R_{it}$ and $x^Y_{it}$; two lags of $x^P$, $x^H$, $x^R$, $x^Y$, the change in the current account and real GDP growth; and the first lags of the levels of $s^C$, $s^H$, the current account, $\pi$ and $\sigma$. $\alpha$ and $\delta$ are country and year fixed effects. Both sources enter jointly, so each coefficient holds the other source fixed. There are no thresholds and no episode selection; every country-year contributes.

We report each sector's response to its own source, normalized by the impact response:

$$
\theta^C_h = \beta^{C,P}_h / \beta^{C,P}_0, \qquad \theta^H_h = \beta^{H,H}_h / \beta^{H,H}_0 ,
$$

the share of the initial rise in the sector's saving–investment balance that remains after $h$ years.

## Estimation and inference

- **Fixed sample.** The sample is restricted to country-years in which both sector balances are observed at every horizon $h = 0,\dots,6$ (start years up to 2019; 871 observations). The composition of the sample is therefore the same at every horizon.
- **Weights.** Two-step weighted least squares. The impact ($h = 0$) equation for each sector is first estimated by OLS. Each country is then weighted by the inverse of the variance of its residuals, and the same weights are used at every horizon. This keeps a few economies with very noisy sector accounts from dominating the estimates. The residual standard deviation of corporate net lending is 8.2 in Romania and 3.0–3.4 in Slovakia, Lithuania and Ireland, against a median of 1.5.
- **Inference.** Country-cluster bootstrap (499 replications). Countries are resampled with replacement, the weights are held fixed, and the normalized responses, their horizon averages and the corporate–household differences are recomputed in each replication. Bands are percentile intervals.

## Results

A rise in the corporate balance that comes from higher profits fades quickly. About 0.4 of it remains after one year, and it is fully gone after two years. A rise in the household balance that comes from a higher saving rate fades gradually: 0.8 remains after one year, 0.6 after two years, and 0.3–0.4 three to six years later.

| | Years 1–2 | Years 3–6 |
|---|---|---|
| Corporate, profit-driven | 0.18 [0.04, 0.39] | −0.17 [−0.26, 0.09] |
| Household, saving-rate-driven | 0.70 [0.59, 0.74] | 0.36 [0.21, 0.40] |
| Difference | −0.51 [−0.65, −0.26] | −0.52 [−0.61, −0.16] |

Average normalized response; 68 percent bootstrap bands in brackets. The 90 percent band of the difference excludes zero in years 1–2 (−0.75 to −0.08). In years 3–6 its upper end is 0.03.

**Robustness.** Restricting the sample to windows that end before 2020 (start years up to 2013; 675 observations) leaves the results unchanged: 0.24 and −0.13 for corporations, 0.69 and 0.37 for households. Without weights the corporate response is noisier but follows the same pattern (0.26 in years 1–2 and 0.14 in years 3–6, against 0.73 and 0.30 for households).

## Interpretation and limits

The comparison is between saving raised by higher profits and saving raised by a change in household behavior; it does not imply that all corporate saving is less persistent than household saving. Without separating the source, the persistence of corporate and household net lending is similar. $x^P$ captures all profit-driven changes, not only terms-of-trade windfalls, and the estimates are conditional correlations rather than causal effects. The evidence complements the Korea-specific pass-through estimates, which remain the main quantification.
