# 로컬 PC에서 다시 돌리기 (최종 사양)

## 준비

- Python 3.10 이상
- `pip install pandas numpy matplotlib`
- playwright는 없어도 됩니다. 없으면 한글 HTML 미리보기 PNG만 건너뛰고, 추정 결과(csv)와 논문 그림(`output/paper/`)은 그대로 만들어집니다.
- 이 폴더(`sector_saving_ca`)에서 실행합니다.

## 자료

- `data/panel.csv`: 분석에 쓰는 국가×연도 패널(GDP 대비 %). 아래 결과는 모두 이 파일에서 나옵니다.
- `data/raw/`: 패널을 처음부터 다시 만들 때 쓰는 원자료(OECD Table 14 전체본, GDP, 상품 교역, WDI 디플레이터).
  다시 만들려면 `python build_panel.py` (→ `data/panel.csv`, `data/panel_gni.csv`).

## 실행 순서 (주 사양: 2단계 가중최소제곱, S−I 정규화, 사건 전 2개 시차 통제)

| 순서 | 명령 | 결과 |
|---|---|---|
| 1 | `python ep_report.py --fgls --nl --lags` | `output/ep_report_fgls_nl_lags.csv` (Figure 1, 반응·구간·차이) |
| 2 | `python ep_report_extra.py --lags` | `output/ep_report_extra_lags.csv` (가계 저축률·소비·경상수지) |
| 3 | `python render_ep_bars.py --lags` | `output/ep_report_bars_lags.csv` (Figure 2 기업 분해) |
| 4 | `python ep_report_growth.py` | `output/ep_report_diag_lags.csv`, `ep_report_growth_lags.csv` (Figure 3·5·6) |
| 5 | `python ep_report_leads.py --lags` | `output/ep_report_leads_lags.csv` (사전 추세 −5·−4기) |
| 6 | `python ep_corp_ratios.py` | `output/ep_corp_ratios_lags.csv` (Figure 4 기업 재투자율·배당성향) |
| 7 | `python fig_paper.py --lags` | `output/paper/fig1~6_*_lags.png/pdf` |

점검용:

- `python ep_balanced.py`: t+6까지 관측되는 사례만 남긴 균형 표본 점추정 → `output/ep_balanced_lags.csv`
- `python ep_delta.py`: 델타법 vs 부트스트랩 구간 → `output/ep_delta_lags.csv`, `output/paper/figA_*`

1~6은 국가 단위 부트스트랩(499회)이라 몇 분 이상 걸립니다.
`ep_report.py`에 `--point`를 붙이면 부트스트랩 없이 점추정만 빠르게 확인합니다(이때 csv의 구간 열은 점추정치와 같음).

## 확인용 수치 (Figure 1, S−I, 시작 연도 = 1)

| h | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 가계 (36건) | 1 | 0.82 | 0.56 | 0.66 | 0.81 | 0.92 | 0.70 |
| 기업 (52건) | 1 | 0.40 | 0.27 | 0.34 | 0.34 | 0.25 | 0.15 |
