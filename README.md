# oil_forecast

이 레포는 유가 예측 실험 중 현재 핵심인 세 가지 축만 정리한 클린 버전이다.

## 새 단변량 학계식 프로토콜
- `scripts/academic_patchtst_residuals.py`
- `shift(1)`과 외생변수를 쓰지 않고 `WTI Oil`, `Brent Oil`을 각각 별도 단변량 시계열로 평가한다.
- 평가 프로토콜은 `expanding-window ts-cv 24개 + final holdout 12주`이다.
- 공통 설정:
  - 전체 구간 `2013-04-01 ~ 2026-01-12` (`668` weekly observations)
  - `horizon=12`
  - `step size=4`
  - `number of windows=24`
  - `final holdout=12`
  - `input size=48`
  - `season length=52`
- 비교 모델:
  - `PatchTST`
  - `PatchTST + NLinear`
  - `PatchTST + XGB`
  - `PatchTST + LGBM`
- 보고 지표:
  - `RMSE`
  - `MAE`
  - `MAPE`
  - `NRMSE`

실행 예시:

```bash
python scripts/academic_patchtst_residuals.py
```

결과 저장 위치:
- `results/academic_patchtst_residuals/window_results.csv`
- `results/academic_patchtst_residuals/tscv_summary.csv`
- `results/academic_patchtst_residuals/tscv_leaderboard.csv`
- `results/academic_patchtst_residuals/holdout_results.csv`
- `results/academic_patchtst_residuals/holdout_leaderboard.csv`
- `results/academic_patchtst_residuals/forecast_predictions.csv`
- `results/academic_patchtst_residuals/config.json`
- `results/academic_patchtst_residuals/summary_tscv_mape_table.png`
- `results/academic_patchtst_residuals/summary_holdout_mape_table.png`
- `results/academic_patchtst_residuals/holdout_metrics_wti_table.png`
- `results/academic_patchtst_residuals/holdout_metrics_brent_table.png`
- `results/academic_patchtst_residuals/tscv_leaderboard.png`
- `results/academic_patchtst_residuals/holdout_leaderboard.png`

보고서 템플릿:
- `REPORT_TEMPLATE_WITH_RESIDUAL.md`
- residual 보정모델 열이 포함된 본문 표/leaderboard 표 양식을 제공한다.

## 포함 내용
- 구조 실험: `ExponentialSmoothing + NLinear + LightGBM`
- Transformer 실험: `PatchTST + iTransformer`
- 교수님 추가 비교: `PatchTST + NLinear`
- baseline 교체 실험: `PatchTST + NLinear + LightGBM`

## 시작 순서
1. `TEAM_REPRO_GUIDE.md`를 읽는다.
2. `RESULTS_SUMMARY.csv`로 핵심 숫자를 확인한다.
3. `notebooks/`와 `results/`를 함께 본다.

## 노트북
- `notebooks/01_structure_es_nlinear_lgbm.ipynb`
- `notebooks/02_transformer_patchtst_family.ipynb`

## 핵심 결과
- `ExponentialSmoothing + NLinear + LightGBM`: test RMSE `1.2275`
- `PatchTST + iTransformer`: test RMSE `1.2308`
- `PatchTST + NLinear`: test RMSE `1.9251`
- `PatchTST + NLinear + LightGBM`: test RMSE `1.8609`

## 주의
- 모든 실험은 `Com_BrentCrudeOil`를 타깃으로 한다.
- 모든 설명변수는 `shift(1)`을 적용한다.
- 모델 선택은 validation 기준이고 test는 최종 확인용이다.
- `PatchTST + NLinear`와 `PatchTST + NLinear + LightGBM`은 이름이 비슷하지만 다른 실험이다.
  - 전자는 transformer framework 안의 same-condition 비교다.
  - 후자는 stagewise 구조에서 baseline만 PatchTST로 교체한 실험이다.

위 주의사항은 아래 노트북/기존 결과물에 대한 설명이다.
- `notebooks/`
- `results/patchtst_baseline_swap/`
- `results/transformer/`
- `results/structure/`
