# 01. 핵심쟁점

---

- 다변량·외생변수 기반 `PatchTST` 기본모델(Bench-mark)과 residual 보정모델(`NLinear`, `XGB`, `LGBM`)을 비교했을 때 성능 향상이 있는지 확인한다.
- 성능 비교는 `RMSE`, `MAE`, `MAPE`, `NRMSE` 기준으로 수행한다.
- 모델 선택은 `expanding-window ts-cv` 평균 성능을 우선 보고, 최종 `holdout`은 사후 검증으로 분리해 해석한다.

# 02. 데이터 및 모델 세팅

---

- **예측 타깃**
  - `WTI Oil` (`Com_CrudeOil`)
  - `Brent Oil` (`Com_BrentCrudeOil`)
- **예측 단위**
  - 주간 예측
- **데이터 구간 세팅**
  - 전체 기간: `2013-04-01 ~ 2026-01-12` (총 `668주`)
  - 초기 TrainSet 기간: `2013-04-01 ~ 2023-10-23` (총 `552주`)
    - 이후 `expanding window` 방식으로 fold마다 학습구간을 `4주`씩 확장
    - 마지막 ts-cv fold의 학습 종료일: `2025-07-28` (총 `644주`)
  - ValidationSet 기간: `2023-10-30 ~ 2025-10-20`
    - 총 `24개 fold`
    - 실질 평가영역: 최근 `104주`
    - Cross-Validation Fold 1: `2023-10-30 ~ 2024-01-15` (총 `12주`)
    - Cross-Validation Fold 2: `2023-11-27 ~ 2024-02-12` (총 `12주`)
    - Cross-Validation Fold 3: `2023-12-25 ~ 2024-03-11` (총 `12주`)
    - ...
    - Cross-Validation Fold 24: `2025-08-04 ~ 2025-10-20` (총 `12주`)
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)
- **피처리스트**
  - 원천 외생변수 후보 `22개`

```python
'Com_Gasoline', 'Com_NaturalGas', 'Com_Uranium', 'Com_Coal',
'Com_LME_Cu_Cash', 'Com_Steel', 'Com_Iron_Ore',
'Idx_DxyUSD', 'EX_USD_CNY', 'Bonds_US_10Y', 'Bonds_US_2Y', 'Bonds_US_3M',
'Idx_SnPVIX', 'Com_Gold', 'Idx_SnP500', 'Idx_CSI300', 'EX_USD_KRW',
'Bonds_KOR_10Y', 'EX_USD_JPY', 'Com_Corn', 'Com_Soybeans', 'Com_PalmOil'
```

  - 파생 피처 후보 `33개`
    - 각 원천변수의 수익률 피처 `22개`: `*_ret`
    - 스프레드/비율 피처 `3개`

```python
'Spread_10Y_2Y', 'Spread_Crack', 'Ratio_Gold_Oil'
```

    - 이동평균 괴리율 피처 `8개`

```python
'Com_Gasoline_ma4r', 'Com_Gasoline_ma12r',
'Com_NaturalGas_ma4r', 'Com_NaturalGas_ma12r',
'Idx_SnPVIX_ma4r', 'Idx_SnPVIX_ma12r',
'Idx_DxyUSD_ma4r', 'Idx_DxyUSD_ma12r'
```

  - 총 후보 피처는 `55개`이며, 기존 multivariate/exogenous 설정과 동일하게 모든 설명변수에는 `shift(1)`을 적용했다.
  - 각 window마다 `LightGBM SHAP ranking + TimeSeriesSplit(5)`로 최적 피처 개수를 선택했다.
    - 후보 개수: `[10, 15, 20, 25, 30, 40, 55]`
    - 평균 선택 개수: `WTI 14.8개`, `Brent 12.2개`
- **모델 세팅**
  - 기본모델 `PatchTST`

```python
patchtst_params = {
    "input_size": 24,
    "hidden_size": 64,
    "attention_heads": 4,
    "linear_hidden_size": 256,
    "patch_len": 4,
    "stride": 2,
    "dropout": 0.2,
    "encoder_layers": 2,
    "learning_rate": 0.001,
    "epochs": 60,
    "patience": 12,
    "scaler_type": "robust",
}
```

  - Residual 보정모델 `NLinear`

```python
nlinear_params = {
    "hidden_size": 64,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "epochs": 60,
    "patience": 12,
}
```

  - Residual 보정모델 `XGB`

```python
xgb_params = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
```

  - Residual 보정모델 `LGBM`

```python
lgbm_params = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
```

# 03. 실험 설계 및 적용

---

- 기존 multivariate/exogenous notebook의 feature engineering 흐름을 유지하되, 평가 프로토콜은 `expanding-window ts-cv 24개 + final holdout 12주`로 통일했다.
- 각 window마다 학습 구간만 사용해 `SHAP ranking -> 피처 개수 선택 -> RobustScaler fit` 순으로 전처리를 수행했다.
- `PatchTST`는 선택된 다변량 시퀀스(`24주 x 선택 피처`)를 입력으로 받아 `1-step seq-to-one` 방식으로 학습했고, 평가 시에는 rolling 방식으로 `12주`를 예측했다.
- residual 보정모델은 `PatchTST`의 train fitted residual을 학습 대상으로 두고 동일 입력 시퀀스에서 residual을 예측했다.
- 최종 예측식은 아래와 같다.

```text
Final Forecast = PatchTST Forecast + Residual Correction
```

- 비교군은 아래 네 종류로 구성하였다.
  - `PatchTST`
  - `PatchTST + NLinear`
  - `PatchTST + XGB`
  - `PatchTST + LGBM`
- window별 승자 빈도는 아래와 같았다.
  - `WTI Oil`: `PatchTST+XGB 14회`, `PatchTST+NLinear 4회`, `PatchTST+LGBM 4회`, `PatchTST 3회`
  - `Brent Oil`: `PatchTST+NLinear 9회`, `PatchTST+LGBM 8회`, `PatchTST+XGB 6회`, `PatchTST 2회`
- 다만 최종 모델 선택은 단순 win count가 아니라 `ts-cv 평균 성능`을 기준으로 수행했다.

# 04. 실험(모델링) 결과

### 04-01. 결과 요약 (MAPE 기준)

---

- **ts-cv 평균 MAPE 기준**

| Target | Bench-mark: PatchTST (%) | Residual-NLinear (%) | Residual-XGB (%) | Residual-LGBM (%) | ts-cv 채택 모델 | Bench-mark 대비 증감 (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| WTI Oil | 9.528 | 8.814 | 6.943 | 7.247 | PatchTST + XGB | -2.586 |
| Brent Oil | 9.467 | 8.692 | 7.120 | 7.351 | PatchTST + XGB | -2.346 |

시각화:
- `results/academic_multivariate_exog_patchtst_residuals/summary_tscv_mape_table.png`

![Summary ts-cv MAPE Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/summary_tscv_mape_table.png)

#### 핵심 Leaderboard (ts-cv 평균)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - | 7.455 | 6.794 | 9.528 | 10.441 |
| WTI Oil | PatchTST | NLinear | 6.893 | 6.318 | 8.814 | 9.626 |
| WTI Oil | PatchTST | XGB | 5.610 | 4.875 | 6.943 | 7.953 |
| WTI Oil | PatchTST | LGBM | 5.942 | 5.095 | 7.247 | 8.423 |
| Brent Oil | PatchTST | - | 7.960 | 7.288 | 9.467 | 10.381 |
| Brent Oil | PatchTST | NLinear | 7.394 | 6.691 | 8.692 | 9.658 |
| Brent Oil | PatchTST | XGB | 6.209 | 5.392 | 7.120 | 8.215 |
| Brent Oil | PatchTST | LGBM | 6.352 | 5.544 | 7.351 | 8.419 |

시각화:
- `results/academic_multivariate_exog_patchtst_residuals/tscv_leaderboard.png`

![ts-cv Leaderboard](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/tscv_leaderboard.png)

- **최종 holdout MAPE 기준**

| Target | Bench-mark: PatchTST (%) | Residual-NLinear (%) | Residual-XGB (%) | Residual-LGBM (%) | holdout 최저오차 모델 | Bench-mark 대비 증감 (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| WTI Oil | 4.440 | 4.061 | 2.432 | 3.256 | PatchTST + XGB | -2.008 |
| Brent Oil | 9.660 | 10.164 | 4.728 | 4.814 | PatchTST + XGB | -4.933 |

시각화:
- `results/academic_multivariate_exog_patchtst_residuals/summary_holdout_mape_table.png`

![Summary Holdout MAPE Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/summary_holdout_mape_table.png)

해석:
- `ts-cv 평균` 기준으로는 두 타깃 모두 `PatchTST + XGB`가 가장 안정적으로 낮은 오차를 보였다.
- `WTI Oil`은 기본 `PatchTST` 대비 `MAPE 9.528% -> 6.943%`, `holdout 4.440% -> 2.432%`로 개선됐다.
- `Brent Oil`은 기본 `PatchTST` 대비 `MAPE 9.467% -> 7.120%`, `holdout 9.660% -> 4.728%`로 개선폭이 더 컸다.
- 따라서 multivariate/exogenous 설정에서는 residual correction을 선택적 옵션이 아니라 실질적인 성능 향상 단계로 볼 근거가 생겼고, 그중 `XGB`가 가장 일관됐다.

#### 핵심 Leaderboard (holdout)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - | 2.872 | 2.600 | 4.440 | 4.886 |
| WTI Oil | PatchTST | NLinear | 2.721 | 2.375 | 4.061 | 4.630 |
| WTI Oil | PatchTST | XGB | 1.752 | 1.420 | 2.432 | 2.982 |
| WTI Oil | PatchTST | LGBM | 2.268 | 1.915 | 3.256 | 3.860 |
| Brent Oil | PatchTST | - | 6.197 | 6.031 | 9.660 | 9.884 |
| Brent Oil | PatchTST | NLinear | 6.560 | 6.344 | 10.164 | 10.463 |
| Brent Oil | PatchTST | XGB | 3.211 | 2.939 | 4.728 | 5.121 |
| Brent Oil | PatchTST | LGBM | 3.370 | 2.991 | 4.814 | 5.375 |

시각화:
- `results/academic_multivariate_exog_patchtst_residuals/holdout_leaderboard.png`

![Holdout Leaderboard](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/holdout_leaderboard.png)

정리:
- 위 `핵심 Leaderboard (ts-cv 평균)`은 모델 선택용 평균 성능표다.
- 위 `핵심 Leaderboard (holdout)`과 아래 `세부 결과`는 동일한 holdout 구간 수치이므로 서로 일치해야 한다.

### 04-02. 세부 결과

---

- **Test Set Metric**
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)

#### WTI Oil

| Model Type | Base Model | Residual Model | RMSE | MAE | MAPE (%) | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Bench-mark | PatchTST | - | 2.872 | 2.600 | 4.440 | 4.886 |
| Residual Correction | PatchTST | NLinear | 2.721 | 2.375 | 4.061 | 4.630 |
| Residual Correction | PatchTST | XGB | 1.752 | 1.420 | 2.432 | 2.982 |
| Residual Correction | PatchTST | LGBM | 2.268 | 1.915 | 3.256 | 3.860 |

시각화:
- `results/academic_multivariate_exog_patchtst_residuals/holdout_metrics_wti_table.png`

![WTI Holdout Metrics Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/holdout_metrics_wti_table.png)

#### Brent Oil

| Model Type | Base Model | Residual Model | RMSE | MAE | MAPE (%) | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Bench-mark | PatchTST | - | 6.197 | 6.031 | 9.660 | 9.884 |
| Residual Correction | PatchTST | NLinear | 6.560 | 6.344 | 10.164 | 10.463 |
| Residual Correction | PatchTST | XGB | 3.211 | 2.939 | 4.728 | 5.121 |
| Residual Correction | PatchTST | LGBM | 3.370 | 2.991 | 4.814 | 5.375 |

시각화:
- `results/academic_multivariate_exog_patchtst_residuals/holdout_metrics_brent_table.png`

![Brent Holdout Metrics Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/holdout_metrics_brent_table.png)

- **ts-cv leaderboard 표**
- 위 `핵심 Leaderboard (ts-cv 평균)`과 동일한 표다.

| Target | Base Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - | 7.455 | 6.794 | 9.528 | 10.441 |
| WTI Oil | PatchTST | NLinear | 6.893 | 6.318 | 8.814 | 9.626 |
| WTI Oil | PatchTST | XGB | 5.610 | 4.875 | 6.943 | 7.953 |
| WTI Oil | PatchTST | LGBM | 5.942 | 5.095 | 7.247 | 8.423 |
| Brent Oil | PatchTST | - | 7.960 | 7.288 | 9.467 | 10.381 |
| Brent Oil | PatchTST | NLinear | 7.394 | 6.691 | 8.692 | 9.658 |
| Brent Oil | PatchTST | XGB | 6.209 | 5.392 | 7.120 | 8.215 |
| Brent Oil | PatchTST | LGBM | 6.352 | 5.544 | 7.351 | 8.419 |

시각화:
- `results/academic_multivariate_exog_patchtst_residuals/tscv_leaderboard.png`

![ts-cv Leaderboard Repeat](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/tscv_leaderboard.png)

- **Plot**
  - holdout 실제값 vs 예측값 비교
  - `PatchTST`, `PatchTST + NLinear`, `PatchTST + XGB`, `PatchTST + LGBM`을 모두 표시

![Holdout Forecasts](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/holdout_predictions.png)

- **원본 수치 파일**
  - `results/academic_multivariate_exog_patchtst_residuals/tscv_leaderboard.csv`
  - `results/academic_multivariate_exog_patchtst_residuals/holdout_results.csv`
  - `results/academic_multivariate_exog_patchtst_residuals/forecast_predictions.csv`

# 05. 결론 및 얻게 된 인사이트

---

- multivariate/exogenous 설정에서는 단변량 실험과 달리 residual correction이 두 타깃 모두에서 의미 있는 성능 개선을 보였다.
- `PatchTST + XGB`는 `ts-cv 평균`과 `holdout` 모두에서 `WTI Oil`, `Brent Oil`의 최저 오차를 기록했다. 즉 이번 설정에서는 가장 일관된 최종 채택 후보다.
- `NLinear`는 일부 window에서 승리했지만 평균 성능과 holdout에서는 `XGB`보다 약했다. residual 구조가 비선형적이고 외생변수 상호작용이 강하다는 신호로 볼 수 있다.
- `LGBM`도 전반적으로 baseline보다 개선됐지만, 평균과 holdout 모두 `XGB`보다 한 단계 아래였다.
- 선택 빈도가 높았던 변수는 `Com_Coal`, `Com_Gasoline`, `Com_Gasoline_ma12r`, `Com_PalmOil`, `EX_USD_KRW`, `Idx_SnPVIX`, `Ratio_Gold_Oil`, `Spread_Crack`였고, Brent에서는 `Bonds_US_3M_ret`와 `Bonds_US_10Y`도 자주 선택됐다. 즉 상품가격 공행성, 위험지수, 환율, 금리 스프레드가 residual correction에도 핵심적이었다.

# 06. 향후 Action Plan

---

- `PatchTST + XGB`를 multivariate/exogenous 기본 채택 모델 후보로 두고, seed 반복 또는 bootstrap 기반 안정성 검증을 추가한다.
- 현재 `window별 feature selection`이 성능을 끌어올리는 핵심 요소이므로, 고정 feature set 대비 이득을 별도 ablation으로 검증한다.
- `WTI Oil`과 `Brent Oil` 각각에 대해 residual error distribution, feature importance, 구간별 실패 사례를 추가 분석해 모델 해석력을 보강한다.
