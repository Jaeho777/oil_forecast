# Exploratory Note

> 이 문서는 updated exogenous를 허용한 exploratory 결과입니다.  
> strict `48 -> 12 direct` benchmark와 동일한 leaderboard로 해석하지 않습니다.  
> strict protocol 제출/공유용 기준본은 [REPORT_ACADEMIC_VALIDATED_CLEAN.md](/Users/jaeholee/Desktop/oil_forecast/REPORT_ACADEMIC_VALIDATED_CLEAN.md)를 사용합니다.

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
    - 평균 선택 개수: `WTI 27.9개`, `Brent 27.9개`
- **모델 세팅**
  - 기본모델 `PatchTST`

```python
patchtst_params = {
    "input_size": 48,
    "hidden_size": 128,
    "attention_heads": 16,
    "linear_hidden_size": 256,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.2,
    "encoder_layers": 3,
    "attn_dropout": 0.0,
    "fc_dropout": 0.2,
    "learning_rate": 0.0001,
    "max_steps": 5000,
    "scaler_type": "identity",
}
```

- **지표 정의**
  - `RMSE`, `MAE`, `MAPE`는 일반적인 정의를 사용하였다.
  - `NRMSE`는 본 exploratory 보고서에서 `RMSE / abs(mean(y_true)) * 100`으로 계산하였다.

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
- 각 window마다 학습 구간만 사용해 `SHAP ranking -> 피처 개수 선택` 순으로 전처리를 수행했고, PatchTST 입력 스케일링은 `identity`로 두었다.
- `PatchTST`는 선택된 다변량 시퀀스(`48주 x 선택 피처`)를 입력으로 받아 `1-step seq-to-one` 방식으로 학습했고, 평가 시에는 rolling 방식으로 `12주`를 예측했다.
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
  - `WTI Oil`: `PatchTST+XGB 11회`, `PatchTST+NLinear 11회`, `PatchTST+LGBM 2회`, `PatchTST 1회`
  - `Brent Oil`: `PatchTST+XGB 12회`, `PatchTST+NLinear 6회`, `PatchTST+LGBM 4회`, `PatchTST 3회`
- 다만 최종 모델 선택은 단순 win count가 아니라 `ts-cv 평균 성능`을 기준으로 수행했다.

# 04. 실험(모델링) 결과

### 04-01. 결과 요약 (MAPE 기준)

---

- **ts-cv 평균 MAPE 기준**

| Target | Bench-mark: PatchTST (%) | Residual-NLinear (%) | Residual-XGB (%) | Residual-LGBM (%) | ts-cv 채택 모델 | Bench-mark 대비 증감 (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| WTI Oil | 8.202 | 5.709 | 5.160 | 5.677 | PatchTST + XGB | -3.042 |
| Brent Oil | 9.433 | 7.236 | 5.062 | 5.874 | PatchTST + XGB | -4.372 |

시각화:
- `results/academic_multivariate_exog_patchtst_residuals/summary_tscv_mape_table.png`

![Summary ts-cv MAPE Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/summary_tscv_mape_table.png)

#### 핵심 Leaderboard (ts-cv 평균)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - | 6.353 | 5.777 | 8.202 | 8.935 |
| WTI Oil | PatchTST | NLinear | 4.719 | 4.116 | 5.709 | 6.581 |
| WTI Oil | PatchTST | XGB | 4.355 | 3.686 | 5.160 | 6.105 |
| WTI Oil | PatchTST | LGBM | 4.864 | 4.059 | 5.677 | 6.804 |
| Brent Oil | PatchTST | - | 7.469 | 6.946 | 9.433 | 10.043 |
| Brent Oil | PatchTST | NLinear | 5.887 | 5.354 | 7.236 | 7.892 |
| Brent Oil | PatchTST | XGB | 4.298 | 3.724 | 5.062 | 5.774 |
| Brent Oil | PatchTST | LGBM | 4.916 | 4.304 | 5.874 | 6.618 |

시각화:
- `results/academic_multivariate_exog_patchtst_residuals/tscv_leaderboard.png`

![ts-cv Leaderboard](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/tscv_leaderboard.png)

- **최종 holdout MAPE 기준**

| Target | Bench-mark: PatchTST (%) | Residual-NLinear (%) | Residual-XGB (%) | Residual-LGBM (%) | holdout 최저오차 모델 | Bench-mark 대비 증감 (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| WTI Oil | 14.924 | 9.060 | 2.764 | 3.843 | PatchTST + XGB | -12.160 |
| Brent Oil | 11.336 | 5.940 | 2.055 | 2.731 | PatchTST + XGB | -9.281 |

시각화:
- `results/academic_multivariate_exog_patchtst_residuals/summary_holdout_mape_table.png`

![Summary Holdout MAPE Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/summary_holdout_mape_table.png)

해석:
- `ts-cv 평균` 기준으로는 두 타깃 모두 `PatchTST + XGB`가 가장 안정적으로 낮은 오차를 보였다.
- `WTI Oil`은 기본 `PatchTST` 대비 `MAPE 8.202% -> 5.160%`, `holdout 14.924% -> 2.764%`로 개선됐다.
- `Brent Oil`은 기본 `PatchTST` 대비 `MAPE 9.433% -> 5.062%`, `holdout 11.336% -> 2.055%`로 개선폭이 더 컸다.
- 따라서 multivariate/exogenous 설정에서는 residual correction을 선택적 옵션이 아니라 실질적인 성능 향상 단계로 볼 근거가 생겼고, 그중 `XGB`가 가장 일관됐다.

#### 핵심 Leaderboard (holdout)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - | 8.828 | 8.741 | 14.924 | 15.021 |
| WTI Oil | PatchTST | NLinear | 5.432 | 5.309 | 9.060 | 9.242 |
| WTI Oil | PatchTST | XGB | 1.903 | 1.616 | 2.764 | 3.237 |
| WTI Oil | PatchTST | LGBM | 2.434 | 2.250 | 3.843 | 4.142 |
| Brent Oil | PatchTST | - | 7.199 | 7.075 | 11.336 | 11.483 |
| Brent Oil | PatchTST | NLinear | 3.954 | 3.721 | 5.940 | 6.307 |
| Brent Oil | PatchTST | XGB | 1.626 | 1.289 | 2.055 | 2.594 |
| Brent Oil | PatchTST | LGBM | 2.053 | 1.716 | 2.731 | 3.274 |

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
| Bench-mark | PatchTST | - | 8.828 | 8.741 | 14.924 | 15.021 |
| Residual Correction | PatchTST | NLinear | 5.432 | 5.309 | 9.060 | 9.242 |
| Residual Correction | PatchTST | XGB | 1.903 | 1.616 | 2.764 | 3.237 |
| Residual Correction | PatchTST | LGBM | 2.434 | 2.250 | 3.843 | 4.142 |

시각화:
- `results/academic_multivariate_exog_patchtst_residuals/holdout_metrics_wti_table.png`

![WTI Holdout Metrics Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/holdout_metrics_wti_table.png)

#### Brent Oil

| Model Type | Base Model | Residual Model | RMSE | MAE | MAPE (%) | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Bench-mark | PatchTST | - | 7.199 | 7.075 | 11.336 | 11.483 |
| Residual Correction | PatchTST | NLinear | 3.954 | 3.721 | 5.940 | 6.307 |
| Residual Correction | PatchTST | XGB | 1.626 | 1.289 | 2.055 | 2.594 |
| Residual Correction | PatchTST | LGBM | 2.053 | 1.716 | 2.731 | 3.274 |

시각화:
- `results/academic_multivariate_exog_patchtst_residuals/holdout_metrics_brent_table.png`

![Brent Holdout Metrics Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_multivariate_exog_patchtst_residuals/holdout_metrics_brent_table.png)

- **ts-cv leaderboard 표**
- 위 `핵심 Leaderboard (ts-cv 평균)`과 동일한 표다.

| Target | Base Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - | 6.353 | 5.777 | 8.202 | 8.935 |
| WTI Oil | PatchTST | NLinear | 4.719 | 4.116 | 5.709 | 6.581 |
| WTI Oil | PatchTST | XGB | 4.355 | 3.686 | 5.160 | 6.105 |
| WTI Oil | PatchTST | LGBM | 4.864 | 4.059 | 5.677 | 6.804 |
| Brent Oil | PatchTST | - | 7.469 | 6.946 | 9.433 | 10.043 |
| Brent Oil | PatchTST | NLinear | 5.887 | 5.354 | 7.236 | 7.892 |
| Brent Oil | PatchTST | XGB | 4.298 | 3.724 | 5.062 | 5.774 |
| Brent Oil | PatchTST | LGBM | 4.916 | 4.304 | 5.874 | 6.618 |

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

- multivariate/exogenous exploratory 설정에서는 단변량 strict 결과와 달리 residual correction이 두 타깃 모두에서 큰 성능 개선을 보였다.
- `PatchTST + XGB`는 `ts-cv 평균`과 `holdout` 모두에서 `WTI Oil`, `Brent Oil`의 최저 오차를 기록했다. 즉 이번 exploratory 설정에서는 가장 일관된 최종 후보였다.
- `NLinear`는 일부 window에서 승리했지만 평균 성능과 holdout에서는 `XGB`보다 약했다. residual 구조가 비선형적이고 외생변수 상호작용이 강하다는 신호로 볼 수 있다.
- `LGBM`도 전반적으로 baseline보다 개선됐지만, 평균과 holdout 모두 `XGB`보다 한 단계 아래였다.
- 선택 빈도가 높았던 변수는 `Com_Coal`, `Com_Gasoline`, `Com_Gasoline_ma12r`, `Com_PalmOil`, `EX_USD_KRW`, `Idx_SnPVIX`, `Ratio_Gold_Oil`, `Spread_Crack`였고, Brent에서는 `Bonds_US_3M_ret`와 `Bonds_US_10Y`도 자주 선택됐다. 즉 상품가격 공행성, 위험지수, 환율, 금리 스프레드가 residual correction에도 핵심적이었다.

# 06. 향후 Action Plan

---

- `PatchTST + XGB`를 multivariate/exogenous 기본 채택 모델 후보로 두고, seed 반복 또는 bootstrap 기반 안정성 검증을 추가한다.
- 현재 `window별 feature selection`이 성능을 끌어올리는 핵심 요소이므로, 고정 feature set 대비 이득을 별도 ablation으로 검증한다.
- `WTI Oil`과 `Brent Oil` 각각에 대해 residual error distribution, feature importance, 구간별 실패 사례를 추가 분석해 모델 해석력을 보강한다.
