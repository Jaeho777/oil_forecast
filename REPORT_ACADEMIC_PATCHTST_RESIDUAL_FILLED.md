# 01. 핵심쟁점

---

- `PatchTST` 기본모델(Bench-mark)과 residual 보정모델(`NLinear`, `XGB`, `LGBM`)을 비교했을 때 성능 향상이 있는지 확인한다.
- 성능 비교는 `RMSE`, `MAE`, `MAPE`, `NRMSE` 기준으로 수행한다.
- 모델 선택 관점에서는 `expanding-window ts-cv` 평균 성능을 우선 보고, 최종 `holdout`은 사후 검증으로 분리해 해석한다.

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
  - 기본모델
    - 외생변수 없음
    - 각 타깃별 단변량 시계열의 최근 `48주` 값만 입력으로 사용
  - Residual 보정모델
    - `PatchTST`의 train fitted residual series에서 최근 `48주` residual을 입력으로 사용
- **모델 세팅**
  - 기본모델 `PatchTST`

```python
patchtst_params = {
    "input_size": 48,
    "horizon": 12,
    "hidden_size": 128,
    "attention_heads": 16,
    "linear_hidden_size": 256,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.2,
    "encoder_layers": 3,
    "attn_dropout": 0.0,
    "fc_dropout": 0.2,
    "max_steps": 5000,
    "learning_rate": 0.0001,
    "scaler_type": "identity",
}
```

  - Residual 보정모델 `NLinear`

```python
nlinear_params = {
    "input_size": 48,
    "horizon": 12,
    "learning_rate": 0.001,
    "max_steps": 2000,
    "batch_size": 32,
}
```

  - Residual 보정모델 `XGB`

```python
xgb_params = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
```

  - Residual 보정모델 `LGBM`

```python
lgbm_params = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
```

# 03. 실험 설계 및 적용

---

- 각 타깃(`WTI Oil`, `Brent Oil`)은 별도의 단변량 시계열로 분리하여 학습하였다.
- `PatchTST`로 먼저 `12-step direct forecast`를 수행한 뒤, train 구간의 overlapping fitted forecast를 평균하여 residual series를 구성하였다.
- residual 보정모델은 해당 residual series를 사용해 `48 -> 12` direct residual forecast를 수행하였다.
- 최종 예측식은 아래와 같다.

```text
Final Forecast = PatchTST Forecast + Residual Correction
```

- 비교군은 아래 네 종류로 구성하였다.
  - `PatchTST`
  - `PatchTST + NLinear`
  - `PatchTST + XGB`
  - `PatchTST + LGBM`
- 모델 선택 해석은 `ts-cv 평균 성능`을 기준으로 하고, `holdout`은 최종 사후 점검으로 분리하였다.

# 04. 실험(모델링) 결과

### 04-01. 결과 요약 (MAPE 기준)

---

- **ts-cv 평균 MAPE 기준**

| Target | Bench-mark: PatchTST (%) | Residual-NLinear (%) | Residual-XGB (%) | Residual-LGBM (%) | ts-cv 채택 모델 | Bench-mark 대비 증감 (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| WTI Oil | 5.502 | 6.897 | 7.995 | 8.188 | PatchTST | 0.000 |
| Brent Oil | 5.423 | 6.413 | 6.470 | 6.315 | PatchTST | 0.000 |

시각화:
- `results/academic_patchtst_residuals/summary_tscv_mape_table.png`

- **최종 holdout MAPE 기준**

| Target | Bench-mark: PatchTST (%) | Residual-NLinear (%) | Residual-XGB (%) | Residual-LGBM (%) | holdout 최저오차 모델 | Bench-mark 대비 증감 (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| WTI Oil | 9.834 | 3.881 | 13.632 | 14.221 | PatchTST + NLinear | -5.953 |
| Brent Oil | 2.701 | 4.212 | 4.602 | 5.554 | PatchTST | 0.000 |

시각화:
- `results/academic_patchtst_residuals/summary_holdout_mape_table.png`

해석:
- `ts-cv 평균` 기준으로는 두 타깃 모두 residual 보정 없이 `PatchTST`가 가장 안정적이었다.
- 다만 `WTI Oil`의 최종 holdout에서는 `PatchTST + NLinear`가 `MAPE`를 `9.834% -> 3.881%`로 크게 낮췄다.
- `Brent Oil`은 `ts-cv`와 `holdout` 모두에서 residual 보정보다 `PatchTST` 기본모델이 우세했다.

### 04-02. 세부 결과

---

- **Test Set Metric**
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)

#### WTI Oil

| Model Type | Base Model | Residual Model | RMSE | MAE | MAPE (%) | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Bench-mark | PatchTST | - | 5.887 | 5.751 | 9.834 | 10.016 |
| Residual Correction | PatchTST | NLinear | 2.659 | 2.269 | 3.881 | 4.525 |
| Residual Correction | PatchTST | XGB | 9.112 | 7.976 | 13.632 | 15.504 |
| Residual Correction | PatchTST | LGBM | 9.265 | 8.296 | 14.221 | 15.764 |

시각화:
- `results/academic_patchtst_residuals/holdout_metrics_wti_table.png`

#### Brent Oil

| Model Type | Base Model | Residual Model | RMSE | MAE | MAPE (%) | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Bench-mark | PatchTST | - | 2.059 | 1.709 | 2.701 | 3.284 |
| Residual Correction | PatchTST | NLinear | 3.408 | 2.675 | 4.212 | 5.436 |
| Residual Correction | PatchTST | XGB | 3.383 | 2.887 | 4.602 | 5.395 |
| Residual Correction | PatchTST | LGBM | 3.940 | 3.474 | 5.554 | 6.285 |

시각화:
- `results/academic_patchtst_residuals/holdout_metrics_brent_table.png`

- **ts-cv leaderboard 표**

| Target | Base Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - | 4.566 | 3.914 | 5.502 | 6.392 |
| WTI Oil | PatchTST | NLinear | 5.586 | 4.897 | 6.897 | 7.843 |
| WTI Oil | PatchTST | XGB | 6.566 | 5.787 | 7.995 | 9.058 |
| WTI Oil | PatchTST | LGBM | 6.768 | 5.885 | 8.188 | 9.388 |
| Brent Oil | PatchTST | - | 4.643 | 4.026 | 5.423 | 6.225 |
| Brent Oil | PatchTST | NLinear | 5.421 | 4.755 | 6.413 | 7.283 |
| Brent Oil | PatchTST | XGB | 5.536 | 4.816 | 6.470 | 7.411 |
| Brent Oil | PatchTST | LGBM | 5.499 | 4.699 | 6.315 | 7.364 |

시각화:
- `results/academic_patchtst_residuals/tscv_leaderboard.png`

- **Plot**
  - ts-cv leaderboard 표 이미지: `results/academic_patchtst_residuals/tscv_leaderboard.png`
  - holdout 예측 곡선 이미지: `results/academic_patchtst_residuals/holdout_predictions.png`
  - 원본 수치 파일:
    - `results/academic_patchtst_residuals/tscv_leaderboard.csv`
    - `results/academic_patchtst_residuals/holdout_results.csv`
    - `results/academic_patchtst_residuals/forecast_predictions.csv`

# 05. 결론 및 얻게 된 인사이트

---

- `ts-cv 평균` 기준으로는 `WTI Oil`과 `Brent Oil` 모두 residual 보정모델이 `PatchTST` 기본모델을 안정적으로 이기지 못했다.
- `WTI Oil`에서는 사후 holdout 기준으로 `PatchTST + NLinear`가 가장 낮은 오차를 기록했다. 이는 특정 구간에서 residual 구조가 강하게 남아 있었음을 시사한다.
- `Brent Oil`은 `ts-cv`와 `holdout` 모두에서 기본 `PatchTST`가 가장 강했다. 즉 Brent에는 residual 보정을 기본 옵션으로 넣을 근거가 약하다.
- 따라서 현재 설정에서는 residual 보정모델을 “항상 추가하는 기본 단계”로 보기보다, 타깃별 또는 구간별 선택적 보정 단계로 해석하는 것이 타당하다.

# 06. 향후 Action Plan

---

- `WTI Oil`에 대해서는 `NLinear residual correction`의 holdout 개선이 반복적으로 재현되는지 추가 rolling-origin 실험으로 확인한다.
- `Brent Oil`에 대해서는 residual 보정을 제거한 `PatchTST` 단독 모델을 기본 배포 후보로 유지한다.
- residual 보정모델 채택 여부를 `ts-cv gate`로 자동 결정하는 선택 규칙을 추가해, holdout 사후 선택 편향을 줄인다.
- 보고서 최종본에는 `ts-cv leaderboard`와 `holdout prediction plot`을 함께 넣어, 평균 성능과 최종 구간 성능의 차이를 동시에 보여준다.
