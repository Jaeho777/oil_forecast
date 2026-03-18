# 01. 핵심쟁점

---

- 본 보고서는 `실험 프로토콜과 완전히 일치하는 결과`만 포함한다.
- 비교 대상은 `PatchTST` 기본모델(Bench-mark)과 residual 보정모델(`NLinear`, `XGB`, `LGBM`)이다.
- 성능 비교는 `RMSE`, `MAE`, `MAPE`, `NRMSE` 기준으로 수행한다.
- 모델 선택은 `expanding-window ts-cv` 평균 성능을 우선 보고, `holdout`은 최종 사후 검증으로 분리해 해석한다.

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
  - 최근 평가영역 길이: `116주 = 104주(ts-cv) + 12주(holdout) = 52*2 + 12`
- **피처리스트**
  - 외생변수 없음
  - 각 타깃은 별도의 단변량 시계열로 구성되며, 최근 `48주` 값만 입력으로 사용
  - residual 보정모델은 `PatchTST`의 train fitted residual series에서 최근 `48주` residual을 입력으로 사용
- **공통 설정**

```python
common_config = {
    "horizon": 12,
    "step_size": 4,
    "number_of_windows": 24,
    "final_holdout": 12,
    "input_size": 48,
    "season_length": 52,
}
```

- **모델 세팅**
  - 기본모델 `PatchTST`

```python
patchtst_params = {
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

  - Residual 보정모델 `NLinear`, `XGB`, `LGBM`

# 03. 실험 설계 및 적용

---

- 각 타깃(`WTI Oil`, `Brent Oil`)은 별도의 단변량 시계열로 분리하여 학습하였다.
- `PatchTST`는 `48 -> 12`의 direct forecast를 수행하였다.
- residual 보정모델은 `PatchTST`의 train 구간 fitted residual series를 입력으로 받아 `48 -> 12` direct residual forecast를 수행하였다.
- 따라서 residual 보정 결과는 `out-of-fold residual benchmark`가 아니라, 동일 학습 window 내 2단계 보정 성능으로 해석하였다.
- 최종 예측식은 아래와 같다.

```text
Final Forecast = PatchTST Forecast + Residual Correction
```

- 비교군은 아래 네 종류로 구성하였다.
  - `PatchTST`
  - `PatchTST + NLinear`
  - `PatchTST + XGB`
  - `PatchTST + LGBM`
- 본 보고서에는 위 프로토콜과 완전히 일치하는 단변량 실험만 포함하였다.
- 다변량·외생변수 실험은 별도 exploratory 결과로 보관하되, 현재 본문 채택 결과에는 포함하지 않았다.

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

![Summary ts-cv MAPE Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/summary_tscv_mape_table.png)

#### 핵심 Leaderboard (ts-cv 평균)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
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

![ts-cv Leaderboard](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/tscv_leaderboard.png)

- **최종 holdout MAPE 기준**

| Target | Bench-mark: PatchTST (%) | Residual-NLinear (%) | Residual-XGB (%) | Residual-LGBM (%) | holdout 최저오차 모델 | Bench-mark 대비 증감 (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| WTI Oil | 9.834 | 3.881 | 13.632 | 14.221 | PatchTST + NLinear | -5.953 |
| Brent Oil | 2.701 | 4.212 | 4.602 | 5.554 | PatchTST | 0.000 |

시각화:
- `results/academic_patchtst_residuals/summary_holdout_mape_table.png`

![Summary Holdout MAPE Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/summary_holdout_mape_table.png)

해석:
- `ts-cv 평균` 기준으로는 두 타깃 모두 residual 보정 없이 `PatchTST`가 가장 안정적이었다.
- `WTI Oil`의 단일 holdout 구간에서는 `PatchTST + NLinear`가 `MAPE 9.834% -> 3.881%`로 크게 개선되었다.
- `Brent Oil`은 `ts-cv`와 `holdout` 모두에서 `PatchTST` 기본모델이 가장 우세했다.

#### 핵심 Leaderboard (holdout)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - | 5.887 | 5.751 | 9.834 | 10.016 |
| WTI Oil | PatchTST | NLinear | 2.659 | 2.269 | 3.881 | 4.525 |
| WTI Oil | PatchTST | XGB | 9.112 | 7.976 | 13.632 | 15.504 |
| WTI Oil | PatchTST | LGBM | 9.265 | 8.296 | 14.221 | 15.764 |
| Brent Oil | PatchTST | - | 2.059 | 1.709 | 2.701 | 3.284 |
| Brent Oil | PatchTST | NLinear | 3.408 | 2.675 | 4.212 | 5.436 |
| Brent Oil | PatchTST | XGB | 3.383 | 2.887 | 4.602 | 5.395 |
| Brent Oil | PatchTST | LGBM | 3.940 | 3.474 | 5.554 | 6.285 |

시각화:
- `results/academic_patchtst_residuals/holdout_leaderboard.png`

![Holdout Leaderboard](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/holdout_leaderboard.png)

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

![WTI Holdout Metrics Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/holdout_metrics_wti_table.png)

#### Brent Oil

| Model Type | Base Model | Residual Model | RMSE | MAE | MAPE (%) | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Bench-mark | PatchTST | - | 2.059 | 1.709 | 2.701 | 3.284 |
| Residual Correction | PatchTST | NLinear | 3.408 | 2.675 | 4.212 | 5.436 |
| Residual Correction | PatchTST | XGB | 3.383 | 2.887 | 4.602 | 5.395 |
| Residual Correction | PatchTST | LGBM | 3.940 | 3.474 | 5.554 | 6.285 |

시각화:
- `results/academic_patchtst_residuals/holdout_metrics_brent_table.png`

![Brent Holdout Metrics Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/holdout_metrics_brent_table.png)

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

![ts-cv Leaderboard Repeat](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/tscv_leaderboard.png)

- **Plot**

![Holdout Forecasts](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/holdout_predictions.png)

# 05. 결론 및 얻게 된 인사이트

---

- 현재 strict protocol 기준에서는 `WTI Oil`, `Brent Oil` 모두 residual 보정모델이 `ts-cv 평균`에서 `PatchTST` 기본모델을 안정적으로 이기지 못했다.
- `WTI Oil`의 단일 holdout에서는 `PatchTST + NLinear`가 크게 개선되었으나, 이는 최종 사후 구간의 특이성일 가능성이 있어 채택 근거는 `ts-cv 평균`보다 약하다.
- `Brent Oil`은 `ts-cv`와 `holdout` 모두에서 기본 `PatchTST`가 가장 강했다.
- 따라서 현재 검증 완료 범위에서는 residual 보정모델을 기본 단계로 채택하기보다, 타깃별·구간별 선택적 보정 단계로 해석하는 것이 가장 논리적으로 안전하다.

# 06. 향후 Action Plan

---

- `WTI Oil`에 대해서는 `NLinear residual correction`의 holdout 개선이 반복적으로 재현되는지 추가 rolling-origin 검증을 수행한다.
- `Brent Oil`에 대해서는 `PatchTST` 단독 모델을 기본 비교 기준으로 유지한다.
- 다변량·외생변수 실험은 strict `48 -> 12 direct` ts-cv 프로토콜에 맞춰 별도로 재구성한 뒤 후속 보고서로 분리한다.
