# 01. 핵심쟁점

---

- 본 보고서는 `실험 프로토콜과 완전히 일치하는 결과`만 포함한다.
- 비교 대상은 `PatchTST` 기본모델(Bench-mark)과 residual 보정모델(`NLinear`, `XGB`, `LGBM`)이다.
- 성능 비교는 `RMSE`, `MAE`, `MAPE`, `NRMSE` 기준으로 수행한다.
- 본문에서의 채택 여부는 `단변량 PatchTST baseline 프로토콜`을 기준으로만 판단한다.
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
  - residual 보정모델은 train 내부 calibration tail에서 생성한 `out-of-sample residual series`의 최근 `48주` residual을 입력으로 사용
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

- **지표 정의**
  - `RMSE`, `MAE`, `MAPE`는 일반적인 정의를 사용하였다.
  - `NRMSE`는 본 보고서에서 `RMSE / abs(mean(y_true)) * 100`으로 계산하였다.

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
- `PatchTST` baseline은 각 fold의 전체 train 구간으로 학습한 뒤, 다음 `12주`를 `48 -> 12` direct forecast로 예측하였다.
- residual 보정모델은 train 내부 calibration tail에서 얻은 `out-of-sample residual series`를 입력으로 받아 `48 -> 12` direct residual forecast를 수행하였다.
- calibration residual 생성에는 학습 window의 마지막 `104개` supervised windows를 사용했다.
- 이때 residual calibration용 baseline은 calibration 구간보다 앞선 train window로만 학습했고, calibration 구간에는 out-of-sample 예측만 생성했다.
- 최종 holdout/ts-cv baseline 자체는 각 fold의 전체 train 구간으로 다시 학습했다.
- 따라서 residual 보정은 `fitted residual`이 아니라 `out-of-sample residual` 기반이라는 점에서 정보 누수 없이 해석할 수 있다.
- 이 설정에서 holdout 기준 residual series 길이는 `115`, residual supervised sample 수는 `56`이다.
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
| WTI Oil | 5.379 | 15.952 | 16.656 | 17.370 | PatchTST | 0.000 |
| Brent Oil | 4.786 | 15.307 | 15.471 | 16.038 | PatchTST | 0.000 |

시각화:
- `results/academic_patchtst_residuals/summary_tscv_mape_table.png`

![Summary ts-cv MAPE Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/summary_tscv_mape_table.png)

#### 핵심 Leaderboard (ts-cv 평균)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - | 4.435 | 3.815 | 5.379 | 6.230 |
| WTI Oil | PatchTST | NLinear | 12.305 | 11.514 | 15.952 | 16.899 |
| WTI Oil | PatchTST | XGB | 12.649 | 12.001 | 16.656 | 17.426 |
| WTI Oil | PatchTST | LGBM | 13.112 | 12.534 | 17.370 | 18.074 |
| Brent Oil | PatchTST | - | 4.179 | 3.548 | 4.786 | 5.596 |
| Brent Oil | PatchTST | NLinear | 12.423 | 11.689 | 15.307 | 16.153 |
| Brent Oil | PatchTST | XGB | 12.401 | 11.809 | 15.471 | 16.141 |
| Brent Oil | PatchTST | LGBM | 12.893 | 12.240 | 16.038 | 16.789 |

시각화:
- `results/academic_patchtst_residuals/tscv_leaderboard.png`

![ts-cv Leaderboard](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/tscv_leaderboard.png)

- **최종 holdout MAPE 기준**

| Target | Bench-mark: PatchTST (%) | Residual-NLinear (%) | Residual-XGB (%) | Residual-LGBM (%) | holdout 최저오차 모델 | Bench-mark 대비 증감 (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| WTI Oil | 2.475 | 11.902 | 7.860 | 6.949 | PatchTST | 0.000 |
| Brent Oil | 2.515 | 11.488 | 7.789 | 7.346 | PatchTST | 0.000 |

시각화:
- `results/academic_patchtst_residuals/summary_holdout_mape_table.png`

![Summary Holdout MAPE Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/summary_holdout_mape_table.png)

해석:
- strict residual calibration 구조로 다시 계산한 결과, `ts-cv 평균`과 `holdout` 모두에서 두 타깃 모두 `PatchTST` 기본모델이 가장 우세했다.
- 즉 기존의 residual 개선은 `train fitted residual` 구조에 민감했던 것으로 보이며, `out-of-sample residual` 기준으로 다시 계산하면 재현되지 않았다.
- 따라서 현재 검증 완료 구조에서는 residual correction을 채택할 근거가 없다.

#### 핵심 Leaderboard (holdout)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - | 1.724 | 1.466 | 2.475 | 2.933 |
| WTI Oil | PatchTST | NLinear | 7.338 | 7.022 | 11.902 | 12.486 |
| WTI Oil | PatchTST | XGB | 5.481 | 4.651 | 7.860 | 9.325 |
| WTI Oil | PatchTST | LGBM | 5.136 | 4.124 | 6.949 | 8.739 |
| Brent Oil | PatchTST | - | 1.723 | 1.581 | 2.515 | 2.749 |
| Brent Oil | PatchTST | NLinear | 7.433 | 7.222 | 11.488 | 11.857 |
| Brent Oil | PatchTST | XGB | 6.302 | 4.945 | 7.789 | 10.053 |
| Brent Oil | PatchTST | LGBM | 6.124 | 4.673 | 7.346 | 9.769 |

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
| Bench-mark | PatchTST | - | 1.724 | 1.466 | 2.475 | 2.933 |
| Residual Correction | PatchTST | NLinear | 7.338 | 7.022 | 11.902 | 12.486 |
| Residual Correction | PatchTST | XGB | 5.481 | 4.651 | 7.860 | 9.325 |
| Residual Correction | PatchTST | LGBM | 5.136 | 4.124 | 6.949 | 8.739 |

시각화:
- `results/academic_patchtst_residuals/holdout_metrics_wti_table.png`

![WTI Holdout Metrics Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/holdout_metrics_wti_table.png)

#### Brent Oil

| Model Type | Base Model | Residual Model | RMSE | MAE | MAPE (%) | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Bench-mark | PatchTST | - | 1.723 | 1.581 | 2.515 | 2.749 |
| Residual Correction | PatchTST | NLinear | 7.433 | 7.222 | 11.488 | 11.857 |
| Residual Correction | PatchTST | XGB | 6.302 | 4.945 | 7.789 | 10.053 |
| Residual Correction | PatchTST | LGBM | 6.124 | 4.673 | 7.346 | 9.769 |

시각화:
- `results/academic_patchtst_residuals/holdout_metrics_brent_table.png`

![Brent Holdout Metrics Table](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/holdout_metrics_brent_table.png)

- **ts-cv leaderboard 표**

| Target | Base Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - | 4.435 | 3.815 | 5.379 | 6.230 |
| WTI Oil | PatchTST | NLinear | 12.305 | 11.514 | 15.952 | 16.899 |
| WTI Oil | PatchTST | XGB | 12.649 | 12.001 | 16.656 | 17.426 |
| WTI Oil | PatchTST | LGBM | 13.112 | 12.534 | 17.370 | 18.074 |
| Brent Oil | PatchTST | - | 4.179 | 3.548 | 4.786 | 5.596 |
| Brent Oil | PatchTST | NLinear | 12.423 | 11.689 | 15.307 | 16.153 |
| Brent Oil | PatchTST | XGB | 12.401 | 11.809 | 15.471 | 16.141 |
| Brent Oil | PatchTST | LGBM | 12.893 | 12.240 | 16.038 | 16.789 |

시각화:
- `results/academic_patchtst_residuals/tscv_leaderboard.png`

![ts-cv Leaderboard Repeat](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/tscv_leaderboard.png)

- **Plot**

![Holdout Forecasts](/Users/jaeholee/Desktop/oil_forecast/results/academic_patchtst_residuals/holdout_predictions.png)

# 05. 결론 및 얻게 된 인사이트

---

- 현재 strict protocol 기준에서는 `WTI Oil`, `Brent Oil` 모두 residual 보정모델이 `ts-cv 평균`과 `holdout`에서 모두 `PatchTST` 기본모델을 이기지 못했다.
- 즉 residual correction은 `out-of-sample residual` 기반 strict 구조에서는 일관된 추가 성능 향상을 만들지 못했다.
- residual calibration 표본이 holdout 기준 `56개 supervised samples`로 매우 작다는 점도, residual 보정모델의 불안정성을 키운 원인으로 해석된다.
- 따라서 현재 검증 완료 범위에서는 `PatchTST` 단독 모델이 가장 논리적으로 안전한 채택안이다.

# 06. 향후 Action Plan

---

- strict 기준본은 `PatchTST` 단독 모델로 고정하고, residual correction은 별도 연구 가설로만 유지한다.
- residual correction을 다시 시험하려면 반드시 `out-of-sample residual generation` 구조를 유지한 상태에서 반복 검증한다.
- 다변량·외생변수 실험은 strict `48 -> 12 direct` ts-cv 프로토콜에 맞춰 별도로 재구성한 뒤 후속 보고서로 분리한다.
