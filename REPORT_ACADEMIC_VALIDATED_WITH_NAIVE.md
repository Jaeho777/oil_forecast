# 01. 핵심쟁점

---

- “기본모델(Bench-mark)”인 `Naive`와 “실험모델”인 `PatchTST`를 비교했을 때, 예측 성능 향상이 있는지 확인한다.
- `PatchTST`에 residual 보정모델(`NLinear`, `XGB`, `LGBM`)을 추가했을 때, `Naive` 및 `PatchTST` 대비 추가 성능 개선이 있는지 확인한다.
- 성능 비교는 `RMSE`, `MAE`, `MAPE`, `NRMSE` 기준으로 수행한다.

# 02. 데이터 및 모델 세팅

---

- **예측 타깃**
  - `WTI Oil` (`Com_CrudeOil`)
  - `Brent Oil` (`Com_BrentCrudeOil`)
- **예측 단위**
  - 주간 예측
- **데이터 구간 세팅**
  - TrainSet 기간: `2013-04-01 ~ 2023-10-23` (총 `552주`)
  - ValidationSet 기간: `2023-10-30 ~ 2025-10-20` (총 `24개 fold`, 실질 평가영역 `104주`)
    - Cross-Validation Fold 1: `2023-10-30 ~ 2024-01-15` (총 `12주`)
    - Cross-Validation Fold 2: `2023-11-27 ~ 2024-02-12` (총 `12주`)
    - Cross-Validation Fold 3: `2023-12-25 ~ 2024-03-11` (총 `12주`)
    - ...
    - Cross-Validation Fold 24: `2025-08-04 ~ 2025-10-20` (총 `12주`)
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)
  - 전체 데이터 구간: `2013-04-01 ~ 2026-01-12` (총 `668주`)
  - 최근 평가영역 길이: `116주 = 104주(ts-cv) + 12주(holdout) = 52*2 + 12`
- **피처리스트**
  - 기본모델 `Naive`
    - 외생변수 없음
    - forecast origin의 마지막 관측치 `y_t`를 향후 `12주`에 그대로 반복
  - 실험모델 `PatchTST`
    - 외생변수 없음
    - 각 타깃은 별도의 단변량 시계열로 구성되며, 최근 `48주` 값만 입력으로 사용
  - Residual 보정모델
    - train 내부 calibration tail에서 생성한 `out-of-sample residual series`의 최근 `48주` residual만 입력으로 사용
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
  - 기본모델 `Naive`

```python
naive_params = {
    "strategy": "repeat_last_observation",
    "forecast_rule": "y_hat[t+1:t+12] = y_t",
}
```

  - 실험모델 `PatchTST`

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

  - Residual 보정모델
    - `NLinear`
    - `XGB`
    - `LGBM`

```python
residual_params = {
    "NLinear": {
        "learning_rate": 0.001,
        "max_steps": 2000,
        "batch_size": 32,
    },
    "XGB": {
        "n_estimators": 400,
        "learning_rate": 0.03,
    },
    "LGBM": {
        "n_estimators": 400,
        "learning_rate": 0.03,
    },
}
```

- **지표 정의**
  - `RMSE`, `MAE`, `MAPE`는 일반적인 정의를 사용하였다.
  - `NRMSE`는 `RMSE / abs(mean(y_true)) * 100`으로 계산하였다.

# 03. 실험 설계 및 적용

---

- 기본모델 `Naive`는 각 forecast origin에서 마지막 관측치를 다음 `12주` 전체에 반복하는 방식으로 예측하였다.
- 실험모델 `PatchTST`는 동일한 Train/Validation/Test 프로토콜에서 `48 -> 12` direct forecast를 수행하였다.
- residual 보정모델은 `PatchTST`의 train 내부 calibration tail에서 생성한 `out-of-sample residual series`를 입력으로 받아 `48 -> 12` direct residual forecast를 수행하였다.
- calibration residual 생성에는 학습 window의 마지막 `104개` supervised windows를 사용했다.
- 코드 점검 후 calibration model 학습 window에 `horizon-1` embargo를 추가하여, calibration target 구간이 학습 label에 겹치지 않도록 수정하였다.
- 최종 비교군은 아래 다섯 종류로 구성하였다.
  - Bench-mark: `Naive`
  - Experimental: `PatchTST`
  - Residual Correction: `PatchTST + NLinear`
  - Residual Correction: `PatchTST + XGB`
  - Residual Correction: `PatchTST + LGBM`
- ValidationSet에서는 fold 평균 성능으로 모델을 해석하고, TestSet은 최종 성능 보고에만 사용하였다.
- 본 보고서는 `Naive`를 포함한 직접 비교본이며, `Naive`를 제외한 strict 제출용 버전은 `REPORT_ACADEMIC_VALIDATED_NO_NAIVE.md`에 따로 정리한다.

# 04. 실험(모델링) 결과

### 04-01. 결과 요약 (MAPE 기준)

---

| Target | Bench-mark (%) | 실험모델 (%) | Residual-NLinear (%) | Residual-XGB (%) | Residual-LGBM (%) | Bench-mark 대비 최종 증감 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WTI Oil | 6.300 | 5.379 | 17.493 | 18.435 | 18.080 | -0.921 |
| Brent Oil | 5.516 | 4.786 | 16.612 | 17.273 | 17.570 | -0.730 |

- `ts-cv 평균` 기준 최종 채택 모델은 두 타깃 모두 `PatchTST`였다.
- `holdout` 기준 최저오차 모델은 두 타깃 모두 `Naive`였다.
- 따라서 본 비교본에서는 `평균 ts-cv`와 `최종 holdout`의 모델 순위가 다르다는 점을 함께 해석해야 한다.

#### 핵심 Leaderboard (ts-cv 평균)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | Naive | - | 5.186 | 4.484 | 6.300 | 7.265 |
| WTI Oil | PatchTST | - | 4.435 | 3.815 | 5.379 | 6.230 |
| WTI Oil | PatchTST | NLinear | 13.389 | 12.742 | 17.493 | 18.245 |
| WTI Oil | PatchTST | XGB | 13.977 | 13.407 | 18.435 | 19.065 |
| WTI Oil | PatchTST | LGBM | 13.745 | 13.201 | 18.080 | 18.726 |
| Brent Oil | Naive | - | 4.817 | 4.120 | 5.516 | 6.422 |
| Brent Oil | PatchTST | - | 4.179 | 3.548 | 4.786 | 5.596 |
| Brent Oil | PatchTST | NLinear | 13.371 | 12.738 | 16.612 | 17.325 |
| Brent Oil | PatchTST | XGB | 13.753 | 13.256 | 17.273 | 17.806 |
| Brent Oil | PatchTST | LGBM | 14.107 | 13.512 | 17.570 | 18.246 |

#### 핵심 Leaderboard (holdout)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | Naive | - | 1.648 | 1.279 | 2.212 | 2.804 |
| WTI Oil | PatchTST | - | 1.724 | 1.466 | 2.475 | 2.933 |
| WTI Oil | PatchTST | NLinear | 4.306 | 3.776 | 6.374 | 7.327 |
| WTI Oil | PatchTST | XGB | 5.930 | 5.391 | 9.197 | 10.089 |
| WTI Oil | PatchTST | LGBM | 4.485 | 3.910 | 6.663 | 7.631 |
| Brent Oil | Naive | - | 1.818 | 1.383 | 2.247 | 2.900 |
| Brent Oil | PatchTST | - | 1.723 | 1.581 | 2.515 | 2.749 |
| Brent Oil | PatchTST | NLinear | 3.596 | 3.247 | 5.145 | 5.736 |
| Brent Oil | PatchTST | XGB | 4.855 | 4.639 | 7.391 | 7.743 |
| Brent Oil | PatchTST | LGBM | 4.625 | 4.245 | 6.751 | 7.377 |

### 04-02. 세부 결과

---

- **Test Set Metric**
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)

#### WTI Oil

| Model Type | Base Model | Residual Model | RMSE | MAE | MAPE (%) | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Bench-mark | Naive | - | 1.648 | 1.279 | 2.212 | 2.804 |
| Experimental | PatchTST | - | 1.724 | 1.466 | 2.475 | 2.933 |
| Residual Correction | PatchTST | NLinear | 4.306 | 3.776 | 6.374 | 7.327 |
| Residual Correction | PatchTST | XGB | 5.930 | 5.391 | 9.197 | 10.089 |
| Residual Correction | PatchTST | LGBM | 4.485 | 3.910 | 6.663 | 7.631 |

#### Brent Oil

| Model Type | Base Model | Residual Model | RMSE | MAE | MAPE (%) | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Bench-mark | Naive | - | 1.818 | 1.383 | 2.247 | 2.900 |
| Experimental | PatchTST | - | 1.723 | 1.581 | 2.515 | 2.749 |
| Residual Correction | PatchTST | NLinear | 3.596 | 3.247 | 5.145 | 5.736 |
| Residual Correction | PatchTST | XGB | 4.855 | 4.639 | 7.391 | 7.743 |
| Residual Correction | PatchTST | LGBM | 4.625 | 4.245 | 6.751 | 7.377 |

- **ts-cv leaderboard 표**
  - 아래 표는 `Naive`를 포함한 직접 비교본의 메인 표로 사용한다.
  - `Target`, `Base Model`, `Residual Model`, `RMSE`, `MAE`, `MAPE`, `NRMSE` 순으로 정렬한다.

| Target | Base Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | Naive | - | 5.186 | 4.484 | 6.300 | 7.265 |
| WTI Oil | PatchTST | - | 4.435 | 3.815 | 5.379 | 6.230 |
| WTI Oil | PatchTST | NLinear | 13.389 | 12.742 | 17.493 | 18.245 |
| WTI Oil | PatchTST | XGB | 13.977 | 13.407 | 18.435 | 19.065 |
| WTI Oil | PatchTST | LGBM | 13.745 | 13.201 | 18.080 | 18.726 |
| Brent Oil | Naive | - | 4.817 | 4.120 | 5.516 | 6.422 |
| Brent Oil | PatchTST | - | 4.179 | 3.548 | 4.786 | 5.596 |
| Brent Oil | PatchTST | NLinear | 13.371 | 12.738 | 16.612 | 17.325 |
| Brent Oil | PatchTST | XGB | 13.753 | 13.256 | 17.273 | 17.806 |
| Brent Oil | PatchTST | LGBM | 14.107 | 13.512 | 17.570 | 18.246 |

- **Plot**
  - `Naive` 포함 비교본은 별도 PNG 시각화를 재생성하지 않았으므로, 본 문서에서는 위 수치표를 기준 결과로 사용한다.

# 05. 결론 및 얻게 된 인사이트

---

- `ts-cv 평균` 기준에서는 `WTI Oil`, `Brent Oil` 모두에서 `PatchTST`가 `Naive`보다 `RMSE`, `MAE`, `MAPE`, `NRMSE` 전 지표에서 우수했다.
- 반면 `최종 holdout 12주`에서는 두 타깃 모두 `Naive`가 `PatchTST`보다 더 낮은 `MAPE`를 기록했다.
- 코드 감사로 calibration residual leakage를 제거한 뒤, residual 보정모델(`NLinear`, `XGB`, `LGBM`)의 수치는 이전 결과와 다르게 재계산되었지만 `Naive`와 `PatchTST`를 안정적으로 상회하지는 못했다.
- 따라서 본 비교본에서 가장 신뢰 가능한 해석은 “`PatchTST`는 평균적 일반화 성능에서는 `Naive`보다 우수하지만, 마지막 holdout 단일 구간에서는 `Naive`가 더 강했다”는 점이다.

# 06. 향후 Action Plan

---

- 제출용으로는 `Naive 제외본`과 `Naive 포함본`을 분리해 사용한다.
- `Naive`를 기준선으로 계속 유지하되, 보고서에는 `ts-cv 평균`과 `holdout`의 순위 차이를 명시한다.
- `PatchTST`의 holdout underforecast 성향을 줄이기 위해 `Naive`와의 고정 가중 앙상블 또는 level-adjusted calibration을 별도 후속 실험으로 검증한다.
