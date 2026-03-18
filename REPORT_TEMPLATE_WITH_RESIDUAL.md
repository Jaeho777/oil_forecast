# 01. 핵심쟁점

---

- “기본모델(Bench-mark)”과 “실험모델(OOO 방법론 적용)”을 비교했을 때, 예측 성능 향상이 있는지 확인한다.
- 실험모델에 residual 보정모델을 추가했을 때, 기본모델 및 실험모델 대비 추가 성능 개선이 있는지 확인한다.
- 성능 비교는 `RMSE`, `MAE`, `MAPE`, `NRMSE` 기준으로 수행한다.

# 02. 데이터 및 모델 세팅

---

- **예측 타깃**
  - “OOO”의 현물 가격
- **예측 단위**
  - 주간 예측
- **데이터 구간 세팅**
  - TrainSet 기간: `2000-00-00 ~ 2000-00-00` (총 `00주`)
  - ValidationSet 기간: `2000-00-00 ~ 2000-00-00`
    - Cross-Validation Fold 1: `2000-00-00 ~ 2000-00-00` (총 `00주`)
    - Cross-Validation Fold 2: `2000-00-00 ~ 2000-00-00` (총 `00주`)
    - Cross-Validation Fold 3: `2000-00-00 ~ 2000-00-00` (총 `00주`)
    - ...
    - Cross-Validation Fold n: `2000-00-00 ~ 2000-00-00` (총 `00주`)
  - TestSet 기간: `2000-00-00 ~ 2000-00-00` (총 `00주`)
- **피처리스트**
  - 기본모델 (총 56개)

```python
'Year', 'Month', 'WeekByYear', 'WeekByMonth',
'lag_1', 'lag_2', 'lag_3','lag_4', 'lag_5', 'lag_6',
'diff_lag1_lag2', 'diff_lag2_lag3', 'diff_lag3_lag4', 'diff_lag4_lag5', 'diff_lag5_lag6',
'mean_window_3', 'mean_window_6', 'std_window_3', 'std_window_6',
'USD_CNY', 'USD_KRW', 'USD_JPY', 'USD_AUD', 'USD_BRL', 'USD_DXY',
'Stocks_US500', 'Stocks_USVIX', 'Stocks_CH50', 'Stocks_CSI300',
'Stocks_SHANGHAI50', 'Stocks_SHANGHAI', 'Stocks_HK50', 'Stocks_GEI',
'Bonds_CHN_30Y', 'Bonds_CHN_20Y', 'Bonds_CHN_10Y', 'Bonds_CHN_5Y', 'Bonds_CHN_2Y','Bonds_CHN_1Y',
'Bonds_US_10Y', 'Bonds_US_2Y', 'Bonds_US_1Y', 'Bonds_US_3M',
'Bonds_AUS_10Y', 'Bonds_AUS_1Y', 'Bonds_BRZ_10Y', 'Bonds_BRZ_1Y',
'Bonds_KOR_10Y', 'Bonds_KOR_1Y',
'Com_CrudeOil', 'Com_BrentCrudeOil', 'Com_Gasoline', 'Com_NaturalGas',
'Com_Uranium', 'Com_LME_Index', 'Com_HRC_Steel'
```

  - 실험모델 (총 78개)
    - 기본모델 56개 + 실험변수 22개

```python
'EPU_GEPU_current', 'EPU_GEPU_ppp', 'EPU_Australia', 'EPU_Brazil',
'EPU_Canada', 'EPU_Chile', 'EPU_Hybrid China', 'EPU_France',
'EPU_Germany', 'EPU_Greece', 'EPU_India', 'EPU_Ireland', 'EPU_Italy',
'EPU_Japan', 'EPU_Korea', 'EPU_Pakistan', 'EPU_Russia', 'EPU_Spain',
'EPU_Singapore', 'EPU_UK', 'EPU_US', 'EPU_Mainland China'
```

- **모델 세팅**
  - 기본모델 `OOO`

```python
# HyperParameters
params = {}
```

  - 실험모델 `OOO`

```python
# HyperParameters
params = {}
```

  - Residual 보정모델
    - `NLinear`
    - `XGB`
    - `LGBM`

```python
# HyperParameters
residual_params = {
    "NLinear": {},
    "XGB": {},
    "LGBM": {},
}
```

# 03. 실험 설계 및 적용

---

- 기본모델을 먼저 학습한 뒤, 동일한 Train/Validation/Test 프로토콜로 실험모델을 학습하였다.
- 실험모델의 예측오차(residual)를 별도 학습 대상으로 두고, `NLinear`, `XGB`, `LGBM` 보정모델을 각각 적용하였다.
- 최종 비교군은 아래 네 종류로 구성하였다.
  - Bench-mark: 기본모델
  - Experimental: 실험모델
  - Experimental + Residual(NLinear)
  - Experimental + Residual(XGB)
  - Experimental + Residual(LGBM)
- ValidationSet에서는 fold 평균 성능으로 모델을 선정하고, TestSet은 최종 성능 보고에만 사용하였다.

# 04. 실험(모델링) 결과

### 04-01. 결과 요약 (MAPE 기준)

---

| Target | Bench-mark (%) | 실험모델 (%) | Residual-NLinear (%) | Residual-XGB (%) | Residual-LGBM (%) | Bench-mark 대비 최종 증감 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nickel | 2.01 | 1.89 | 1.84 | 1.80 | 1.78 | -0.23 |
| Copper |  |  |  |  |  |  |
| Aluminum |  |  |  |  |  |  |

작성 규칙:
- 마지막 열은 최종 채택 모델의 `MAPE - Bench-mark MAPE`로 계산한다.
- 최종 채택 모델은 `Residual-LGBM`일 수도 있고, `Residual-XGB`, `Residual-NLinear`, 혹은 residual 미적용 실험모델일 수도 있다.

#### 핵심 Leaderboard

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - |  |  |  |  |
| WTI Oil | PatchTST | NLinear |  |  |  |  |
| WTI Oil | PatchTST | XGB |  |  |  |  |
| WTI Oil | PatchTST | LGBM |  |  |  |  |
| Brent Oil | PatchTST | - |  |  |  |  |
| Brent Oil | PatchTST | NLinear |  |  |  |  |
| Brent Oil | PatchTST | XGB |  |  |  |  |
| Brent Oil | PatchTST | LGBM |  |  |  |  |

작성 규칙:
- 결과 요약 바로 아래에 넣는 핵심 표다.
- `Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE` 형식을 고정한다.
- 일반적으로 `ts-cv 평균 leaderboard`를 넣고, 필요하면 holdout 기준 별도 표를 추가한다.

### 04-02. 세부 결과

---

- **Test Set Metric**
  - TestSet 기간: `2000-00-00 ~ 2000-00-00` (총 `00주`)

| Model Type | Base Model | Residual Model | RMSE | MAE | MAPE (%) | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Bench-mark | OOO | - |  |  |  |  |
| Experimental | OOO | - |  |  |  |  |
| Residual Correction | OOO | NLinear |  |  |  |  |
| Residual Correction | OOO | XGB |  |  |  |  |
| Residual Correction | OOO | LGBM |  |  |  |  |

- **ts-cv leaderboard 표**
  - 아래 표는 논문/보고서용 메인 표로 사용한다.
  - 이미지 예시와 동일하게 `Target`, `Base Model`, `Residual Model`, `RMSE`, `MAE`, `MAPE`, `NRMSE` 순으로 정렬한다.

| Target | Base Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WTI Oil | PatchTST | - |  |  |  |  |
| WTI Oil | PatchTST | NLinear |  |  |  |  |
| WTI Oil | PatchTST | XGB |  |  |  |  |
| WTI Oil | PatchTST | LGBM |  |  |  |  |
| Brent Oil | PatchTST | - |  |  |  |  |
| Brent Oil | PatchTST | NLinear |  |  |  |  |
| Brent Oil | PatchTST | XGB |  |  |  |  |
| Brent Oil | PatchTST | LGBM |  |  |  |  |

- **Plot**
  - TestSet 실제값 vs 예측값 비교 그림
  - Bench-mark / Experimental / Residual Correction 모델을 모두 한 그림에 표시

# 05. 결론 및 얻게 된 인사이트

---

- 실험모델은 Bench-mark 대비 `MAPE`, `RMSE`, `MAE`, `NRMSE` 기준에서 `OOO` 수준의 개선을 보였다.
- Residual 보정모델 중 `OOO`가 가장 안정적으로 성능을 개선하였다.
- 다만 residual 보정모델의 개선폭은 타깃별로 차이가 있었으며, 일부 타깃에서는 Bench-mark 대비 개선은 있었지만 실험모델 대비 추가 개선폭은 제한적이었다.

# 06. 향후 Action Plan

---

- 타깃별로 residual 보정모델의 일관성을 추가 검증한다.
- ValidationSet과 TestSet 간 성능 차이가 큰 경우, 과적합 여부를 재점검한다.
- 최종 채택 모델 기준으로 feature importance 또는 residual error analysis를 추가 수행한다.
