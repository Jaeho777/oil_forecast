# Oil Forecast Reproduction Guide

이 문서는 팀원이 새 환경에서 같은 실험을 다시 돌릴 때, 어떤 조건을 고정하고 어떤 순서로 실행해야 하는지 설명하는 재현 가이드다.

## 1. 공통 실험 규칙

- 타깃변수는 `Com_BrentCrudeOil`로 고정한다.
- 데이터 파일은 주간 데이터이며 `dt` 컬럼을 날짜로 변환한 뒤 오름차순 정렬한다.
- 분할은 아래처럼 고정한다.
  - Train: `2013-04-01 ~ 2025-07-28`
  - Validation: `2025-08-04 ~ 2025-10-20`
  - Test: `2025-10-27 ~ 2026-01-12`
- 모든 설명변수는 `shift(1)`을 적용한다.
  - 즉 `X_(t-1)`로 `y_t`를 예측한다.
  - 이유는 동시점 정보 누수를 막기 위해서다.
- 모델 선택은 `validation`으로만 한다.
- `test`는 최종 보고용으로만 사용한다.
- 보고 지표는 `RMSE`, `MAE`, `MAPE(%)`, `NRMSE(%)`, `R²`로 통일한다.

## 2. 환경

- Python `3.8+`
- 의존성 설치:

```bash
pip install -r requirements.txt
```

- 현재 주요 패키지:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `lightgbm`
  - `shap`
  - `torch`
  - `statsmodels`

## 3. 구조 실험: ExponentialSmoothing + NLinear + LightGBM

이 레포에는 결과와 노트북만 남겨두었고, 아래 내용은 구조 실험을 재구현하기 위한 명세다.

### 3-1. 입력 변수 구성

이 실험은 자동 변수선택을 하지 않는다. 도메인 기반으로 미리 정한 22개 원변수와 33개 파생변수를 모두 사용한다.

원변수 22개:
- `Com_Gasoline`
- `Com_NaturalGas`
- `Com_Uranium`
- `Com_Coal`
- `Com_LME_Cu_Cash`
- `Com_Steel`
- `Com_Iron_Ore`
- `Idx_DxyUSD`
- `EX_USD_CNY`
- `Bonds_US_10Y`
- `Bonds_US_2Y`
- `Bonds_US_3M`
- `Idx_SnPVIX`
- `Com_Gold`
- `Idx_SnP500`
- `Idx_CSI300`
- `EX_USD_KRW`
- `Bonds_KOR_10Y`
- `EX_USD_JPY`
- `Com_Corn`
- `Com_Soybeans`
- `Com_PalmOil`

파생변수 33개:
- 위 22개 원변수 각각의 주간 수익률 `pct_change()`
- `Spread_US_10Y_2Y = Bonds_US_10Y - Bonds_US_2Y`
- `Spread_Crack = Com_Gasoline - Com_BrentCrudeOil`
- `Ratio_Gold_Oil = Com_Gold / Com_BrentCrudeOil`
- 아래 4개 변수에 대한 `ma4_ratio`, `ma12_ratio`
  - `Com_Gasoline`
  - `Com_NaturalGas`
  - `Idx_SnPVIX`
  - `Idx_DxyUSD`

총 feature 수:
- `22 raw + 33 derived = 55`

모든 feature는 마지막에 `shift(1)` 한다.

### 3-2. Stage 1: ExponentialSmoothing baseline

- 모델:
  - `trend="add"`
  - `seasonal="add"`
  - `seasonal_periods=52`
  - `initialization_method="estimated"`
  - `use_boxcox=False`
- train `y`에 한 번 fit
- validation 12주 + test 12주를 한 번에 multi-step forecast

즉 이 단계는 외생변수 없이 유가 시계열 자체만으로 baseline 예측을 만든다.

### 3-3. Stage 2: NLinear로 1차 잔차 보정

1차 잔차는 아래처럼 정의한다.

```text
resid1 = actual_price - baseline_prediction
```

NLinear 입력:
- 최근 `24주` residual sequence
- 현재 시점의 `55개 lagged exogenous feature`

전처리:
- exogenous feature는 `RobustScaler`를 train에만 fit
- residual 자체는 별도 scaling하지 않음

NLinear 설정:
- `SEQ_LEN = 24`
- `PRED_LEN = 1`
- `D_HIDDEN = 64`
- `LR = 1e-3`
- `N_EPOCHS = 300`
- `BATCH_SIZE = 32`
- `PATIENCE = 40`
- `N_SEEDS = 10`
- validation residual RMSE 기준 상위 `TOP_K = 5` seed ensemble

예측 방식:
- validation: `teacher forcing`
- test: `autoregressive`

결합식:

```text
stage2_pred = baseline_pred + nlinear_residual_pred
```

### 3-4. Stage 3: LightGBM으로 2차 잔차 보정

2차 잔차는 아래처럼 정의한다.

```text
ror = actual_resid1 - predicted_resid1
```

입력:
- 같은 `55개 lagged feature`

학습 방식:
- `expanding-window 5-fold OOF`

파라미터:
- `learning_rate = 0.02`
- `num_leaves = 10`
- `min_child_samples = 40`
- `subsample = 0.6`
- `colsample_bytree = 0.5`
- `reg_alpha = 2.0`
- `reg_lambda = 2.0`
- OOF model `n_estimators = 500`
- final model `n_estimators = 300`

최종 결합식:

```text
final_pred = baseline_pred + nlinear_pred + lambda * lgbm_ror_pred
```

여기서 `lambda`는 validation gate다.
- stage3 validation RMSE가 stage2 validation RMSE보다 낮을 때만 `lambda = 1`
- 아니면 `lambda = 0`

### 3-5. 검산 숫자

구조 실험의 test 결과는 아래처럼 나와야 한다.

- `ExpSmoothing (Baseline)` test RMSE `4.1754`
- `Baseline + NLinear` test RMSE `1.2648`
- `Full Framework (B+NL+RoR)` test RMSE `1.2275`

최종 full framework의 test 지표:
- RMSE `1.2275`
- MAE `0.9144`
- MAPE `1.4708%`
- NRMSE `1.9580%`
- R² `0.1809`

결과 파일:
- `output_oil_academic/results_table.csv`
- `output_oil_academic/results_val_table.csv`

## 4. Transformer 실험: PatchTST + iTransformer

이 항목도 실행 스크립트가 아니라 재구현 명세로 읽으면 된다.

### 4-1. 입력 후보 변수 구성

시작점은 구조 실험과 유사하다.
- 원변수 22개
- 파생변수 생성
- 전부 `shift(1)`

하지만 실제 모델 입력은 전체를 다 쓰지 않는다.

### 4-2. Feature selection

이 실험은 train 데이터에서만 SHAP + TimeSeriesCV로 변수를 고른다.

절차:
1. train 구간에서 LightGBM 학습
   - `n_estimators = 300`
   - `learning_rate = 0.05`
   - `num_leaves = 31`
   - `random_state = 42`
2. SHAP mean absolute importance 계산
3. 변수 개수 후보 `[10, 15, 20, 25, 30, 40, 전체]` 생성
4. 각 후보에 대해 `TimeSeriesSplit(n_splits=5)` 수행
5. fold마다 LightGBM 학습
   - `n_estimators = 200`
   - `learning_rate = 0.05`
   - `num_leaves = 20`
   - `random_state = 42`
6. 평균 CV RMSE가 가장 낮은 변수 개수를 선택

현재 clean run에서 선택된 변수 수:
- `10개`

현재 clean run의 선택 변수 10개:
- `Spread_Crack`
- `EX_USD_KRW`
- `Com_Gasoline`
- `Ratio_Gold_Oil`
- `Com_Coal`
- `Com_Gasoline_ma12r`
- `Com_PalmOil`
- `Idx_SnPVIX`
- `Bonds_US_10Y`
- `Bonds_US_3M_ret`

### 4-3. 전처리와 시퀀스

- 선택된 10개 변수만 사용
- `RobustScaler`를 train에만 fit
- y는 train 평균/표준편차로 표준화
- 시퀀스 길이 `24주`
- 예측 길이 `1주`

train 전체를 겹치는 24주 창으로 잘라서 학습한다.
예:
- 1~24주 입력 -> 25주 예측
- 2~25주 입력 -> 26주 예측
- ...

validation/test도 직전 24주 buffer를 붙여 같은 방식으로 시퀀스를 만든다.

### 4-4. 모델 구조

1단계 baseline:
- `PatchTST`
- 설정:
  - `d_model = 64`
  - `n_heads = 4`
  - `n_layers = 2`
  - `patch_len = 4`
  - `stride = 2`
  - `dropout = 0.2`

2단계 잔차 보정:
- `iTransformer`
- 설정:
  - `d_model = 64`
  - `n_heads = 4`
  - `n_layers = 2`
  - `dropout = 0.2`

1차 잔차 정의:

```text
resid1 = standardized_actual_y - stage1_prediction
```

결합식:

```text
stage2_pred = stage1_prediction + predicted_residual
```

마지막에 다시 가격 단위로 복원해서 평가한다.

### 4-5. 학습 방식

전체 25개 조합을 먼저 screening한다.

screening 설정:
- `1 seed`
- `epochs = 60`
- `patience = 12`

screening 상위 8개 조합만 confirmatory 재학습한다.

confirmatory 설정:
- `3 seeds`
- `epochs = 120`
- `patience = 20`

보고용 숫자는 confirmatory 기준이다.

### 4-6. 검산 숫자

`PatchTST+iTransformer` confirmatory test 결과:
- validation RMSE `1.3534`
- test RMSE `1.2308`
- test MAE `0.9460`
- test MAPE `1.5054%`
- test NRMSE `1.9633%`
- test R² `0.1764`

PatchTST baseline 자체의 숫자:
- validation RMSE `1.5235`
- test RMSE `1.5339`

결과 파일:
- `output_oil_transformer_clean/stage2_experiments.csv`
- `output_oil_transformer_clean/feature_shap.csv`
- `output_oil_transformer_clean/feature_selection_cv.csv`

## 5. PatchTST + NLinear 공정 비교

이 항목은 transformer framework 안에서 `PatchTST+iTransformer`와 같은 조건으로 다시 비교하는 명세다.

이 비교는 `PatchTST+iTransformer`와 같은 transformer framework 안에서, 2단계 residual model만 `iTransformer`에서 `NLinear`로 바꾼 것이다.

공정 비교 조건:
- 같은 train/validation/test 분할
- 같은 선택된 10개 변수
- 같은 24주 입력
- 같은 confirmatory 설정
  - `3 seeds`
  - `120 epochs`
  - `patience 20`

즉 baseline PatchTST까지는 동일하고, residual model만 바뀐다.

검산 숫자:
- validation RMSE `1.4303`
- test RMSE `1.9251`
- test MAE `1.3985`
- test MAPE `2.2132%`
- test NRMSE `3.0707%`
- test R² `-1.0147`

결과 파일:
- `output_oil_transformer_clean/confirm_patchtst_nlinear.csv`

## 6. PatchTST baseline을 구조 실험에 넣은 별도 실험

이 항목은 stagewise 구조에서 baseline만 PatchTST로 교체한 별도 명세다.

이 실험은 `ExponentialSmoothing + NLinear + LightGBM` 구조에서 baseline만 `PatchTST`로 바꾼 것이다.

즉:
- feature selection은 하지 않는다
- 구조 실험과 같은 `55개 full feature`를 쓴다
- NLinear와 LightGBM 설정도 구조 실험과 동일하다
- 오직 baseline만 ES에서 PatchTST로 교체한다

검산 숫자:
- `PatchTST (Baseline)` test RMSE `1.4494`
- `PatchTST + NLinear` test RMSE `1.8609`
- `PatchTST + NLinear + LightGBM` test RMSE `1.8609`
- `RoR lambda = 0.0`

결과 파일:
- `output_oil_patchtst_nlinear_lgbm/results_table.csv`

## 7. 해석할 때 주의할 점

- `PatchTST+iTransformer`와 `ExponentialSmoothing+NLinear+LightGBM`은 같은 날짜 분할에서 평가했지만, 완전히 동일한 feature selection 철학을 쓴 쌍둥이 실험은 아니다.
- `PatchTST+iTransformer`는 SHAP + TimeSeriesCV로 고른 10개 변수를 사용하는 transformer 탐색 실험이다.
- `ExponentialSmoothing+NLinear+LightGBM`은 도메인 기반 55개 full feature를 사용하는 구조 실험이다.
- `PatchTST+NLinear`의 공정 비교는 반드시 `run_confirm_patchtst_nlinear.py`로 확인한다.
- `oil_patchtst_nlinear_lgbm.py` 결과는 `PatchTST+NLinear`와 이름이 비슷해 보여도, 구조 실험 baseline 교체 버전이라 별개다.

## 8. 팀원에게 짧게 설명할 때

`재현하려면 먼저 타깃을 Com_BrentCrudeOil로 고정하고, train 2013-04-01~2025-07-28, validation 2025-08-04~2025-10-20, test 2025-10-27~2026-01-12로 분할하고, 모든 설명변수를 shift(1)해야 한다. 구조 실험은 55개 lagged full feature를 쓰고 ExponentialSmoothing -> NLinear -> LightGBM 순으로 보정해서 최종 test RMSE 1.2275가 나와야 한다. Transformer 실험은 train에서만 SHAP+TimeSeriesCV로 10개 변수를 고르고 PatchTST -> iTransformer 순으로 보정해서 test RMSE 1.2308이 나와야 한다. 같은 조건에서 PatchTST 뒤 residual model만 NLinear로 바꾸면 test RMSE 1.9251이 나와야 하고, 구조 실험 baseline만 PatchTST로 바꾸면 final test RMSE는 1.8609가 나와야 한다.` 
