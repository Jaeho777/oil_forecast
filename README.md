# oil_forecast

이 레포는 유가 예측 실험 중 현재 핵심인 세 가지 축만 정리한 클린 버전이다.

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
