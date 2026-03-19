#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

os.makedirs("/tmp/matplotlib-cache", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


TARGETS = {
    "WTI Oil": "Com_CrudeOil",
    "Brent Oil": "Com_BrentCrudeOil",
}


@dataclass
class PatchTSTConfig:
    hidden_size: int = 128
    attention_heads: int = 16
    linear_hidden_size: int = 256
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.2
    encoder_layers: int = 3
    attn_dropout: float = 0.0
    fc_dropout: float = 0.2
    max_steps: int = 5000
    learning_rate: float = 1e-4
    scaler_type: str = "identity"
    batch_size: int = 32
    weight_decay: float = 1e-4
    eval_interval: int = 50
    patience_steps: int = 400


@dataclass
class ResidualNLinearConfig:
    learning_rate: float = 1e-3
    max_steps: int = 2000
    batch_size: int = 32
    weight_decay: float = 0.0
    eval_interval: int = 50
    patience_steps: int = 300


@dataclass
class TreeConfig:
    n_estimators: int = 400
    learning_rate: float = 0.03


@dataclass
class ExperimentConfig:
    data_path: str = "data_weekly_260120.csv"
    output_dir: str = "results/academic_patchtst_residuals"
    input_size: int = 48
    horizon: int = 12
    step_size: int = 4
    number_of_windows: int = 24
    final_holdout: int = 12
    season_length: int = 52
    residual_calibration_windows: int = 104
    seed: int = 42
    max_windows: int | None = None
    run_holdout: bool = True
    patchtst: PatchTSTConfig = field(default_factory=PatchTSTConfig)
    residual_nlinear: ResidualNLinearConfig = field(default_factory=ResidualNLinearConfig)
    xgb: TreeConfig = field(default_factory=TreeConfig)
    lgbm: TreeConfig = field(default_factory=TreeConfig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def mean_normalized_scale(y_true: np.ndarray) -> float:
    # Keep the report's mean-normalized NRMSE definition, but guard against
    # sign flips or near-zero denominators.
    return float(np.clip(np.abs(np.mean(y_true)), a_min=1e-8, a_max=None))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.clip(np.abs(y_true), a_min=1e-8, a_max=None)
    scale = mean_normalized_scale(y_true)
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0),
        "NRMSE": float(rmse(y_true, y_pred) / scale * 100.0),
    }


def split_model_components(model_name: str) -> tuple[str, str]:
    if model_name == "PatchTST":
        return "PatchTST", "-"
    if model_name == "PatchTST+NLinear":
        return "PatchTST", "NLinear"
    if model_name == "PatchTST+XGB":
        return "PatchTST", "XGB"
    if model_name == "PatchTST+LGBM":
        return "PatchTST", "LGBM"
    return model_name, "-"


def make_supervised_windows(
    values: np.ndarray,
    input_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = len(values) - input_size - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Need at least {input_size + horizon} observations, got {len(values)}."
        )

    X = np.zeros((n_samples, input_size), dtype=np.float32)
    Y = np.zeros((n_samples, horizon), dtype=np.float32)
    for idx in range(n_samples):
        X[idx] = values[idx : idx + input_size]
        Y[idx] = values[idx + input_size : idx + input_size + horizon]
    return X, Y


def split_train_val(
    X: np.ndarray,
    Y: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(X) < 4:
        return X, Y, X, Y

    val_size = max(horizon, len(X) // 10)
    val_size = min(val_size, len(X) - 1)
    split_at = len(X) - val_size
    # Leave a small embargo so the training windows do not share target
    # timestamps with the validation windows used for early stopping.
    train_end = max(1, split_at - (horizon - 1))
    return X[:train_end], Y[:train_end], X[split_at:], Y[split_at:]


class PatchTSTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_heads: int,
        linear_hidden_size: int,
        dropout: float,
        attn_dropout: float,
        fc_dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, linear_hidden_size),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_hidden_size, hidden_size),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.drop1(attn_output)
        ffn_input = self.norm2(x)
        x = x + self.drop2(self.ffn(ffn_input))
        return x


class PatchTSTForecaster(nn.Module):
    def __init__(self, input_size: int, horizon: int, config: PatchTSTConfig) -> None:
        super().__init__()
        if input_size < config.patch_len:
            raise ValueError("input_size must be >= patch_len for PatchTST.")

        self.input_size = input_size
        self.horizon = horizon
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.n_patches = (input_size - config.patch_len) // config.stride + 1

        self.patch_proj = nn.Linear(config.patch_len, config.hidden_size)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, config.hidden_size) * 0.02
        )
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                PatchTSTBlock(
                    hidden_size=config.hidden_size,
                    attention_heads=config.attention_heads,
                    linear_hidden_size=config.linear_hidden_size,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                    fc_dropout=config.fc_dropout,
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_patches * config.hidden_size, config.linear_hidden_size),
            nn.GELU(),
            nn.Dropout(config.fc_dropout),
            nn.Linear(config.linear_hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        z = self.patch_proj(patches)
        z = self.input_dropout(z + self.pos_embedding)
        for block in self.blocks:
            z = block(z)
        z = self.final_norm(z)
        return self.head(z)


class NLinearResidualForecaster(nn.Module):
    def __init__(self, input_size: int, horizon: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last_value = x[:, -1:]
        x_norm = x - last_value
        return self.linear(x_norm) + last_value


def iterate_minibatches(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
):
    indices = np.arange(len(X))
    while True:
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            yield X[batch_idx], Y[batch_idx]


def fit_torch_model(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    learning_rate: float,
    max_steps: int,
    batch_size: int,
    weight_decay: float,
    eval_interval: int,
    patience_steps: int,
    horizon: int,
    seed: int,
    device: torch.device,
) -> nn.Module:
    Xtr, Ytr, Xva, Yva = split_train_val(X, Y, horizon)
    Xva_tensor = torch.from_numpy(Xva).to(device)
    Yva_tensor = torch.from_numpy(Yva).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()
    model.to(device)
    model.train()

    rng = np.random.default_rng(seed)
    train_iter = iterate_minibatches(Xtr, Ytr, min(batch_size, len(Xtr)), rng)

    best_state = None
    best_val = float("inf")
    best_step = 0

    for step in range(1, max_steps + 1):
        batch_x, batch_y = next(train_iter)
        batch_x_tensor = torch.from_numpy(batch_x).to(device)
        batch_y_tensor = torch.from_numpy(batch_y).to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch_x_tensor), batch_y_tensor)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == 1 or step == max_steps:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xva_tensor), Yva_tensor).item()
            if val_loss < best_val:
                best_val = val_loss
                best_step = step
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            model.train()

            if step - best_step >= patience_steps:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def predict_torch_model(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.vstack(outputs)


class DirectTreeMultiOutput:
    def __init__(self, model_name: str, horizon: int, config: TreeConfig, seed: int) -> None:
        self.model_name = model_name
        self.horizon = horizon
        self.config = config
        self.seed = seed
        self.models: list[object] = []

    def _make_estimator(self, horizon_idx: int):
        if self.model_name == "XGB":
            return XGBRegressor(
                objective="reg:squarederror",
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=4,
                min_child_weight=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=self.seed + horizon_idx,
                n_jobs=1,
                verbosity=0,
            )

        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=self.seed + horizon_idx,
            verbose=-1,
            n_jobs=1,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "DirectTreeMultiOutput":
        self.models = []
        for horizon_idx in range(self.horizon):
            estimator = self._make_estimator(horizon_idx)
            estimator.fit(X, Y[:, horizon_idx])
            self.models.append(estimator)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [model.predict(X) for model in self.models]
        return np.column_stack(preds)


def fit_patchtst_from_windows(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> PatchTSTForecaster:
    set_seed(seed)
    model = PatchTSTForecaster(
        input_size=config.input_size,
        horizon=config.horizon,
        config=config.patchtst,
    )
    model = fit_torch_model(
        model=model,
        X=X_train,
        Y=Y_train,
        learning_rate=config.patchtst.learning_rate,
        max_steps=config.patchtst.max_steps,
        batch_size=config.patchtst.batch_size,
        weight_decay=config.patchtst.weight_decay,
        eval_interval=config.patchtst.eval_interval,
        patience_steps=config.patchtst.patience_steps,
        horizon=config.horizon,
        seed=seed,
        device=device,
    )
    return model


def fit_patchtst(
    train_series: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> tuple[PatchTSTForecaster, np.ndarray, np.ndarray]:
    X_train, Y_train = make_supervised_windows(
        train_series,
        input_size=config.input_size,
        horizon=config.horizon,
    )
    model = fit_patchtst_from_windows(
        X_train=X_train,
        Y_train=Y_train,
        config=config,
        device=device,
        seed=seed,
    )
    return model, X_train, Y_train


def make_oof_residual_series(
    train_series: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    X_all, Y_all = make_supervised_windows(
        train_series,
        input_size=config.input_size,
        horizon=config.horizon,
    )
    min_train_windows = max(4, config.horizon)
    max_calibration = len(X_all) - min_train_windows
    calibration_windows = min(config.residual_calibration_windows, max_calibration)
    if calibration_windows < config.input_size + 1:
        raise ValueError(
            "Not enough windows to build an out-of-sample residual calibration series."
        )

    split_at = len(X_all) - calibration_windows
    calibration_model = fit_patchtst_from_windows(
        X_train=X_all[:split_at],
        Y_train=Y_all[:split_at],
        config=config,
        device=device,
        seed=seed + 17,
    )
    window_preds = predict_torch_model(
        model=calibration_model,
        X=X_all[split_at:],
        device=device,
        batch_size=max(128, config.patchtst.batch_size),
    )
    fitted_sum = np.zeros(len(train_series), dtype=np.float64)
    fitted_count = np.zeros(len(train_series), dtype=np.float64)

    for local_idx, pred in enumerate(window_preds):
        idx = split_at + local_idx
        start = idx + config.input_size
        end = start + config.horizon
        fitted_sum[start:end] += pred
        fitted_count[start:end] += 1.0

    valid_start = split_at + config.input_size
    valid_mask = fitted_count > 0
    if not np.all(valid_mask[valid_start:]):
        raise RuntimeError("Failed to create a complete out-of-sample residual series.")

    fitted_values = fitted_sum[valid_mask] / fitted_count[valid_mask]
    actual_values = train_series[valid_mask]
    residual_series = (actual_values - fitted_values).astype(np.float32)
    min_residual_len = config.input_size + config.horizon
    if len(residual_series) < min_residual_len:
        raise ValueError(
            f"Residual calibration series is too short: need {min_residual_len}, got {len(residual_series)}."
        )
    return residual_series


def predict_patchtst_window(
    model: PatchTSTForecaster,
    train_series: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    last_window = train_series[-config.input_size :].astype(np.float32)[None, :]
    pred = predict_torch_model(
        model=model,
        X=last_window,
        device=device,
        batch_size=1,
    )[0]
    return pred.astype(np.float64)


def forecast_nlinear_residual(
    residual_series: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    X_resid, Y_resid = make_supervised_windows(
        residual_series,
        input_size=config.input_size,
        horizon=config.horizon,
    )
    set_seed(seed)
    model = NLinearResidualForecaster(
        input_size=config.input_size,
        horizon=config.horizon,
    )
    model = fit_torch_model(
        model=model,
        X=X_resid,
        Y=Y_resid,
        learning_rate=config.residual_nlinear.learning_rate,
        max_steps=config.residual_nlinear.max_steps,
        batch_size=config.residual_nlinear.batch_size,
        weight_decay=config.residual_nlinear.weight_decay,
        eval_interval=config.residual_nlinear.eval_interval,
        patience_steps=config.residual_nlinear.patience_steps,
        horizon=config.horizon,
        seed=seed,
        device=device,
    )
    last_residual_window = residual_series[-config.input_size :].astype(np.float32)[None, :]
    pred = predict_torch_model(model, last_residual_window, device=device, batch_size=1)[0]
    return pred.astype(np.float64)


def forecast_tree_residual(
    model_name: str,
    residual_series: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    X_resid, Y_resid = make_supervised_windows(
        residual_series,
        input_size=config.input_size,
        horizon=config.horizon,
    )
    tree_cfg = config.xgb if model_name == "XGB" else config.lgbm
    model = DirectTreeMultiOutput(
        model_name=model_name,
        horizon=config.horizon,
        config=tree_cfg,
        seed=config.seed + (202 if model_name == "XGB" else 303),
    )
    model.fit(X_resid, Y_resid)
    last_residual_window = residual_series[-config.input_size :].astype(np.float32)[None, :]
    pred = model.predict(last_residual_window)[0]
    return pred.astype(np.float64)


def window_result_row(
    target_label: str,
    window_name: str,
    train_end: int,
    forecast_dates: pd.Series,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residual_series_length: int | None = None,
    residual_supervised_samples: int | None = None,
) -> dict[str, object]:
    metrics = compute_metrics(y_true, y_pred)
    base_model, residual_model = split_model_components(model_name)
    return {
        "target": target_label,
        "window": window_name,
        "train_size": int(train_end),
        "forecast_start": forecast_dates.iloc[0].strftime("%Y-%m-%d"),
        "forecast_end": forecast_dates.iloc[-1].strftime("%Y-%m-%d"),
        "model": model_name,
        "base_model": base_model,
        "residual_model": residual_model,
        "residual_series_length": residual_series_length,
        "residual_supervised_samples": residual_supervised_samples,
        **metrics,
    }


def prediction_rows(
    target_label: str,
    window_name: str,
    model_name: str,
    forecast_dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    base_model, residual_model = split_model_components(model_name)
    rows = []
    for date, actual, pred in zip(forecast_dates, y_true, y_pred):
        rows.append(
            {
                "target": target_label,
                "window": window_name,
                "model": model_name,
                "base_model": base_model,
                "residual_model": residual_model,
                "ds": date.strftime("%Y-%m-%d"),
                "actual": float(actual),
                "prediction": float(pred),
            }
        )
    return rows


def build_cv_train_ends(n_obs: int, config: ExperimentConfig) -> list[int]:
    first_train_end = n_obs - (
        config.final_holdout
        + config.horizon
        + (config.number_of_windows - 1) * config.step_size
    )
    if first_train_end < config.input_size + config.horizon:
        raise ValueError("The configured windows leave too little data for training.")

    train_ends = [
        first_train_end + window_idx * config.step_size
        for window_idx in range(config.number_of_windows)
    ]
    if config.max_windows is not None:
        train_ends = train_ends[: config.max_windows]
    return train_ends


def run_single_window(
    target_label: str,
    target_series: pd.Series,
    dates: pd.Series,
    train_end: int,
    window_name: str,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_values = target_series.iloc[:train_end].to_numpy(dtype=np.float32)
    test_values = target_series.iloc[train_end : train_end + config.horizon].to_numpy(
        dtype=np.float64
    )
    forecast_dates = dates.iloc[train_end : train_end + config.horizon].reset_index(drop=True)

    baseline_model, _, _ = fit_patchtst(
        train_series=train_values,
        config=config,
        device=device,
        seed=config.seed + train_end,
    )
    baseline_forecast = predict_patchtst_window(baseline_model, train_values, config, device)
    residual_series = make_oof_residual_series(
        train_series=train_values,
        config=config,
        device=device,
        seed=config.seed + train_end,
    )
    residual_supervised_samples = len(residual_series) - config.input_size - config.horizon + 1

    rows = [
        window_result_row(
            target_label=target_label,
            window_name=window_name,
            train_end=train_end,
            forecast_dates=forecast_dates,
            model_name="PatchTST",
            y_true=test_values,
            y_pred=baseline_forecast,
            residual_series_length=len(residual_series),
            residual_supervised_samples=residual_supervised_samples,
        )
    ]
    pred_rows = prediction_rows(
        target_label=target_label,
        window_name=window_name,
        model_name="PatchTST",
        forecast_dates=forecast_dates,
        y_true=test_values,
        y_pred=baseline_forecast,
    )

    nlinear_resid = forecast_nlinear_residual(
        residual_series,
        config,
        device,
        seed=config.seed + train_end + 101,
    )
    xgb_resid = forecast_tree_residual("XGB", residual_series, config)
    lgbm_resid = forecast_tree_residual("LGBM", residual_series, config)

    corrections = {
        "PatchTST+NLinear": baseline_forecast + nlinear_resid,
        "PatchTST+XGB": baseline_forecast + xgb_resid,
        "PatchTST+LGBM": baseline_forecast + lgbm_resid,
    }

    for model_name, forecast in corrections.items():
        rows.append(
            window_result_row(
                target_label=target_label,
                window_name=window_name,
                train_end=train_end,
                forecast_dates=forecast_dates,
                model_name=model_name,
                y_true=test_values,
                y_pred=forecast,
                residual_series_length=len(residual_series),
                residual_supervised_samples=residual_supervised_samples,
            )
        )
        pred_rows.extend(
            prediction_rows(
                target_label=target_label,
                window_name=window_name,
                model_name=model_name,
                forecast_dates=forecast_dates,
                y_true=test_values,
                y_pred=forecast,
            )
        )

    return rows, pred_rows


def summarize_windows(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            ["target", "window", "model", "base_model", "residual_model"],
            as_index=False,
        )[["RMSE", "MAE", "MAPE", "NRMSE"]]
        .mean()
        .sort_values(["target", "window", "RMSE"])
        .reset_index(drop=True)
    )
    return summary


def summarize_tscv(df: pd.DataFrame) -> pd.DataFrame:
    tscv_rows = df[df["window"].str.startswith("tscv_")]
    summary = (
        tscv_rows.groupby(
            ["target", "model", "base_model", "residual_model"],
            as_index=False,
        )[["RMSE", "MAE", "MAPE", "NRMSE"]]
        .mean()
        .rename(
            columns={
                "RMSE": "mean_RMSE",
                "MAE": "mean_MAE",
                "MAPE": "mean_MAPE",
                "NRMSE": "mean_NRMSE",
            }
        )
        .sort_values(["target", "mean_RMSE"])
        .reset_index(drop=True)
    )
    return summary


def make_leaderboard(
    df: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    leaderboard = (
        df[["target", "base_model", "residual_model", *metric_columns]]
        .rename(
            columns={
                "target": "Target",
                "base_model": "Base Model",
                "residual_model": "Residual Model",
            }
        )
        .sort_values(["Target", metric_columns[0], "Residual Model"])
        .reset_index(drop=True)
    )
    return leaderboard


def load_dataframe(config: ExperimentConfig) -> pd.DataFrame:
    df = pd.read_csv(config.data_path, parse_dates=["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    cols = ["dt"] + list(TARGETS.values())
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df[cols].dropna().reset_index(drop=True)
    if len(df) != 668:
        raise ValueError(f"Expected 668 weekly observations, found {len(df)}.")
    return df


def run_experiment(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    device = get_device()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(config)
    dates = df["dt"]
    cv_train_ends = build_cv_train_ends(len(df), config)
    holdout_train_end = len(df) - config.final_holdout

    print("=" * 72, flush=True)
    print("Academic PatchTST Residual Correction Experiment", flush=True)
    print("=" * 72, flush=True)
    print(
        f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} weekly obs)",
        flush=True,
    )
    print(f"Device: {device}", flush=True)
    print(
        "Protocol: "
        f"h={config.horizon}, step={config.step_size}, "
        f"windows={len(cv_train_ends)}, holdout={config.final_holdout}, "
        f"input_size={config.input_size}, season_length={config.season_length}",
        flush=True,
    )
    print(f"PatchTST scaler_type: {config.patchtst.scaler_type}", flush=True)

    all_rows: list[dict[str, object]] = []
    all_pred_rows: list[dict[str, object]] = []

    for target_label, column_name in TARGETS.items():
        print(f"\n[{target_label}]", flush=True)
        target_series = df[column_name]

        for window_idx, train_end in enumerate(cv_train_ends, start=1):
            window_name = f"tscv_{window_idx:02d}"
            forecast_start = dates.iloc[train_end].date()
            forecast_end = dates.iloc[train_end + config.horizon - 1].date()
            started = time.time()
            print(
                f"  {window_name}: train_end={dates.iloc[train_end - 1].date()} "
                f"forecast={forecast_start}..{forecast_end}",
                flush=True,
            )

            rows, pred_rows = run_single_window(
                target_label=target_label,
                target_series=target_series,
                dates=dates,
                train_end=train_end,
                window_name=window_name,
                config=config,
                device=device,
            )
            elapsed = time.time() - started
            best_rmse = min(row["RMSE"] for row in rows)
            print(f"    done in {elapsed:.1f}s, best RMSE={best_rmse:.4f}", flush=True)
            all_rows.extend(rows)
            all_pred_rows.extend(pred_rows)

        if config.run_holdout:
            print(
                f"  holdout: train_end={dates.iloc[holdout_train_end - 1].date()}",
                flush=True,
            )
            started = time.time()
            rows, pred_rows = run_single_window(
                target_label=target_label,
                target_series=target_series,
                dates=dates,
                train_end=holdout_train_end,
                window_name="holdout",
                config=config,
                device=device,
            )
            elapsed = time.time() - started
            best_rmse = min(row["RMSE"] for row in rows)
            print(f"    done in {elapsed:.1f}s, best RMSE={best_rmse:.4f}", flush=True)
            all_rows.extend(rows)
            all_pred_rows.extend(pred_rows)

    results_df = pd.DataFrame(all_rows)
    pred_df = pd.DataFrame(all_pred_rows)
    window_summary_df = summarize_windows(results_df)
    tscv_summary_df = summarize_tscv(results_df)
    holdout_df = (
        results_df[results_df["window"] == "holdout"]
        .sort_values(["target", "RMSE", "residual_model"])
        .reset_index(drop=True)
    )
    tscv_leaderboard_df = make_leaderboard(
        tscv_summary_df.rename(
            columns={
                "mean_RMSE": "RMSE",
                "mean_MAE": "MAE",
                "mean_MAPE": "MAPE",
                "mean_NRMSE": "NRMSE",
            }
        ),
        ["RMSE", "MAE", "MAPE", "NRMSE"],
    )
    holdout_leaderboard_df = make_leaderboard(
        holdout_df,
        ["RMSE", "MAE", "MAPE", "NRMSE"],
    )

    results_df.to_csv(out_dir / "window_results.csv", index=False)
    pred_df.to_csv(out_dir / "forecast_predictions.csv", index=False)
    window_summary_df.to_csv(out_dir / "window_summary.csv", index=False)
    tscv_summary_df.to_csv(out_dir / "tscv_summary.csv", index=False)
    holdout_df.to_csv(out_dir / "holdout_results.csv", index=False)
    tscv_leaderboard_df.to_csv(out_dir / "tscv_leaderboard.csv", index=False)
    holdout_leaderboard_df.to_csv(out_dir / "holdout_leaderboard.csv", index=False)

    config_payload = asdict(config)
    config_payload["targets"] = TARGETS
    config_payload["device"] = str(device)
    config_payload["nrmse_definition"] = "RMSE / abs(mean(y_true)) * 100"
    config_payload["data_start"] = dates.iloc[0].strftime("%Y-%m-%d")
    config_payload["data_end"] = dates.iloc[-1].strftime("%Y-%m-%d")
    config_payload["initial_train_end"] = dates.iloc[cv_train_ends[0] - 1].strftime("%Y-%m-%d")
    config_payload["tscv_eval_start"] = dates.iloc[cv_train_ends[0]].strftime("%Y-%m-%d")
    config_payload["tscv_eval_end"] = dates.iloc[cv_train_ends[-1] + config.horizon - 1].strftime("%Y-%m-%d")
    config_payload["holdout_start"] = dates.iloc[holdout_train_end].strftime("%Y-%m-%d")
    config_payload["holdout_end"] = dates.iloc[-1].strftime("%Y-%m-%d")
    with (out_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config_payload, fp, ensure_ascii=False, indent=2)

    print("\nSaved:", flush=True)
    print(f"  - {out_dir / 'window_results.csv'}", flush=True)
    print(f"  - {out_dir / 'forecast_predictions.csv'}", flush=True)
    print(f"  - {out_dir / 'window_summary.csv'}", flush=True)
    print(f"  - {out_dir / 'tscv_summary.csv'}", flush=True)
    print(f"  - {out_dir / 'tscv_leaderboard.csv'}", flush=True)
    if config.run_holdout:
        print(f"  - {out_dir / 'holdout_results.csv'}", flush=True)
        print(f"  - {out_dir / 'holdout_leaderboard.csv'}", flush=True)
    print(f"  - {out_dir / 'config.json'}", flush=True)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Academic-style PatchTST baseline with residual correction models."
    )
    parser.add_argument("--data-path", default="data_weekly_260120.csv")
    parser.add_argument("--output-dir", default="results/academic_patchtst_residuals")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--patch-max-steps", type=int, default=5000)
    parser.add_argument("--resid-max-steps", type=int, default=2000)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed,
        max_windows=args.max_windows,
        run_holdout=not args.skip_holdout,
    )
    cfg.patchtst.max_steps = args.patch_max_steps
    cfg.residual_nlinear.max_steps = args.resid_max_steps
    return cfg


if __name__ == "__main__":
    run_experiment(parse_args())
