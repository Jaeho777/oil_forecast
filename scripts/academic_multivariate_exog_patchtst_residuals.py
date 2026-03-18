#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

os.makedirs("/tmp/matplotlib-cache", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor


TARGETS = {
    "WTI Oil": "Com_CrudeOil",
    "Brent Oil": "Com_BrentCrudeOil",
}

BASE_FEATURES = [
    "Com_Gasoline",
    "Com_NaturalGas",
    "Com_Uranium",
    "Com_Coal",
    "Com_LME_Cu_Cash",
    "Com_Steel",
    "Com_Iron_Ore",
    "Idx_DxyUSD",
    "EX_USD_CNY",
    "Bonds_US_10Y",
    "Bonds_US_2Y",
    "Bonds_US_3M",
    "Idx_SnPVIX",
    "Com_Gold",
    "Idx_SnP500",
    "Idx_CSI300",
    "EX_USD_KRW",
    "Bonds_KOR_10Y",
    "EX_USD_JPY",
    "Com_Corn",
    "Com_Soybeans",
    "Com_PalmOil",
]


@dataclass
class PatchTSTConfig:
    hidden_size: int = 64
    attention_heads: int = 4
    linear_hidden_size: int = 256
    patch_len: int = 4
    stride: int = 2
    dropout: float = 0.2
    encoder_layers: int = 2
    attn_dropout: float = 0.0
    fc_dropout: float = 0.2
    epochs: int = 60
    patience: int = 12
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    scaler_type: str = "robust"


@dataclass
class ResidualNLinearConfig:
    hidden_size: int = 64
    epochs: int = 60
    patience: int = 12
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-5


@dataclass
class TreeConfig:
    n_estimators: int = 300
    learning_rate: float = 0.03


@dataclass
class ExperimentConfig:
    data_path: str = "data_weekly_260120.csv"
    output_dir: str = "results/academic_multivariate_exog_patchtst_residuals"
    input_size: int = 24
    horizon: int = 12
    step_size: int = 4
    number_of_windows: int = 24
    final_holdout: int = 12
    seed: int = 42
    max_windows: int | None = None
    run_holdout: bool = True
    feature_candidates: tuple[int, ...] = (10, 15, 20, 25, 30, 40, 55)
    shap_estimators: int = 300
    shap_learning_rate: float = 0.05
    shap_num_leaves: int = 31
    cv_estimators: int = 200
    cv_learning_rate: float = 0.05
    cv_num_leaves: int = 20
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.clip(np.abs(y_true), a_min=1e-8, a_max=None)
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0),
        "NRMSE": float(rmse(y_true, y_pred) / np.mean(y_true) * 100.0),
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


def build_engineered_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    derived = pd.DataFrame(index=df.index)
    for col in BASE_FEATURES:
        derived[f"{col}_ret"] = df[col].pct_change()

    derived["Spread_10Y_2Y"] = df["Bonds_US_10Y"] - df["Bonds_US_2Y"]
    derived["Spread_Crack"] = df["Com_Gasoline"] - df[target_col]
    derived["Ratio_Gold_Oil"] = df["Com_Gold"] / df[target_col]

    for col in ["Com_Gasoline", "Com_NaturalGas", "Idx_SnPVIX", "Idx_DxyUSD"]:
        derived[f"{col}_ma4r"] = df[col] / df[col].rolling(4).mean() - 1.0
        derived[f"{col}_ma12r"] = df[col] / df[col].rolling(12).mean() - 1.0

    return derived


def build_feature_frame(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    derived = build_engineered_features(df, target_col)
    all_features = BASE_FEATURES + list(derived.columns)
    feature_df = pd.concat([df[BASE_FEATURES], derived], axis=1).shift(1)
    return feature_df, all_features


def select_features(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    config: ExperimentConfig,
    seed_offset: int = 0,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    lgb_shap = lgb.LGBMRegressor(
        n_estimators=config.shap_estimators,
        learning_rate=config.shap_learning_rate,
        num_leaves=config.shap_num_leaves,
        verbose=-1,
        random_state=config.seed + seed_offset,
    )
    lgb_shap.fit(X_train_raw, y_train)
    explainer = shap.TreeExplainer(lgb_shap)
    shap_values = explainer.shap_values(X_train_raw)
    mean_shap = np.abs(shap_values).mean(axis=0)

    shap_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    ranked = shap_df["feature"].tolist()
    tscv = TimeSeriesSplit(n_splits=5)
    cv_rows = []

    for n_features in config.feature_candidates:
        n_features = min(n_features, len(feature_names))
        idx = [feature_names.index(f) for f in ranked[:n_features]]
        X_slice = X_train_raw[:, idx]
        fold_rmses = []

        for train_idx, val_idx in tscv.split(X_slice):
            fold_model = lgb.LGBMRegressor(
                n_estimators=config.cv_estimators,
                learning_rate=config.cv_learning_rate,
                num_leaves=config.cv_num_leaves,
                verbose=-1,
                random_state=config.seed + seed_offset,
            )
            fold_model.fit(X_slice[train_idx], y_train[train_idx])
            fold_pred = fold_model.predict(X_slice[val_idx])
            fold_rmses.append(rmse(y_train[val_idx], fold_pred))

        cv_rows.append({"n_features": n_features, "cv_rmse": float(np.mean(fold_rmses))})

    cv_df = pd.DataFrame(cv_rows)
    best_n = int(cv_df.loc[cv_df["cv_rmse"].idxmin(), "n_features"])
    return ranked[:best_n], shap_df, cv_df


def make_seq_to_one_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for idx in range(seq_len, len(X)):
        Xs.append(X[idx - seq_len : idx])
        ys.append(y[idx])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def make_eval_sequences(X_full: np.ndarray, start_idx: int, end_idx: int, seq_len: int) -> np.ndarray:
    seqs = []
    for idx in range(start_idx, end_idx):
        seqs.append(X_full[idx - seq_len : idx])
    return np.asarray(seqs, dtype=np.float32)


class PatchTSTOneStep(nn.Module):
    def __init__(self, seq_len: int, n_features: int, config: PatchTSTConfig) -> None:
        super().__init__()
        self.patch_len = config.patch_len
        self.stride = config.stride
        if seq_len < self.patch_len:
            raise ValueError("seq_len must be >= patch_len.")

        n_patches = (seq_len - self.patch_len) // self.stride + 1
        self.proj = nn.Linear(self.patch_len * n_features, config.hidden_size)
        self.cls = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, config.hidden_size) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.linear_hidden_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.encoder_layers)
        self.head = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        patches = []
        for start in range(0, x.size(1) - self.patch_len + 1, self.stride):
            patches.append(x[:, start : start + self.patch_len, :].reshape(batch, -1))
        z = self.proj(torch.stack(patches, dim=1))
        z = torch.cat([self.cls.expand(batch, -1, -1), z], dim=1) + self.pos
        z = self.encoder(z)
        return self.head(self.dropout(z[:, 0])).squeeze(-1)


class NLinearResidualExog(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden_size: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.time_proj = nn.Linear(seq_len, hidden_size)
        self.feature_proj = nn.Linear(n_features, hidden_size)
        self.output = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_channel = x[:, :, 0]
        time_channel = time_channel - time_channel[:, -1:]
        h1 = self.act(self.time_proj(time_channel))
        h2 = self.act(self.feature_proj(x[:, -1, :]))
        return self.output(self.dropout(torch.cat([h1, h2], dim=-1))).squeeze(-1)


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    holdout_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(X) <= holdout_size:
        raise ValueError("Not enough sequence samples for inner validation split.")
    split_at = len(X) - holdout_size
    return X[:split_at], y[:split_at], X[split_at:], y[split_at:]


def train_torch_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> nn.Module:
    Xtr, ytr, Xva, yva = split_train_val(X, y, holdout_size=max(12, len(X) // 10))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=min(batch_size, len(Xtr)),
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
        batch_size=min(batch_size, len(Xva)),
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    wait = 0
    set_seed(seed)
    model.to(device)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def predict_torch_model(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            xb = torch.from_numpy(X[start : start + 256]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def train_tree_residual_model(
    model_name: str,
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    config: ExperimentConfig,
    seed: int,
):
    X_flat = X_train_seq.reshape(len(X_train_seq), -1)
    if model_name == "XGB":
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=config.xgb.n_estimators,
            learning_rate=config.xgb.learning_rate,
            max_depth=4,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    else:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=config.lgbm.n_estimators,
            learning_rate=config.lgbm.learning_rate,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=seed,
            verbose=-1,
            n_jobs=1,
        )
    model.fit(X_flat, y_train)
    return model


def predict_tree_residual_model(model, X_seq: np.ndarray) -> np.ndarray:
    return model.predict(X_seq.reshape(len(X_seq), -1))


def window_result_row(
    target_label: str,
    window_name: str,
    train_end: int,
    forecast_dates: pd.Series,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
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


def summarize_windows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["target", "window", "model", "base_model", "residual_model"], as_index=False)[
            ["RMSE", "MAE", "MAPE", "NRMSE"]
        ]
        .mean()
        .sort_values(["target", "window", "RMSE"])
        .reset_index(drop=True)
    )


def summarize_tscv(df: pd.DataFrame) -> pd.DataFrame:
    tscv_rows = df[df["window"].str.startswith("tscv_")]
    return (
        tscv_rows.groupby(["target", "model", "base_model", "residual_model"], as_index=False)[
            ["RMSE", "MAE", "MAPE", "NRMSE"]
        ]
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


def make_leaderboard(df: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    return (
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


def load_dataframe(config: ExperimentConfig) -> pd.DataFrame:
    df = pd.read_csv(config.data_path, parse_dates=["dt"]).sort_values("dt").reset_index(drop=True)
    required = ["dt"] + list(TARGETS.values()) + BASE_FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    if len(df) != 668:
        raise ValueError(f"Expected 668 weekly observations, found {len(df)}.")
    return df


def run_single_window(
    df: pd.DataFrame,
    target_label: str,
    target_col: str,
    train_end: int,
    window_name: str,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    feature_df, feature_names = build_feature_frame(df, target_col)
    target_series = df[target_col]
    eval_end = train_end + config.horizon

    X_train_raw = np.nan_to_num(feature_df.iloc[:train_end].to_numpy(dtype=np.float32), nan=0.0)
    y_train = target_series.iloc[:train_end].to_numpy(dtype=np.float32)
    y_eval = target_series.iloc[train_end:eval_end].to_numpy(dtype=np.float64)
    forecast_dates = df["dt"].iloc[train_end:eval_end].reset_index(drop=True)

    selected_features, shap_df, cv_df = select_features(
        X_train_raw,
        y_train.astype(float),
        feature_names,
        config,
        seed_offset=train_end,
    )
    selected_idx = [feature_names.index(f) for f in selected_features]

    X_window_raw = np.nan_to_num(
        feature_df.iloc[:eval_end, selected_idx].to_numpy(dtype=np.float32),
        nan=0.0,
    )

    scaler_x = RobustScaler()
    scaler_x.fit(X_window_raw[:train_end])
    X_window_sc = scaler_x.transform(X_window_raw).astype(np.float32)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std == 0:
        y_std = 1.0
    y_train_n = ((y_train - y_mean) / y_std).astype(np.float32)

    Xtr_seq, ytr_seq = make_seq_to_one_sequences(
        X_window_sc[:train_end],
        y_train_n,
        seq_len=config.input_size,
    )
    Xeval_seq = make_eval_sequences(
        X_window_sc,
        start_idx=train_end,
        end_idx=eval_end,
        seq_len=config.input_size,
    )

    baseline_model = PatchTSTOneStep(
        seq_len=config.input_size,
        n_features=Xtr_seq.shape[2],
        config=config.patchtst,
    )
    baseline_model = train_torch_model(
        model=baseline_model,
        X=Xtr_seq,
        y=ytr_seq,
        epochs=config.patchtst.epochs,
        patience=config.patchtst.patience,
        batch_size=config.patchtst.batch_size,
        learning_rate=config.patchtst.learning_rate,
        weight_decay=config.patchtst.weight_decay,
        seed=config.seed + train_end,
        device=device,
    )

    base_train_pred_n = predict_torch_model(baseline_model, Xtr_seq, device)
    base_eval_pred_n = predict_torch_model(baseline_model, Xeval_seq, device)
    base_eval_pred = base_eval_pred_n * y_std + y_mean

    rows = [
        window_result_row(
            target_label,
            window_name,
            train_end,
            forecast_dates,
            "PatchTST",
            y_eval,
            base_eval_pred,
        )
    ]
    pred_rows = prediction_rows(
        target_label,
        window_name,
        "PatchTST",
        forecast_dates,
        y_eval,
        base_eval_pred,
    )

    resid_train_target_n = (ytr_seq - base_train_pred_n).astype(np.float32)

    nlinear_model = NLinearResidualExog(
        seq_len=config.input_size,
        n_features=Xtr_seq.shape[2],
        hidden_size=config.residual_nlinear.hidden_size,
    )
    nlinear_model = train_torch_model(
        model=nlinear_model,
        X=Xtr_seq,
        y=resid_train_target_n,
        epochs=config.residual_nlinear.epochs,
        patience=config.residual_nlinear.patience,
        batch_size=config.residual_nlinear.batch_size,
        learning_rate=config.residual_nlinear.learning_rate,
        weight_decay=config.residual_nlinear.weight_decay,
        seed=config.seed + 1000 + train_end,
        device=device,
    )
    resid_nlinear_n = predict_torch_model(nlinear_model, Xeval_seq, device)

    xgb_model = train_tree_residual_model(
        "XGB",
        Xtr_seq,
        resid_train_target_n,
        config,
        seed=config.seed + 2000 + train_end,
    )
    resid_xgb_n = predict_tree_residual_model(xgb_model, Xeval_seq)

    lgbm_model = train_tree_residual_model(
        "LGBM",
        Xtr_seq,
        resid_train_target_n,
        config,
        seed=config.seed + 3000 + train_end,
    )
    resid_lgbm_n = predict_tree_residual_model(lgbm_model, Xeval_seq)

    corrections = {
        "PatchTST+NLinear": (base_eval_pred_n + resid_nlinear_n) * y_std + y_mean,
        "PatchTST+XGB": (base_eval_pred_n + resid_xgb_n) * y_std + y_mean,
        "PatchTST+LGBM": (base_eval_pred_n + resid_lgbm_n) * y_std + y_mean,
    }
    for model_name, forecast in corrections.items():
        rows.append(
            window_result_row(
                target_label,
                window_name,
                train_end,
                forecast_dates,
                model_name,
                y_eval,
                forecast,
            )
        )
        pred_rows.extend(
            prediction_rows(
                target_label,
                window_name,
                model_name,
                forecast_dates,
                y_eval,
                forecast,
            )
        )

    feature_rows = []
    for _, row in shap_df.head(len(selected_features)).iterrows():
        feature_rows.append(
            {
                "target": target_label,
                "window": window_name,
                "train_size": train_end,
                "selected": row["feature"] in selected_features,
                "feature": row["feature"],
                "mean_abs_shap": float(row["mean_abs_shap"]),
            }
        )
    for _, row in cv_df.iterrows():
        feature_rows.append(
            {
                "target": target_label,
                "window": window_name,
                "train_size": train_end,
                "selected": False,
                "feature": f"n_features={int(row['n_features'])}",
                "mean_abs_shap": float(row["cv_rmse"]),
            }
        )

    return rows, pred_rows, feature_rows


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
    print("Academic Multivariate Exogenous PatchTST Residual Experiment", flush=True)
    print("=" * 72, flush=True)
    print(
        f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} weekly obs)",
        flush=True,
    )
    print(f"Device: {device}", flush=True)
    print(
        "Protocol: "
        f"h={config.horizon}, step={config.step_size}, windows={len(cv_train_ends)}, "
        f"holdout={config.final_holdout}, input_size={config.input_size}",
        flush=True,
    )

    all_rows: list[dict[str, object]] = []
    all_pred_rows: list[dict[str, object]] = []
    all_feature_rows: list[dict[str, object]] = []

    for target_label, target_col in TARGETS.items():
        print(f"\n[{target_label}]", flush=True)
        for idx, train_end in enumerate(cv_train_ends, start=1):
            window_name = f"tscv_{idx:02d}"
            print(
                f"  {window_name}: train_end={dates.iloc[train_end - 1].date()} "
                f"forecast={dates.iloc[train_end].date()}..{dates.iloc[train_end + config.horizon - 1].date()}",
                flush=True,
            )
            started = time.time()
            rows, pred_rows, feature_rows = run_single_window(
                df=df,
                target_label=target_label,
                target_col=target_col,
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
            all_feature_rows.extend(feature_rows)

        if config.run_holdout:
            print(
                f"  holdout: train_end={dates.iloc[holdout_train_end - 1].date()}",
                flush=True,
            )
            started = time.time()
            rows, pred_rows, feature_rows = run_single_window(
                df=df,
                target_label=target_label,
                target_col=target_col,
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
            all_feature_rows.extend(feature_rows)

    results_df = pd.DataFrame(all_rows)
    pred_df = pd.DataFrame(all_pred_rows)
    feature_df = pd.DataFrame(all_feature_rows)
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
    feature_df.to_csv(out_dir / "feature_selection_log.csv", index=False)
    window_summary_df.to_csv(out_dir / "window_summary.csv", index=False)
    tscv_summary_df.to_csv(out_dir / "tscv_summary.csv", index=False)
    holdout_df.to_csv(out_dir / "holdout_results.csv", index=False)
    tscv_leaderboard_df.to_csv(out_dir / "tscv_leaderboard.csv", index=False)
    holdout_leaderboard_df.to_csv(out_dir / "holdout_leaderboard.csv", index=False)

    payload = asdict(config)
    payload["targets"] = TARGETS
    payload["device"] = str(device)
    with (out_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print("\nSaved:", flush=True)
    print(f"  - {out_dir / 'window_results.csv'}", flush=True)
    print(f"  - {out_dir / 'forecast_predictions.csv'}", flush=True)
    print(f"  - {out_dir / 'feature_selection_log.csv'}", flush=True)
    print(f"  - {out_dir / 'window_summary.csv'}", flush=True)
    print(f"  - {out_dir / 'tscv_summary.csv'}", flush=True)
    print(f"  - {out_dir / 'tscv_leaderboard.csv'}", flush=True)
    if config.run_holdout:
        print(f"  - {out_dir / 'holdout_results.csv'}", flush=True)
        print(f"  - {out_dir / 'holdout_leaderboard.csv'}", flush=True)
    print(f"  - {out_dir / 'config.json'}", flush=True)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Academic multivariate/exogenous PatchTST residual experiment."
    )
    parser.add_argument("--data-path", default="data_weekly_260120.csv")
    parser.add_argument(
        "--output-dir",
        default="results/academic_multivariate_exog_patchtst_residuals",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--patch-epochs", type=int, default=60)
    parser.add_argument("--resid-epochs", type=int, default=60)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed,
        max_windows=args.max_windows,
        run_holdout=not args.skip_holdout,
    )
    cfg.patchtst.epochs = args.patch_epochs
    cfg.residual_nlinear.epochs = args.resid_epochs
    return cfg


if __name__ == "__main__":
    run_experiment(parse_args())
