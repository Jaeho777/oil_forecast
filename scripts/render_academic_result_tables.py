#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.makedirs("/tmp/matplotlib-cache", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = Path("results/academic_patchtst_residuals")
MODEL_ORDER = ["PatchTST", "PatchTST+NLinear", "PatchTST+XGB", "PatchTST+LGBM"]
RESIDUAL_ORDER = ["-", "NLinear", "XGB", "LGBM"]


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def model_label(model_name: str) -> str:
    return model_name.replace("+", " + ")


def render_table_png(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    fontsize: int = 10,
) -> None:
    rows, cols = df.shape
    fig_width = max(8.0, cols * 1.7)
    fig_height = max(1.8, rows * 0.52 + 1.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.35)

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")

    ax.set_title(title, fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary_table(
    source_df: pd.DataFrame,
    metric_col: str,
    value_prefix: str,
    best_label_col: str,
) -> pd.DataFrame:
    rows = []
    for target in ["WTI Oil", "Brent Oil"]:
        sub = source_df[source_df["target"] == target].copy()
        metrics = {}
        for residual in RESIDUAL_ORDER:
            row = sub[sub["residual_model"] == residual].iloc[0]
            metrics[residual] = row[metric_col]

        best_row = sub.sort_values(metric_col).iloc[0]
        bench = metrics["-"]
        best_value = best_row[metric_col]
        rows.append(
            {
                "Target": target,
                f"{value_prefix} PatchTST (%)": fmt3(metrics["-"]),
                "Residual-NLinear (%)": fmt3(metrics["NLinear"]),
                "Residual-XGB (%)": fmt3(metrics["XGB"]),
                "Residual-LGBM (%)": fmt3(metrics["LGBM"]),
                best_label_col: model_label(best_row["model"]),
                "Delta vs Benchmark (%)": fmt3(best_value - bench),
            }
        )
    return pd.DataFrame(rows)


def build_holdout_metrics_table(holdout_df: pd.DataFrame, target: str) -> pd.DataFrame:
    sub = holdout_df[holdout_df["target"] == target].copy()
    order_map = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    sub["order"] = sub["model"].map(order_map)
    sub = sub.sort_values("order")

    rows = []
    for _, row in sub.iterrows():
        rows.append(
            {
                "Model Type": "Bench-mark" if row["residual_model"] == "-" else "Residual Correction",
                "Base Model": row["base_model"],
                "Residual Model": row["residual_model"],
                "RMSE": fmt3(row["RMSE"]),
                "MAE": fmt3(row["MAE"]),
                "MAPE (%)": fmt3(row["MAPE"]),
                "NRMSE": fmt3(row["NRMSE"]),
            }
        )
    return pd.DataFrame(rows)


def format_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"Base Model": "Baseline Model"})
    for col in ["RMSE", "MAE", "MAPE", "NRMSE"]:
        out[col] = out[col].map(fmt3)
    return out


def main() -> None:
    tscv_summary = pd.read_csv(RESULT_DIR / "tscv_summary.csv")
    holdout_results = pd.read_csv(RESULT_DIR / "holdout_results.csv")
    tscv_leaderboard = pd.read_csv(RESULT_DIR / "tscv_leaderboard.csv")
    holdout_leaderboard = pd.read_csv(RESULT_DIR / "holdout_leaderboard.csv")

    summary_tscv = build_summary_table(
        source_df=tscv_summary,
        metric_col="mean_MAPE",
        value_prefix="Bench-mark:",
        best_label_col="ts-cv Selected Model",
    )
    summary_holdout = build_summary_table(
        source_df=holdout_results,
        metric_col="MAPE",
        value_prefix="Bench-mark:",
        best_label_col="Holdout Best Model",
    )
    holdout_wti = build_holdout_metrics_table(holdout_results, "WTI Oil")
    holdout_brent = build_holdout_metrics_table(holdout_results, "Brent Oil")

    render_table_png(
        summary_tscv,
        "Summary Table: ts-cv Average MAPE",
        RESULT_DIR / "summary_tscv_mape_table.png",
        fontsize=10,
    )
    render_table_png(
        summary_holdout,
        "Summary Table: Holdout MAPE",
        RESULT_DIR / "summary_holdout_mape_table.png",
        fontsize=10,
    )
    render_table_png(
        holdout_wti,
        "WTI Oil Holdout Metrics",
        RESULT_DIR / "holdout_metrics_wti_table.png",
        fontsize=10,
    )
    render_table_png(
        holdout_brent,
        "Brent Oil Holdout Metrics",
        RESULT_DIR / "holdout_metrics_brent_table.png",
        fontsize=10,
    )
    render_table_png(
        format_leaderboard(tscv_leaderboard),
        "ts-cv Leaderboard Across PatchTST Residual Variants",
        RESULT_DIR / "tscv_leaderboard.png",
        fontsize=10,
    )
    render_table_png(
        format_leaderboard(holdout_leaderboard),
        "Holdout Leaderboard Across PatchTST Residual Variants",
        RESULT_DIR / "holdout_leaderboard.png",
        fontsize=10,
    )

    print("Saved table visuals:")
    print(RESULT_DIR / "summary_tscv_mape_table.png")
    print(RESULT_DIR / "summary_holdout_mape_table.png")
    print(RESULT_DIR / "holdout_metrics_wti_table.png")
    print(RESULT_DIR / "holdout_metrics_brent_table.png")
    print(RESULT_DIR / "tscv_leaderboard.png")
    print(RESULT_DIR / "holdout_leaderboard.png")


if __name__ == "__main__":
    main()
