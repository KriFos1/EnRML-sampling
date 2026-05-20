#!/usr/bin/env python3
"""
Plot EnRML prior/posterior prediction ensembles and parameter variance.

Requires EnRML/run_script.py output:
    results_enrml/prior_forecast.npz
    results_enrml/posterior_forecast.npz
    results_enrml/posterior_state_estimate.npz
    PRIOR/log_prior.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ENRML_DIR = Path(__file__).resolve().parent
DATA_DIR = ENRML_DIR / "DATA"
DEFAULT_RESULTS_PATH = ENRML_DIR / "results_enrml"
DEFAULT_PRIOR_STATE = ENRML_DIR / "PRIOR/log_prior.npz"
DEFAULT_CONFIG = ENRML_DIR / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EnRML prediction and variance comparisons.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--out-dir", type=Path, default=ENRML_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--prior-forecast", type=Path, default=None)
    parser.add_argument("--posterior-forecast", type=Path, default=None)
    parser.add_argument("--prior-state", type=Path, default=DEFAULT_PRIOR_STATE)
    parser.add_argument("--posterior-state", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state-name", default="log_permx")
    return parser.parse_args()


def samples_to_3d(samples: np.ndarray, n_t: int, n_cols: int) -> np.ndarray:
    expected = n_t * n_cols
    if samples.ndim != 2 or samples.shape[1] != expected:
        raise ValueError(f"Expected samples with shape (n_members, {expected}), got {samples.shape}")
    return np.stack(
        [sample.reshape((n_t, n_cols), order="F") for sample in samples],
        axis=0,
    )


def build_cd_diag(data_df: pd.DataFrame, var_df: pd.DataFrame) -> np.ndarray:
    n_t = len(data_df.index)
    n_cols = len(data_df.columns)
    cd_diag = np.zeros(n_t * n_cols)
    idx = 0
    for col in data_df.columns:
        for row in data_df.index:
            entry = var_df.loc[row, col]
            cd_diag[idx] = entry[1] if isinstance(entry, list) else float(entry)
            idx += 1
    return cd_diag


def build_cd_std(data_df: pd.DataFrame, var_df: pd.DataFrame) -> np.ndarray:
    n_t = len(data_df.index)
    n_cols = len(data_df.columns)
    return np.sqrt(build_cd_diag(data_df, var_df)).reshape((n_t, n_cols), order="F")


def forecast_to_samples(forecast_file: Path, data_df: pd.DataFrame) -> np.ndarray:
    with np.load(forecast_file, allow_pickle=True) as forecast:
        pred_data = forecast["pred_data"]

    n_t = len(data_df.index)
    n_cols = len(data_df.columns)
    if len(pred_data) != n_t:
        raise ValueError(f"{forecast_file} has {len(pred_data)} report entries, expected {n_t}")

    first_col = data_df.columns[0]
    first_values = np.asarray(pred_data[0][first_col], dtype=float).reshape(-1)
    n_members = first_values.size
    samples = np.empty((n_members, n_t * n_cols), dtype=float)

    for time_idx, step_data in enumerate(pred_data):
        missing = [col for col in data_df.columns if col not in step_data]
        if missing:
            raise KeyError(f"{forecast_file} is missing prediction columns: {missing}")

        for col_idx, col in enumerate(data_df.columns):
            values = np.asarray(step_data[col], dtype=float).reshape(-1)
            if values.size != n_members:
                raise ValueError(
                    f"{forecast_file}:{col} at report {time_idx} has {values.size} "
                    f"members, expected {n_members}"
                )
            samples[:, col_idx * n_t + time_idx] = values

    return samples


def state_param_count(state_file: Path, state_name: str) -> int:
    with np.load(state_file, allow_pickle=True) as state:
        values = np.asarray(state[state_name])
    if values.ndim != 2:
        raise ValueError(f"{state_file}:{state_name} must be a 2-D ensemble array")
    return max(values.shape)


def infer_grid_shape(n_params: int) -> tuple[int, ...]:
    side = int(np.sqrt(n_params))
    if side * side == n_params:
        return (side, side)
    return (n_params, 1)


def load_grid_shape(config_file: Path, state_name: str, n_params: int) -> tuple[int, ...]:
    try:
        import yaml

        with config_file.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        grid_shape = tuple(int(v) for v in config["ensemble"][f"prior_{state_name}"]["grid"])
    except Exception:
        return infer_grid_shape(n_params)

    if int(np.prod(grid_shape)) != n_params:
        raise ValueError(
            f"Grid shape {grid_shape} from {config_file} has {int(np.prod(grid_shape))} cells, "
            f"but {state_name} has {n_params} parameters"
        )
    return grid_shape


def load_state_samples(state_file: Path, state_name: str, n_params: int) -> np.ndarray:
    with np.load(state_file, allow_pickle=True) as state:
        values = np.asarray(state[state_name], dtype=float)

    if values.ndim != 2:
        raise ValueError(f"{state_file}:{state_name} must be a 2-D ensemble array")
    if values.shape[0] == n_params:
        return values.T
    if values.shape[1] == n_params:
        return values
    raise ValueError(
        f"{state_file}:{state_name} shape {values.shape} does not match {n_params} grid parameters"
    )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data_df = pd.read_pickle(args.data_dir / "true_data.pkl")
    var_df = pd.read_pickle(args.data_dir / "true_data_var.pkl")
    n_t = len(data_df.index)
    n_cols = len(data_df.columns)

    results_dir = args.results
    prior_forecast_file = args.prior_forecast or results_dir / "prior_forecast.npz"
    posterior_forecast_file = args.posterior_forecast or results_dir / "posterior_forecast.npz"
    posterior_state_file = args.posterior_state or results_dir / "posterior_state_estimate.npz"

    print(f"Loading prior forecast from {prior_forecast_file}")
    d_pred_prior_samples = forecast_to_samples(prior_forecast_file, data_df)
    print(f"Loading posterior forecast from {posterior_forecast_file}")
    d_pred_post_samples = forecast_to_samples(posterior_forecast_file, data_df)
    d_obs = data_df.to_numpy(dtype=float).reshape(n_t * n_cols, order="F")
    cd_diag = build_cd_diag(data_df, var_df)

    if d_pred_prior_samples.shape[0] == 0 or d_pred_post_samples.shape[0] == 0:
        raise RuntimeError("No prediction ensembles found in the EnRML forecast files.")

    data_obs_df = pd.DataFrame(
        d_obs.reshape((n_t, n_cols), order="F"),
        index=data_df.index,
        columns=data_df.columns,
    )
    prior_samples_3d = samples_to_3d(d_pred_prior_samples, n_t, n_cols)
    post_samples_3d = samples_to_3d(d_pred_post_samples, n_t, n_cols)
    cd_std_plot = build_cd_std(data_df, var_df)

    print("Creating prediction comparison plot...")
    rate_types = ["WOPR", "WGPR", "WWPR"]
    rate_labels = [
        "Oil Production Rate (STB/D)",
        "Gas Production Rate (MSCF/D)",
        "Water Production Rate (STB/D)",
    ]
    well_rows = ["P1", "P2", "P3", "P4", "TOTAL"]

    fig, axes = plt.subplots(len(well_rows), len(rate_types), figsize=(18, 16), sharex=True)

    for row_idx, well in enumerate(well_rows):
        for col_idx, (rate_type, rate_label) in enumerate(zip(rate_types, rate_labels)):
            ax = axes[row_idx, col_idx]

            if well == "TOTAL":
                type_cols = [c for c in data_df.columns if c.startswith(rate_type + ":")]
                col_indices = [data_df.columns.get_loc(c) for c in type_cols]
                obs_series = data_obs_df[type_cols].sum(axis=1)
                prior_env = prior_samples_3d[:, :, col_indices].sum(axis=2)
                post_env = post_samples_3d[:, :, col_indices].sum(axis=2)
                obs_std = np.sqrt((cd_std_plot[:, col_indices] ** 2).sum(axis=1))
                row_label = "Total"
            else:
                col = f"{rate_type}:{well}"
                col_idx_data = data_df.columns.get_loc(col)
                obs_series = data_obs_df[col]
                prior_env = prior_samples_3d[:, :, col_idx_data]
                post_env = post_samples_3d[:, :, col_idx_data]
                obs_std = cd_std_plot[:, col_idx_data]
                row_label = well

            dates = data_obs_df.index

            prior_lo = prior_env.min(axis=0)
            prior_hi = prior_env.max(axis=0)
            ax.fill_between(
                dates,
                prior_lo,
                prior_hi,
                alpha=0.16,
                color="#ff7f0e",
                label="Prior ensemble",
            )
            ax.plot(
                dates,
                prior_env.mean(axis=0),
                "--",
                color="#ff7f0e",
                linewidth=1.5,
                alpha=0.9,
                label="Prior mean",
            )

            post_lo = post_env.min(axis=0)
            post_hi = post_env.max(axis=0)
            ax.fill_between(
                dates,
                post_lo,
                post_hi,
                alpha=0.24,
                color="#2ca02c",
                label="EnRML posterior",
            )
            ax.plot(
                dates,
                post_env.mean(axis=0),
                "-",
                color="#2ca02c",
                linewidth=1.6,
                alpha=0.9,
                label="EnRML mean",
            )

            ax.errorbar(
                dates,
                obs_series,
                yerr=2.0 * obs_std,
                fmt="o",
                color="#1f77b4",
                markersize=4,
                capsize=3,
                elinewidth=1.0,
                label="Data +/- 2 sigma",
            )

            if row_idx == 0:
                ax.set_title(rate_label, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold")
            if row_idx == len(well_rows) - 1:
                ax.set_xlabel("Date", fontsize=10)

            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, linestyle="--")
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=9, loc="best", framealpha=0.95)

    plt.suptitle(
        "EnRML prediction comparison: prior vs posterior",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    pred_path = out_dir / "predictions_comparison.png"
    plt.savefig(pred_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {pred_path}")
    plt.close(fig)

    print("\n" + "=" * 70)
    print("PREDICTION STATISTICS")
    print("=" * 70)

    cd_inv = 1.0 / cd_diag
    method_preds = {
        "Prior (mean)": d_pred_prior_samples.mean(axis=0),
        "EnRML (mean)": d_pred_post_samples.mean(axis=0),
    }

    for rate_type, rate_label in zip(rate_types, rate_labels):
        cols = [c for c in data_df.columns if c.startswith(rate_type)]
        col_mask = np.zeros(n_cols, dtype=bool)
        for c in cols:
            col_mask[data_df.columns.get_loc(c)] = True
        flat_mask_r = np.repeat(col_mask, n_t)
        cd_inv_sub = cd_inv[flat_mask_r]
        obs_vals = d_obs[flat_mask_r]
        n_sub = int(flat_mask_r.sum())

        print(f"\n{rate_label} (n_data={n_sub}):")
        for name, pred in method_preds.items():
            r = pred[flat_mask_r] - obs_vals
            misfit = float(r @ (cd_inv_sub * r))
            print(f"  {name:16s}  misfit = {misfit:.6e}  (per-datum = {misfit/n_sub:.4e})")

    print("Creating parameter samples figure...")
    n_params = state_param_count(args.prior_state, args.state_name)
    grid_shape = load_grid_shape(args.config, args.state_name, n_params)
    mask_arr = np.ones(grid_shape, dtype=bool)
    flat_mask = mask_arr.ravel(order="F")
    n_active = int(flat_mask.sum())

    print(f"Loading prior state from {args.prior_state}")
    x_prior = load_state_samples(args.prior_state, args.state_name, n_active)
    print(f"Loading posterior state from {posterior_state_file}")
    x_enrml = load_state_samples(posterior_state_file, args.state_name, n_active)

    def vec_to_grid(vec: np.ndarray) -> np.ndarray:
        grid = np.full(flat_mask.shape, np.nan)
        grid[flat_mask] = vec
        return grid.reshape(grid_shape, order="F").squeeze()

    n_show = min(10, x_prior.shape[0], x_enrml.shape[0])
    if n_show == 0:
        raise RuntimeError("No parameter samples available for plotting")

    prior_vecs = x_prior[:n_show]
    enrml_vecs = x_enrml[:n_show]
    all_grids = np.stack([vec_to_grid(v) for v in np.vstack([prior_vecs, enrml_vecs])])
    finite = all_grids[np.isfinite(all_grids)]
    vmin, vmax = np.percentile(finite, 2), np.percentile(finite, 98)

    n_cols_panel = 5
    rows_per_method = int(np.ceil(n_show / n_cols_panel))
    fig2, axes2 = plt.subplots(
        2 * rows_per_method,
        n_cols_panel,
        figsize=(14, 4.3 * rows_per_method),
        squeeze=False,
    )
    fig2.subplots_adjust(hspace=0.05, wspace=0.05, right=0.88, top=0.92)

    method_labels = ["Prior", "EnRML"]
    method_colors = ["#ff7f0e", "#2ca02c"]
    for method_idx, (label, color) in enumerate(zip(method_labels, method_colors)):
        offset = method_idx * n_show
        for sample_idx in range(n_show):
            row = method_idx * rows_per_method + sample_idx // n_cols_panel
            col = sample_idx % n_cols_panel
            ax = axes2[row, col]
            ax.imshow(
                all_grids[offset + sample_idx].T,
                origin="lower",
                aspect="equal",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0 and sample_idx == 0:
                ax.set_ylabel(
                    label,
                    fontsize=11,
                    fontweight="bold",
                    color=color,
                    rotation=90,
                    labelpad=4,
                )
            if row == 0:
                ax.set_title(f"s{sample_idx + 1}", fontsize=8)

    for ax in axes2.ravel():
        if not ax.images:
            ax.axis("off")

    cbar_ax = fig2.add_axes([0.90, 0.08, 0.025, 0.80])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig2.colorbar(sm, cax=cbar_ax)
    cbar.set_label("log-permeability (log mD)", fontsize=11)

    fig2.suptitle("Parameter samples: Prior / EnRML", fontsize=13, fontweight="bold")
    sample_path = out_dir / "parameter_samples.png"
    fig2.savefig(sample_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {sample_path}")
    plt.close(fig2)

    print("Creating empirical variance figure...")
    var_grids = {
        "Prior": vec_to_grid(x_prior.var(axis=0)),
        "EnRML": vec_to_grid(x_enrml.var(axis=0)),
    }
    all_var_finite = np.concatenate([
        grid[np.isfinite(grid)].ravel() for grid in var_grids.values()
    ])
    vvar_min, vvar_max = 0.0, np.percentile(all_var_finite, 98)

    fig3, axes3 = plt.subplots(1, 2, figsize=(9.5, 4.8))
    fig3.subplots_adjust(wspace=0.08, right=0.86, top=0.86)
    for ax, name, color in zip(axes3, method_labels, method_colors):
        ax.imshow(
            var_grids[name].T,
            origin="lower",
            aspect="equal",
            cmap="hot_r",
            vmin=vvar_min,
            vmax=vvar_max,
        )
        ax.set_title(name, fontsize=12, fontweight="bold", color=color)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax3 = fig3.add_axes([0.88, 0.14, 0.025, 0.68])
    sm3 = plt.cm.ScalarMappable(
        cmap="hot_r",
        norm=plt.Normalize(vmin=vvar_min, vmax=vvar_max),
    )
    sm3.set_array([])
    cbar3 = fig3.colorbar(sm3, cax=cbar_ax3)
    cbar3.set_label("Empirical variance (log mD)^2", fontsize=11)

    fig3.suptitle(
        "Empirical variance comparison: Prior vs EnRML",
        fontsize=12,
        fontweight="bold",
    )
    var_path = out_dir / "variance_comparison.png"
    fig3.savefig(var_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {var_path}")
    plt.close(fig3)

    print("\nDone!")


if __name__ == "__main__":
    main()
