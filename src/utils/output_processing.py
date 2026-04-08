# Copyright (c) 2025 Zichen Zhao
# Columbia University School of Social Work
# Licensed under the MIT Academic Research License
# See LICENSE file in the project root for details.

"""
Output processing and plotting module for benchmark results.
"""

from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from src.commonconst import *


def _sanitize_filename(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s/]+", "_", name)
    return name


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_metric_bar(df: pd.DataFrame, metric: str, output_dir: str):
    if metric not in df.columns:
        print(f"[WARN] Metric '{metric}' not found in dataframe.")
        return

    _ensure_dir(output_dir)

    plot_df = df[["Chatbot", metric]].copy()
    plot_df = plot_df.dropna(subset=[metric])

    if plot_df.empty:
        print(f"[WARN] No non-null values found for '{metric}'.")
        return

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.bar(plot_df["Chatbot"], plot_df[metric])
    plt.xticks(rotation=ROTATION, ha="right")
    plt.ylabel(metric)
    plt.title(metric)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{_sanitize_filename(metric)}.png")
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def plot_identity_dimension(identity_df: pd.DataFrame):
    _ensure_dir(SENSITIVITY_DIR)

    # Probability + reference alignment
    plot_df = identity_df[
        [
            "Chatbot",
            "Identity-Harm Floor Probability",
            "Identity-Specific Reference Alignment",
        ]
    ].copy().set_index("Chatbot")

    ax = plot_df.plot(kind="bar", figsize=PLOT_COMPARISON_FIGSIZE)
    ax.set_title("Identity Dimension Comparison")
    ax.set_ylabel("Score")
    plt.xticks(rotation=ROTATION, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(SENSITIVITY_DIR, "identity_dimension_comparison.png"),
        dpi=DPI,
    )
    plt.close()

    # Pass/fail
    pass_df = identity_df[["Chatbot", "Identity-Harm Floor Pass"]].copy()
    plt.figure(figsize=PLOT_FIGSIZE)
    plt.bar(pass_df["Chatbot"], pass_df["Identity-Harm Floor Pass"])
    plt.xticks(rotation=ROTATION, ha="right")
    plt.ylabel("Pass (1) / Fail (0)")
    plt.title("Identity-Harm Floor Pass/Fail")
    plt.tight_layout()
    plt.savefig(
        os.path.join(SENSITIVITY_DIR, "identity_harm_floor_passfail.png"),
        dpi=DPI,
    )
    plt.close()


def plot_safety_dimension(safety_df: pd.DataFrame):
    _ensure_dir(SENSITIVITY_DIR)

    plot_df = safety_df[
        [
            "Chatbot",
            "Crisis-Support Reference Alignment",
        ]
    ].copy()

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.bar(plot_df["Chatbot"], plot_df["Crisis-Support Reference Alignment"])
    plt.xticks(rotation=ROTATION, ha="right")
    plt.ylabel("Crisis-Support Reference Alignment")
    plt.title("Safety Dimension Comparison")
    plt.tight_layout()
    plt.savefig(
        os.path.join(SENSITIVITY_DIR, "safety_dimension_comparison.png"),
        dpi=DPI,
    )
    plt.close()


def process_all_outputs(
    evaluation_df: pd.DataFrame,
    identity_df: pd.DataFrame | None = None,
    safety_df: pd.DataFrame | None = None,
):
    _ensure_dir(PLOTS_DIR)
    _ensure_dir(SENSITIVITY_DIR)

    for metric in VISUALIZATION_METRICS:
        plot_metric_bar(evaluation_df, metric, PLOTS_DIR)

    if identity_df is not None:
        plot_identity_dimension(identity_df)

    if safety_df is not None:
        plot_safety_dimension(safety_df)