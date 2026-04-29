# Copyright (c) 2025 Zichen Zhao
# Columbia University School of Social Work
# Licensed under the MIT Academic Research License
# See LICENSE file in the project root for details.

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

def _cleanup_plots_directory():
    """Keep Plots/ figure-only by removing stale CSVs and deprecated pass/fail figures."""
    _ensure_dir(PLOTS_DIR)
    for filename in os.listdir(PLOTS_DIR):
        lower_name = filename.lower()
        if lower_name.endswith(".csv") or lower_name == "not_hate_passfail.png":
            try:
                os.remove(os.path.join(PLOTS_DIR, filename))
            except OSError:
                pass

def _remove_overall_average_row(df: pd.DataFrame) -> pd.DataFrame:
    if "Chatbot" not in df.columns:
        return df.copy()
    cleaned_df = df.copy()
    cleaned_df["Chatbot"] = cleaned_df["Chatbot"].astype(str).str.strip()
    return cleaned_df[cleaned_df["Chatbot"].str.lower() != OVERALL_AVERAGE_LABEL.lower()].copy()

def _coerce_metric_column(plot_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    cleaned_df = plot_df.copy()
    cleaned_df[metric] = pd.to_numeric(cleaned_df[metric], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=[metric])
    return cleaned_df

def append_overall_average_row(df: pd.DataFrame, label: str = OVERALL_AVERAGE_LABEL) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    summary_df = df.copy()
    numeric_cols = summary_df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        return summary_df

    overall_row = {}
    for col in summary_df.columns:
        if col == "Chatbot":
            overall_row[col] = label
        elif col in numeric_cols:
            overall_row[col] = round(float(summary_df[col].mean()), 4)
        else:
            overall_row[col] = ""

    return pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)


def plot_metric_bar(df: pd.DataFrame, metric: str, output_dir: str):
    if metric not in df.columns:
        print(f"[WARN] Metric '{metric}' not found in dataframe.")
        return

    _ensure_dir(output_dir)

    plot_df = _remove_overall_average_row(df)

    if "Chatbot" not in plot_df.columns:
        print("[WARN] 'Chatbot' column not found in dataframe.")
        return

    plot_df = plot_df[["Chatbot", metric]].copy()
    plot_df = _coerce_metric_column(plot_df, metric)

    if plot_df.empty:
        print(f"[WARN] No non-null numeric values found for '{metric}'.")
        return

    reference_metric_map = {
        "Negative Sentiment Probability": "Reference Negative Sentiment Probability",
        "Flesch Reading Ease": "Reference Flesch Reading Ease",
    }
    reference_value = None
    reference_col = reference_metric_map.get(metric)
    if reference_col and reference_col in df.columns:
        reference_values = pd.to_numeric(df[reference_col], errors="coerce").dropna()
        if not reference_values.empty:
            reference_value = float(reference_values.iloc[0])

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.bar(plot_df["Chatbot"], plot_df[metric])
    if reference_value is not None:
        plt.axhline(
            y=reference_value,
            linestyle="--",
            linewidth=1.8,
            label=f"Human reference = {reference_value:.4f}",
        )
        plt.legend()
    plt.xticks(rotation=ROTATION, ha="right")
    plt.ylabel(metric)
    plt.title(metric)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{_sanitize_filename(metric)}.png")
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def plot_not_hate_metric(not_hate_df: pd.DataFrame):
    _ensure_dir(PLOTS_DIR)
    clean_df = _remove_overall_average_row(not_hate_df)

    required_cols = ["Chatbot", "Non-Hateful Language Probability"]
    missing_cols = [col for col in required_cols if col not in clean_df.columns]
    if missing_cols:
        print(f"[WARN] Missing Non-Hateful Language columns: {missing_cols}")
        return

    plot_df = clean_df[["Chatbot", "Non-Hateful Language Probability"]].copy()
    plot_df["Non-Hateful Language Probability"] = pd.to_numeric(
        plot_df["Non-Hateful Language Probability"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["Non-Hateful Language Probability"])

    reference_value = None
    if "Reference Non-Hateful Language Probability" in not_hate_df.columns:
        reference_values = pd.to_numeric(
            not_hate_df["Reference Non-Hateful Language Probability"], errors="coerce"
        ).dropna()
        if not reference_values.empty:
            reference_value = float(reference_values.iloc[0])

    if not plot_df.empty:
        plt.figure(figsize=PLOT_FIGSIZE)
        plt.bar(plot_df["Chatbot"], plot_df["Non-Hateful Language Probability"])
        if reference_value is not None:
            plt.axhline(
                y=reference_value,
                linestyle="--",
                linewidth=1.8,
                label=f"Human reference = {reference_value:.4f}",
            )
            plt.legend()
        plt.xticks(rotation=ROTATION, ha="right")
        plt.ylabel("Non-Hateful Language Probability")
        plt.title("Non-Hateful Language Probability")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "non_hateful_language_probability.png"), dpi=DPI)
        plt.close()
    else:
        print("[WARN] Non-Hateful Language Probability plot skipped because the dataframe is empty.")


def plot_urgency_dimension(urgency_df: pd.DataFrame):
    _ensure_dir(PLOTS_DIR)
    clean_df = _remove_overall_average_row(urgency_df)

    required_cols = ["Chatbot", "Crisis-Response Reference Similarity"]
    missing_cols = [col for col in required_cols if col not in clean_df.columns]
    if missing_cols:
        print(f"[WARN] Missing urgency columns: {missing_cols}")
        return

    plot_df = clean_df[["Chatbot", "Crisis-Response Reference Similarity"]].copy()
    plot_df["Crisis-Response Reference Similarity"] = pd.to_numeric(
        plot_df["Crisis-Response Reference Similarity"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["Crisis-Response Reference Similarity"])

    if plot_df.empty:
        print("[WARN] Urgency plot skipped because the dataframe is empty.")
        return

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.bar(plot_df["Chatbot"], plot_df["Crisis-Response Reference Similarity"])
    plt.xticks(rotation=ROTATION, ha="right")
    plt.ylabel("Crisis-Response Reference Similarity")
    plt.title("Crisis-Response Reference Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "crisis_response_reference_similarity.png"), dpi=DPI)
    plt.close()


def plot_risk_factor_dimension(risk_factor_df: pd.DataFrame):
    _ensure_dir(PLOTS_DIR)
    clean_df = _remove_overall_average_row(risk_factor_df)

    required_cols = ["Chatbot", "Risk-Assessment Reference Similarity"]
    missing_cols = [col for col in required_cols if col not in clean_df.columns]
    if missing_cols:
        print(f"[WARN] Missing risk-assessment columns: {missing_cols}")
        return

    plot_df = clean_df[["Chatbot", "Risk-Assessment Reference Similarity"]].copy()
    plot_df["Risk-Assessment Reference Similarity"] = pd.to_numeric(
        plot_df["Risk-Assessment Reference Similarity"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["Risk-Assessment Reference Similarity"])

    if plot_df.empty:
        print("[WARN] Risk-factor plot skipped because the dataframe is empty.")
        return

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.bar(plot_df["Chatbot"], plot_df["Risk-Assessment Reference Similarity"])
    plt.xticks(rotation=ROTATION, ha="right")
    plt.ylabel("Risk-Assessment Reference Similarity")
    plt.title("Risk-Assessment Reference Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "risk_assessment_reference_similarity.png"), dpi=DPI)
    plt.close()


# backward-compatible wrappers
def plot_identity_dimension(identity_df: pd.DataFrame):
    plot_urgency_dimension(identity_df)


def plot_safety_dimension(safety_df: pd.DataFrame):
    plot_risk_factor_dimension(safety_df)

def build_overall_summary_table(
    evaluation_df: pd.DataFrame,
    not_hate_df: pd.DataFrame | None = None,
    urgency_df: pd.DataFrame | None = None,
    risk_factor_df: pd.DataFrame | None = None,
    identity_df: pd.DataFrame | None = None,
    safety_df: pd.DataFrame | None = None,
    include_overall_average: bool = False,
) -> pd.DataFrame:
    summary_df = _remove_overall_average_row(evaluation_df).copy()
    if "Response" in summary_df.columns:
        summary_df = summary_df.drop(columns=["Response"])
    # Accept both the new split dataframes and older argument names.
    if urgency_df is None and identity_df is not None:
        urgency_df = identity_df
    if risk_factor_df is None and safety_df is not None:
        risk_factor_df = safety_df

    for component_df in [not_hate_df, urgency_df, risk_factor_df]:
        if component_df is not None:
            component_clean = _remove_overall_average_row(component_df).copy()
            summary_df = summary_df.merge(component_clean, on="Chatbot", how="left")
    existing_cols = [col for col in OVERALL_SUMMARY_COLUMNS if col in summary_df.columns]
    remaining_cols = [col for col in summary_df.columns if col not in existing_cols]
    summary_df = summary_df[existing_cols + remaining_cols]
    if include_overall_average:
        summary_df = append_overall_average_row(
            summary_df,
            label=OVERALL_AVERAGE_LABEL,
        )
    return summary_df

def save_overall_summary_table(summary_df: pd.DataFrame, output_path: str):
    summary_df.to_csv(output_path, index=False)


# =================================
# ROBUSTNESS / SENSITIVITY ANALYSES
# =================================

def _get_available_robustness_metrics(df: pd.DataFrame) -> list[str]:
    available = []
    for metric in ROBUSTNESS_METRICS:
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors="coerce")
            if values.notna().sum() >= 2:
                available.append(metric)
    return available


def _normalize_metric_value(metric: str, value: float) -> float:
    denominator = METRIC_NORMALIZATION_DENOMINATORS.get(metric, 1.0)
    if denominator in [0, None] or pd.isna(value):
        return np.nan
    return float(value) / float(denominator)


def generate_metric_correlation_matrix(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive Spearman correlation matrix across benchmark metrics.

    This output is a robustness/diagnostic analysis rather than confirmatory
    null-hypothesis inference because the number of evaluated chatbot systems is small.
    """
    clean_df = _remove_overall_average_row(evaluation_df)
    metric_cols = _get_available_robustness_metrics(clean_df)

    if len(metric_cols) < 2:
        print("[WARN] Correlation matrix skipped: fewer than two numeric metrics available.")
        return pd.DataFrame()

    metric_df = clean_df[metric_cols].apply(pd.to_numeric, errors="coerce")
    return metric_df.corr(method="spearman").round(4)


def save_metric_correlation_matrix(corr_df: pd.DataFrame):
    if corr_df.empty:
        return
    _ensure_dir(ROBUSTNESS_DIR)
    corr_df.to_csv(CORRELATION_MATRIX_CSV_PATH)


def plot_metric_correlation_matrix(corr_df: pd.DataFrame):
    if corr_df.empty:
        return

    _ensure_dir(PLOTS_DIR)
    labels = corr_df.columns.tolist()
    matrix = corr_df.to_numpy(dtype=float)

    plt.figure(figsize=(max(9, len(labels) * 1.2), max(7, len(labels) * 1.0)))
    plt.imshow(matrix, vmin=-1, vmax=1)
    plt.colorbar(label="Spearman correlation")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            if np.isfinite(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Spearman Correlation Across Benchmark Metrics")
    plt.tight_layout()
    plt.savefig(CORRELATION_MATRIX_PLOT_PATH, dpi=DPI)
    plt.close()


def _merge_component_scores_for_robustness(
    evaluation_df: pd.DataFrame,
    not_hate_df: pd.DataFrame,
    urgency_df: pd.DataFrame,
    risk_factor_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_df = evaluation_df.copy()
    for component_df in [not_hate_df, urgency_df, risk_factor_df]:
        if component_df is None or component_df.empty:
            continue
        clean_component_df = component_df.copy()
        merge_cols = [col for col in clean_component_df.columns if col != "Response"]
        clean_component_df = clean_component_df[merge_cols]
        duplicate_cols = [
            col for col in clean_component_df.columns
            if col != "Chatbot" and col in merged_df.columns
        ]
        if duplicate_cols:
            merged_df = merged_df.drop(columns=duplicate_cols)
        merged_df = merged_df.merge(clean_component_df, on="Chatbot", how="left")
    return merged_df


def _compute_full_metric_table_from_integrated(integrated_responses: pd.DataFrame) -> pd.DataFrame:
    """Recompute all benchmark metrics from a filtered integrated response table."""
    from src.utils.evaluation_algo import (
        generate_evaluation_scores,
        generate_not_hate_metric_scores,
        generate_urgency_dimension_scores,
        generate_risk_factor_dimension_scores,
    )

    evaluation_df = generate_evaluation_scores(
        integrated_responses,
        include_overall_average=False,
    )
    not_hate_df = generate_not_hate_metric_scores(
        integrated_responses,
        include_overall_average=False,
    )
    urgency_df = generate_urgency_dimension_scores(
        integrated_responses,
        include_overall_average=False,
    )
    risk_factor_df = generate_risk_factor_dimension_scores(
        integrated_responses,
        include_overall_average=False,
    )

    return _merge_component_scores_for_robustness(
        evaluation_df=evaluation_df,
        not_hate_df=not_hate_df,
        urgency_df=urgency_df,
        risk_factor_df=risk_factor_df,
    )


def _topic_sort_key_for_robustness(topic: str):
    if topic in ROBUSTNESS_TOPIC_ORDER:
        return (ROBUSTNESS_TOPIC_ORDER.index(topic), topic)
    if topic in CANONICAL_TOPIC_ORDER:
        return (len(ROBUSTNESS_TOPIC_ORDER) + CANONICAL_TOPIC_ORDER.index(topic), topic)
    return (len(ROBUSTNESS_TOPIC_ORDER) + len(CANONICAL_TOPIC_ORDER) + 1, topic)


def generate_leave_one_topic_out_sensitivity(
    integrated_responses: pd.DataFrame,
    full_evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Leave-one-topic-out sensitivity analysis limited to formal assessment topics.

    The scope note/disclaimer topic is intentionally excluded. Flesch Reading Ease
    is also normalized to a 0-1 scale for cross-metric delta summaries.
    """
    if integrated_responses is None or integrated_responses.empty:
        print("[WARN] Leave-one-topic-out sensitivity skipped: integrated responses not provided.")
        return pd.DataFrame()

    from src.utils.evaluation_algo import standardize_topic

    working_df = integrated_responses.copy()
    if TOPIC_COL not in working_df.columns:
        print("[WARN] Leave-one-topic-out sensitivity skipped: topic column not found.")
        return pd.DataFrame()

    working_df[TOPIC_COL] = working_df[TOPIC_COL].apply(standardize_topic)

    available_reference_topics = set(
        working_df.loc[
            working_df[PLATFORM_COL].astype(str).str.lower() == HUMAN_PLATFORM.lower(),
            TOPIC_COL,
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    target_topics = [topic for topic in ROBUSTNESS_TOPIC_ORDER if topic in available_reference_topics]
    target_topics = sorted(target_topics, key=_topic_sort_key_for_robustness)

    full_df = _remove_overall_average_row(full_evaluation_df).copy()
    metric_cols = _get_available_robustness_metrics(full_df)

    if not target_topics:
        print("[WARN] Leave-one-topic-out sensitivity skipped: none of the target topics were found.")
        return pd.DataFrame()
    if not metric_cols:
        print("[WARN] Leave-one-topic-out sensitivity skipped: no robustness metrics found.")
        return pd.DataFrame()

    full_df = full_df.set_index("Chatbot")
    rows = []

    for excluded_topic in target_topics:
        filtered_df = working_df[working_df[TOPIC_COL] != excluded_topic].copy()
        if filtered_df.empty:
            continue

        try:
            loo_df = _compute_full_metric_table_from_integrated(filtered_df)
        except Exception as exc:
            print(f"[WARN] Sensitivity skipped for topic '{excluded_topic}': {exc}")
            continue

        loo_df = _remove_overall_average_row(loo_df).set_index("Chatbot")
        shared_chatbots = [chatbot for chatbot in full_df.index if chatbot in loo_df.index]

        for chatbot in shared_chatbots:
            for metric in metric_cols:
                if metric not in loo_df.columns:
                    continue

                full_score = pd.to_numeric(pd.Series([full_df.loc[chatbot, metric]]), errors="coerce").iloc[0]
                loo_score = pd.to_numeric(pd.Series([loo_df.loc[chatbot, metric]]), errors="coerce").iloc[0]

                if pd.isna(full_score) or pd.isna(loo_score):
                    continue

                normalized_full_score = _normalize_metric_value(metric, full_score)
                normalized_loo_score = _normalize_metric_value(metric, loo_score)

                rows.append(
                    {
                        "Excluded Topic": excluded_topic,
                        "Chatbot": chatbot,
                        "Metric": metric,
                        "Full Score": round(float(full_score), 4),
                        "Leave-One-Topic-Out Score": round(float(loo_score), 4),
                        "Absolute Delta": round(abs(float(loo_score) - float(full_score)), 4),
                        "Normalized Full Score": round(float(normalized_full_score), 4),
                        "Normalized Leave-One-Topic-Out Score": round(float(normalized_loo_score), 4),
                        "Normalized Absolute Delta": round(abs(float(normalized_loo_score) - float(normalized_full_score)), 4),
                    }
                )

    return pd.DataFrame(rows)


def save_leave_one_topic_out_sensitivity(sensitivity_df: pd.DataFrame):
    if sensitivity_df.empty:
        return
    _ensure_dir(ROBUSTNESS_DIR)
    sensitivity_df.to_csv(LEAVE_ONE_TOPIC_OUT_CSV_PATH, index=False)
    normalized_cols = [
        "Excluded Topic",
        "Chatbot",
        "Metric",
        "Normalized Full Score",
        "Normalized Leave-One-Topic-Out Score",
        "Normalized Absolute Delta",
    ]
    existing_cols = [col for col in normalized_cols if col in sensitivity_df.columns]
    sensitivity_df[existing_cols].to_csv(LEAVE_ONE_TOPIC_OUT_NORMALIZED_CSV_PATH, index=False)


def _plot_sensitivity_summary(
    sensitivity_df: pd.DataFrame,
    delta_col: str,
    ylabel: str,
    title: str,
    output_path: str,
):
    if sensitivity_df.empty or delta_col not in sensitivity_df.columns:
        return
    summary_df = (
        sensitivity_df
        .groupby("Metric", as_index=False)[delta_col]
        .mean()
        .sort_values(delta_col, ascending=False)
    )
    if summary_df.empty:
        return

    plt.figure(figsize=(max(9, len(summary_df) * 1.15), 5.8))
    plt.bar(summary_df["Metric"], summary_df[delta_col])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def plot_leave_one_topic_out_sensitivity(sensitivity_df: pd.DataFrame):
    if sensitivity_df.empty:
        return
    _ensure_dir(PLOTS_DIR)
    _plot_sensitivity_summary(
        sensitivity_df=sensitivity_df,
        delta_col="Absolute Delta",
        ylabel="Mean absolute change after removing one topic",
        title="Leave-One-Topic-Out Sensitivity",
        output_path=LEAVE_ONE_TOPIC_OUT_PLOT_PATH,
    )
    _plot_sensitivity_summary(
        sensitivity_df=sensitivity_df,
        delta_col="Normalized Absolute Delta",
        ylabel="Mean normalized absolute change after removing one topic",
        title="Leave-One-Topic-Out Sensitivity, Normalized Metric Scale",
        output_path=LEAVE_ONE_TOPIC_OUT_NORMALIZED_PLOT_PATH,
    )


def run_robustness_outputs(
    evaluation_df: pd.DataFrame,
    integrated_responses: pd.DataFrame | None = None,
):
    corr_df = generate_metric_correlation_matrix(evaluation_df)
    save_metric_correlation_matrix(corr_df)
    plot_metric_correlation_matrix(corr_df)

    if integrated_responses is not None:
        sensitivity_df = generate_leave_one_topic_out_sensitivity(
            integrated_responses=integrated_responses,
            full_evaluation_df=evaluation_df,
        )
        save_leave_one_topic_out_sensitivity(sensitivity_df)
        plot_leave_one_topic_out_sensitivity(sensitivity_df)


def process_all_outputs(
    evaluation_df: pd.DataFrame,
    integrated_responses: pd.DataFrame | None = None,
    not_hate_df: pd.DataFrame | None = None,
    urgency_df: pd.DataFrame | None = None,
    risk_factor_df: pd.DataFrame | None = None,
    identity_df: pd.DataFrame | None = None,
    safety_df: pd.DataFrame | None = None,
):
    _cleanup_plots_directory()
    for metric in VISUALIZATION_METRICS:
        plot_metric_bar(evaluation_df, metric, PLOTS_DIR)

    # Accept both new split arguments and old positional identity/safety calls.
    if urgency_df is None and identity_df is not None:
        urgency_df = identity_df
    if risk_factor_df is None and safety_df is not None:
        risk_factor_df = safety_df

    if not_hate_df is not None:
        plot_not_hate_metric(not_hate_df)
    if urgency_df is not None:
        plot_urgency_dimension(urgency_df)
    if risk_factor_df is not None:
        plot_risk_factor_dimension(risk_factor_df)

    run_robustness_outputs(
        evaluation_df=evaluation_df,
        integrated_responses=integrated_responses,
    )
