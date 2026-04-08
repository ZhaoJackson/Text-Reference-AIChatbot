# Copyright (c) 2025 Zichen Zhao
# Columbia University School of Social Work
# Licensed under the MIT Academic Research License
# See LICENSE file in the project root for details.

"""
Main execution script for the benchmark pipeline.
"""

from __future__ import annotations

import pandas as pd

from src.commonconst import *
from src.data.data_processing import (
    extract_text_from_docx,
    save_processed_files,
)
from src.utils.evaluation_algo import (
    ensure_output_dirs,
    generate_evaluation_scores,
    generate_identity_dimension_scores,
    generate_safety_dimension_scores,
    save_evaluation_to_csv,
)
from src.utils.output_processing import process_all_outputs


def main():
    ensure_output_dirs()

    # Step 1: load raw docx text
    reference_text = extract_text_from_docx(REFERENCE_DOCX_PATH)
    chatbot_text = extract_text_from_docx(CHATBOT_DOCX_PATH)

    # Step 2: process and save all intermediate files
    # NOTE:
    # save_processed_files() already writes:
    # - processed chatbot CSV
    # - processed reference CSV
    # - integrated responses CSV
    save_processed_files(
        chatbot_text=chatbot_text,
        reference_text=reference_text,
        chatbot_output_path=CHATBOT_PROCESSED_CSV_PATH,
        reference_output_path=REFERENCE_PROCESSED_CSV_PATH,
        integrated_output_path=INTEGRATED_OUTPUT_CSV_PATH,
    )

    # Step 3: load integrated responses
    integrated_responses = pd.read_csv(INTEGRATED_OUTPUT_CSV_PATH)

    # Step 4: primary continuous metrics
    evaluation_df = generate_evaluation_scores(integrated_responses)
    save_evaluation_to_csv(OUTPUT_CSV_PATH, evaluation_df)

    # Step 5: triangulated dimensions
    identity_df = generate_identity_dimension_scores(integrated_responses)
    safety_df = generate_safety_dimension_scores(integrated_responses)

    # Step 6: plotting
    process_all_outputs(evaluation_df, identity_df, safety_df)

    print("Benchmark evaluation complete.")
    print(f"Main results saved to: {OUTPUT_CSV_PATH}")
    print(f"Integrated responses saved to: {INTEGRATED_OUTPUT_CSV_PATH}")
    print(f"Identity dimension results saved to: {IDENTITY_DIMENSION_CSV_PATH}")
    print(f"Safety dimension results saved to: {SAFETY_DIMENSION_CSV_PATH}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Dimension plots saved to: {SENSITIVITY_DIR}")


if __name__ == "__main__":
    main()