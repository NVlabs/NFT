# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import json
from experience_maker import preprocess_box_response_for_qwen_prompt
from verl.utils.reward_score import prime_math, math_verify, math_dapo
import argparse


RESPONSE_COL = "responses"
GROUND_TRUTH_COL = "reward_model"
GROUND_TRUTH_KEY = "ground_truth"


def calculate_accuracy(file_path: str, verifier_type: str) -> None:
    """Loads data, calculates accuracy, and adds verifier results to DataFrame."""
    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
    except Exception as e:
        print(f"Error loading Parquet file '{file_path}': {e}")
        return

    total_correct = 0
    total_checked = 0
    all_results = []

    verifier = {
        "qwen": preprocess_box_response_for_qwen_prompt,
        "prime": lambda response, answer: prime_math.compute_score(response, answer)[
            "acc"
        ],
        "math": lambda response, answer: math_verify.compute_score(
            response, answer
        )["acc"],
        "dapo": lambda response, answer: math_dapo.compute_score(
            response, answer, strict_box_verify=True
        )["acc"],
    }[verifier_type]

    for index, row in df.iterrows():
        # 1. Get Ground Truth
        gt_data = row[GROUND_TRUTH_COL]
        if isinstance(gt_data, str):
            gt_dict = json.loads(gt_data)
        else:
            gt_dict = gt_data
        ground_truth_answer = str(gt_dict[GROUND_TRUTH_KEY])

        # 2. Get Responses (list)
        generated_responses = row[RESPONSE_COL]

        # 3. Compare each response
        row_results = []
        for response in generated_responses:
            try:
                is_correct = verifier(str(response), ground_truth_answer)
            except Exception as e:
                print(f"Warning: Skipping  due to error: {e}")
                is_correct=-1
            is_correct = float(is_correct)
            row_results.append(is_correct)
            total_correct += is_correct
            total_checked += 1
        all_results.append(row_results)

    # Add result column to DataFrame
    df[f"{verifier_type}_results"] = all_results

    # Save updated DataFrame
    df.to_parquet(file_path)
    print(f"Updated DataFrame saved to: {file_path}")

    # Print Summary
    if total_checked > 0:
        accuracy = total_correct / total_checked
        print(f"\n--- Results --- for {verifier_type}")
        print(f"Total responses checked: {total_checked}")
        print(f"Total correct responses: {int(total_correct)}")
        print(f"Accuracy: {accuracy:.4f}")
    else:
        print("\n--- Results ---")
        print("No responses were checked.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate accuracy from generated responses in a Parquet file."
    )
    parser.add_argument("--input_path", help="Path to the input Parquet file.")
    parser.add_argument(
        "--verifier", default="qwen", choices=["qwen", "prime", "math", "dapo", "all"]
    )
    args = parser.parse_args()
    if args.verifier == "all":
        for ven in ["qwen", "prime", "math", "dapo"]:
            calculate_accuracy(args.input_path, ven)
    else:
        calculate_accuracy(args.input_path, args.verifier)