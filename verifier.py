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

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig

def math_verify_compute_score(model_output: str, ground_truth: str) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception as e:
        print(e)

    return {
        "score": 1.0 if ret_score > 0.5 else -1.0,
        "acc": 1.0 if ret_score > 0.5 else 0.0,
        "pred": model_output,
    }

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'openai/gsm8k':
        from verl.utils.reward_score import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval', 'math500', "math500_w_answer", "olympiadbench", "minerva_math"]:
        # from verl.utils.reward_score import math
        # res = math.compute_score(solution_str, ground_truth)
        # Use Math-Verify (https://github.com/huggingface/Math-Verify) for better evaluation accuracy
        from verl.utils.reward_score import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == 'math_dapo':
        from verl.utils.reward_score import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in ['math_dapo_boxed', 'amc_dapo_boxed', 'aime_2025_dapo_boxed', 'amc2023_dapo_boxed']:
        from verl.utils.reward_score import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from verl.utils.reward_score import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from verl.utils.reward_score import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from verl.utils.reward_score import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
