# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    ReasoningBudgetLogitsProcessor,
    validate_logits_processors_parameters,
)


def test_reasoning_budget_soft_penalty_flow():
    output_token_ids: list[int] = []
    params = SamplingParams(
        reasoning_soft_penalty_start_threshold=1,
        reasoning_soft_penalty_coefficient=2.0,
        reasoning_soft_penalty_curve="linear",
        reasoning_soft_penalty_end_token_id=3,
    )
    processor = ReasoningBudgetLogitsProcessor(
        None, torch.device("cpu"), is_pin_memory=False
    )
    processor.update_state(
        BatchUpdate(batch_size=1, removed=(), moved=(), added=((0, params, [], output_token_ids),))
    )

    # Before threshold: unchanged.
    logits_before = torch.zeros(1, 8)
    out = processor.apply(logits_before.clone())
    assert torch.equal(out, logits_before)

    # Reached threshold while reasoning: apply negative continuation bias
    # and positive end-token bias.
    output_token_ids.append(5)
    logits_mid = torch.zeros(1, 8)
    out = processor.apply(logits_mid.clone())
    assert out[0, 3].item() == 2.0
    assert out[0, 1].item() == -2.0

    # After end token has appeared, no further soft penalty.
    output_token_ids.append(3)
    logits_after = torch.zeros(1, 8)
    out = processor.apply(logits_after.clone())
    assert torch.equal(out, logits_after)


def test_reasoning_budget_soft_penalty_validation():
    valid = SamplingParams(
        reasoning_soft_penalty_start_threshold=0,
        reasoning_soft_penalty_coefficient=1.0,
        reasoning_soft_penalty_curve="quadratic",
        reasoning_soft_penalty_end_token_id=1,
    )
    validate_logits_processors_parameters(None, valid)

    invalid_coef = SamplingParams(reasoning_soft_penalty_coefficient=-0.1)
    try:
        validate_logits_processors_parameters(None, invalid_coef)
    except ValueError as e:
        assert "reasoning_soft_penalty_coefficient" in str(e)
    else:
        raise AssertionError("Expected coefficient validation error")
