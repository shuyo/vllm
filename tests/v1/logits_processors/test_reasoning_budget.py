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

    # After end token has appeared, disable repeated end token generation.
    output_token_ids.append(3)
    logits_after = torch.zeros(1, 8)
    out = processor.apply(logits_after.clone())
    assert torch.isneginf(out[0, 3])
    assert out[0, 1].item() == 0.0


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


def test_reasoning_budget_soft_penalty_stops_after_end_token_failsafe():
    output_token_ids: list[int] = [5, 3]
    params = SamplingParams(
        reasoning_soft_penalty_start_threshold=0,
        reasoning_soft_penalty_coefficient=3.0,
        reasoning_soft_penalty_curve="linear",
        reasoning_soft_penalty_end_token_id=3,
    )
    processor = ReasoningBudgetLogitsProcessor(
        None, torch.device("cpu"), is_pin_memory=False
    )
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=(),
            moved=(),
            added=((0, params, [], output_token_ids),),
        )
    )

    # Simulate stale/corrupted state where reasoning flag did not flip.
    req_info = processor.req_info[0]
    req_info.is_reasoning = True
    req_info.processed_len = len(output_token_ids)

    logits = torch.zeros(1, 8)
    out = processor.apply(logits.clone())
    assert torch.isneginf(out[0, 3])


def test_reasoning_budget_ignores_placeholder_tokens():
    output_token_ids: list[int] = [10, 11, -1, -1]
    params = SamplingParams(
        reasoning_soft_penalty_start_threshold=3,
        reasoning_soft_penalty_coefficient=1.0,
        reasoning_soft_penalty_curve="linear",
        reasoning_soft_penalty_end_token_id=3,
    )
    processor = ReasoningBudgetLogitsProcessor(
        None, torch.device("cpu"), is_pin_memory=False
    )
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=(),
            moved=(),
            added=((0, params, [], output_token_ids),),
        )
    )

    # only valid tokens before first -1 should be counted
    req_info = processor.req_info[0]
    assert req_info.reasoning_token_count == 2
    assert req_info.processed_len == 2

    logits = torch.zeros(1, 8)
    out = processor.apply(logits.clone())
    # threshold not reached yet because -1 placeholders are ignored
    assert torch.equal(out, logits)


def test_reasoning_budget_counts_non_prefix_valid_tokens():
    # Placeholders can appear before valid ids in async/spec paths.
    output_token_ids: list[int] = [-1, -1, 20, 21]
    params = SamplingParams(
        reasoning_soft_penalty_start_threshold=1,
        reasoning_soft_penalty_coefficient=1.0,
        reasoning_soft_penalty_curve="linear",
        reasoning_soft_penalty_end_token_id=3,
    )
    processor = ReasoningBudgetLogitsProcessor(
        None, torch.device("cpu"), is_pin_memory=False
    )
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=(),
            moved=(),
            added=((0, params, [], output_token_ids),),
        )
    )
    req_info = processor.req_info[0]
    assert req_info.reasoning_token_count == 2

    logits = torch.zeros(1, 8)
    out = processor.apply(logits.clone())
    # threshold reached => penalty is applied
    assert out[0, 1].item() < 0


def test_reasoning_budget_uses_raw_len_surrogate_when_only_placeholders():
    output_token_ids: list[int] = [-1, -1, -1]
    params = SamplingParams(
        reasoning_soft_penalty_start_threshold=2,
        reasoning_soft_penalty_coefficient=1.0,
        reasoning_soft_penalty_curve="linear",
        reasoning_soft_penalty_end_token_id=3,
    )
    processor = ReasoningBudgetLogitsProcessor(
        None, torch.device("cpu"), is_pin_memory=False
    )
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=(),
            moved=(),
            added=((0, params, [], output_token_ids),),
        )
    )

    logits = torch.zeros(1, 8)
    out = processor.apply(logits.clone())
    # Surrogate count is based on raw_len=3, so threshold=2 is exceeded.
    assert out[0, 1].item() < 0

    req_info = processor.req_info[0]
    assert req_info.reasoning_token_count == 3
    assert req_info.using_surrogate_count

    # Additional surrogate penalty applications should be skipped.
    out2 = processor.apply(logits.clone())
    assert torch.equal(out2, logits)
