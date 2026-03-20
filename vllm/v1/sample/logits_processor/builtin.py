# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch

from vllm import SamplingParams
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

T = TypeVar("T")
logger = init_logger(__name__)


class MinPLogitsProcessor(LogitsProcessor):
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.min_p_count: int = 0

        self.min_p_cpu_tensor = torch.zeros(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=is_pin_memory
        )
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()

        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            # Pre-allocated device tensor
            self.min_p_device: torch.Tensor = torch.empty(
                (max_num_reqs,), dtype=torch.float32, device=device
            )
        else:
            self.min_p_device = self.min_p_cpu_tensor
        # Current slice of the device tensor
        self.min_p: torch.Tensor = self.min_p_device[:0]

    def is_argmax_invariant(self) -> bool:
        """Min-p never impacts greedy sampling"""
        return True

    def get_min_p_by_index(self, index: int) -> float:
        return float(self.min_p_cpu[index])

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return

        needs_update = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            min_p = params.min_p
            min_p_before = self.min_p_cpu[index]
            if min_p_before != min_p:
                needs_update = True
                self.min_p_cpu[index] = min_p
                if min_p and not min_p_before:
                    self.min_p_count += 1
                elif not min_p and min_p_before:
                    self.min_p_count -= 1

        if self.min_p_count:
            # Process removed requests.
            if batch_update.removed:
                needs_update = True
                for index in batch_update.removed:
                    if self.min_p_cpu[index]:
                        self.min_p_cpu[index] = 0
                        self.min_p_count -= 1

            # Process moved requests, unidirectional (a->b) and swap (a<->b).
            for adx, bdx, direct in batch_update.moved:
                min_p_a, min_p_b = self.min_p_cpu[adx], self.min_p_cpu[bdx]
                if min_p_a != min_p_b:
                    needs_update = True
                    self.min_p_cpu[bdx] = min_p_a
                    if direct == MoveDirectionality.SWAP:
                        self.min_p_cpu[adx] = min_p_b
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if min_p_a:
                        self.min_p_cpu[adx] = 0
                    if min_p_b:
                        self.min_p_count -= 1

        # Update tensors if needed.
        size = batch_update.batch_size
        if self.min_p_count and (needs_update or self.min_p.shape[0] != size):
            self.min_p = self.min_p_device[:size]
            if self.use_double_tensor:
                self.min_p.copy_(self.min_p_cpu_tensor[:size], non_blocking=True)
            self.min_p.unsqueeze_(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits

        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        # Adjust min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # Identify valid tokens using threshold comparison
        invalid_token_mask = probability_values < adjusted_min_p
        # Apply mask using boolean indexing
        logits.masked_fill_(invalid_token_mask, -float("inf"))
        return logits


class LogitBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.pin_memory = is_pin_memory
        self.biases: dict[int, dict[int, float]] = {}

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.biases, batch_update, lambda params, _, __: params.logit_bias or None
        )

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            biases: list[float] = []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._device_tensor(biases, torch.float32)
            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            data, device="cpu", dtype=dtype, pin_memory=self.pin_memory
        ).to(device=self.device, non_blocking=True)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class MinTokensLogitsProcessor(LogitsProcessor):
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        # index -> (min_toks, output_token_ids, stop_token_ids)
        self.device = device
        self.pin_memory = is_pin_memory
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}

        # (req_idx_tensor,eos_tok_id_tensor)
        self.logits_slice: tuple[torch.Tensor, torch.Tensor] = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

        self.neg_inf_tensor = torch.tensor(
            -float("inf"), dtype=torch.float32, device=self.device
        )

    def is_argmax_invariant(self) -> bool:
        """By censoring stop tokens, min-tokens can change the outcome
        of the argmax operation in greedy sampling."""
        return False

    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int] | None, output_tok_ids: list[int]
    ) -> tuple[int, Sequence[int], set[int]] | None:
        min_tokens = params.min_tokens
        if not min_tokens or len(output_tok_ids) >= min_tokens:
            return None
        return min_tokens, output_tok_ids, params.all_stop_token_ids

    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.min_toks, batch_update, self.add_request
        )
        if self.min_toks:
            # Check for any requests that have attained their min tokens.
            to_remove = tuple(
                index
                for index, (min_toks, out_tok_ids, _) in self.min_toks.items()
                if len(out_tok_ids) >= min_toks
            )
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            for req, (_, _, stop_tok_ids) in self.min_toks.items():
                reqs.extend([req] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)

            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            data, device="cpu", dtype=dtype, pin_memory=self.pin_memory
        ).to(device=self.device, non_blocking=True)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
        return logits

    def apply_with_spec_decode(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int],
    ) -> torch.Tensor:
        """Spec-decode version of apply().
        Priority: ``min_tokens`` > ``stop_token_ids`` / EOS.
        Example: ``num_draft_tokens = [2, 3, 1]``
          → ``logits`` shape ``[6, V]``, ``cumsum = [0, 2, 5, 6]``
          → request 0 owns rows 0‑1, request 1 rows 2‑4, request 2 row 5.
        """
        if not self.min_toks:
            return logits

        num_draft_arr = np.array(num_draft_tokens, dtype=np.int64)
        cumsum = np.concatenate([[0], np.cumsum(num_draft_arr)])

        entries = [
            (req_idx, min_tok, len(out_tok_ids), list(stop_tok_ids))
            for req_idx, (min_tok, out_tok_ids, stop_tok_ids) in self.min_toks.items()
            if stop_tok_ids
        ]

        if not entries:
            return logits

        all_rows: list[np.ndarray] = []  # row indices to mask
        all_toks: list[np.ndarray] = []  # stop-token ids at those rows

        for req_idx, min_tok, current_len, stop_toks in entries:
            remaining = min_tok - current_len
            # How many leading draft positions still need stop-token masking.
            n_mask = int(min(max(remaining, 0), num_draft_arr[req_idx]))

            if n_mask > 0:
                offset = cumsum[req_idx]
                row_indices = np.arange(offset, offset + n_mask, dtype=np.int64)
                n_stop = len(stop_toks)
                all_rows.append(np.repeat(row_indices, n_stop))
                all_toks.append(np.tile(stop_toks, n_mask))

        if all_rows:
            rows_arr = np.concatenate(all_rows)
            toks_arr = np.concatenate(all_toks)
            # (row_indices, token_indices) for index_put_ to set -inf.
            logits_slice = (
                torch.from_numpy(rows_arr).to(self.device, non_blocking=True),
                torch.from_numpy(toks_arr).to(self.device, non_blocking=True),
            )
            logits.index_put_(logits_slice, self.neg_inf_tensor)

        return logits


@dataclass
class _ReasoningBudgetReqInfo:
    output_token_ids: list[int]
    end_token_id: int
    start_threshold: int
    coefficient: float
    curve: str
    is_reasoning: bool = True
    reasoning_token_count: int = 0
    processed_len: int = 0

    def valid_output_tokens(self) -> list[int]:
        # Async scheduling/spec decode may include -1 placeholders.
        # Treat only non-negative IDs as actual generated tokens.
        return [tok for tok in self.output_token_ids if tok >= 0]

    def valid_output_len(self) -> int:
        return len(self.valid_output_tokens())

    def debug_summary(self) -> dict[str, int | bool | list[int] | None]:
        raw = self.output_token_ids
        raw_len = len(raw)
        raw_tail = raw[-8:]
        num_negative = sum(1 for tok in raw if tok < 0)
        first_nonneg = next((idx for idx, tok in enumerate(raw) if tok >= 0), None)
        first_negative = next((idx for idx, tok in enumerate(raw) if tok < 0), None)
        valid_tokens = self.valid_output_tokens()
        return {
            "raw_len": raw_len,
            "num_negative": num_negative,
            "first_nonneg": first_nonneg,
            "first_negative": first_negative,
            "raw_tail": raw_tail,
            "valid_len": len(valid_tokens),
            "valid_tail": valid_tokens[-8:],
            "end_in_raw": self.end_token_id in raw,
            "end_in_valid": self.end_token_id in valid_tokens,
            "reason_count": self.reasoning_token_count,
            "processed_len": self.processed_len,
            "is_reasoning": self.is_reasoning,
        }

    def update_from_output_tokens(self) -> None:
        valid_tokens = self.valid_output_tokens()
        self.processed_len = len(valid_tokens)
        if not valid_tokens:
            return
        if self.end_token_id in valid_tokens:
            # Only tokens before first end token are considered reasoning.
            first_end = valid_tokens.index(self.end_token_id)
            self.reasoning_token_count = max(self.reasoning_token_count, first_end)
            self.is_reasoning = False
            return
        self.reasoning_token_count = max(self.reasoning_token_count, len(valid_tokens))
        self.is_reasoning = True

    def current_penalty(self) -> float:
        if not self.is_reasoning:
            return 0.0

        overflow = self.reasoning_token_count - self.start_threshold
        if overflow < 0:
            return 0.0
        if self.curve == "linear":
            return self.coefficient * (overflow + 1)
        if self.curve == "quadratic":
            growth = overflow + 1
            return self.coefficient * (growth**2)
        # "exp"
        return self.coefficient * (2**overflow)


class ReasoningBudgetLogitsProcessor(LogitsProcessor):
    """Soft-penalty logits processor for reasoning budget control."""

    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.pin_memory = is_pin_memory
        self.req_info: dict[int, _ReasoningBudgetReqInfo] = {}
        self.debug_enabled = os.getenv("VLLM_DEBUG_REASONING_BUDGET", "") == "1"

    def is_argmax_invariant(self) -> bool:
        return False

    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int] | None, output_tok_ids: list[int]
    ) -> _ReasoningBudgetReqInfo | None:
        threshold = params.reasoning_soft_penalty_start_threshold
        coefficient = params.reasoning_soft_penalty_coefficient
        curve = params.reasoning_soft_penalty_curve
        end_token_id = params.reasoning_soft_penalty_end_token_id
        if (
            threshold is None
            or coefficient is None
            or curve is None
            or end_token_id is None
            or coefficient == 0
        ):
            if logger.isEnabledFor(10):
                logger.debug(
                    "ReasoningBudget disabled for request: "
                    "threshold=%s coefficient=%s curve=%s end_token_id=%s",
                    threshold,
                    coefficient,
                    curve,
                    end_token_id,
                )
            return None
        req_info = _ReasoningBudgetReqInfo(
            output_token_ids=output_tok_ids,
            end_token_id=end_token_id,
            start_threshold=threshold,
            coefficient=coefficient,
            curve=curve,
        )
        req_info.update_from_output_tokens()
        return req_info

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(self.req_info, batch_update, self.add_request)
        if self.debug_enabled and batch_update is not None:
            logger.info(
                "ReasoningBudget update_state: batch_size=%s active_reqs=%s "
                "added=%s removed=%s moved=%s",
                batch_update.batch_size,
                sorted(self.req_info.keys()),
                len(batch_update.added),
                len(batch_update.removed),
                len(batch_update.moved),
            )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info:
            return logits

        for req_idx, req_info in self.req_info.items():
            pre_update = req_info.debug_summary() if self.debug_enabled else None
            req_info.update_from_output_tokens()
            req_logits = logits[req_idx]
            if self.debug_enabled:
                logger.info(
                    "ReasoningBudget req=%s state: pre=%s post=%s",
                    req_idx,
                    pre_update,
                    req_info.debug_summary(),
                )
            if not req_info.is_reasoning:
                # Once reasoning has ended, prevent repeated emission of
                # the reasoning-end token.
                req_logits[req_info.end_token_id] = -float("inf")
                if self.debug_enabled:
                    logger.info(
                        "ReasoningBudget req=%s reasoning ended: "
                        "mask end_token_id=%s out_len=%s reason_count=%s",
                        req_idx,
                        req_info.end_token_id,
                        req_info.valid_output_len(),
                        req_info.reasoning_token_count,
                    )
                continue
            penalty = req_info.current_penalty()
            if penalty <= 0:
                used_fallback = False
                if req_info.is_reasoning and req_info.valid_output_len() == 0:
                    # Fallback for async/spec paths where output IDs may remain
                    # placeholders for several steps.
                    req_info.reasoning_token_count += 1
                    used_fallback = True
                if self.debug_enabled:
                    logger.info(
                        "ReasoningBudget req=%s no penalty: "
                        "reason_count=%s threshold=%s out_len=%s fallback=%s",
                        req_idx,
                        req_info.reasoning_token_count,
                        req_info.start_threshold,
                        req_info.valid_output_len(),
                        used_fallback,
                    )
                continue
            # Soft penalty:
            # - Before threshold: unchanged.
            # - After threshold: negative bias for continuation side tokens.
            # - Positive bias for end token to encourage natural completion.
            req_logits.sub_(penalty)
            req_logits[req_info.end_token_id] += 2 * penalty
            if self.debug_enabled:
                logger.info(
                    "ReasoningBudget req=%s apply penalty=%s curve=%s "
                    "reason_count=%s threshold=%s end_token_id=%s out_tail=%s",
                    req_idx,
                    penalty,
                    req_info.curve,
                    req_info.reasoning_token_count,
                    req_info.start_threshold,
                    req_info.end_token_id,
                    req_info.valid_output_tokens()[-8:],
                )
        return logits


def process_dict_updates(
    req_entries: dict[int, T],
    batch_update: BatchUpdate | None,
    new_state: Callable[[SamplingParams, list[int] | None, list[int]], T | None],
) -> bool:
    """Utility function to update dict state for sparse LogitsProcessors."""

    if not batch_update:
        # Nothing to do.
        return False

    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        if (state := new_state(params, prompt_tok_ids, output_tok_ids)) is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    if req_entries:
        # Process removed requests.
        for index in batch_update.removed:
            if req_entries.pop(index, None):
                updated = True

        # Process moved requests, unidirectional (a->b) and
        # swapped (a<->b)
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry

    return updated
