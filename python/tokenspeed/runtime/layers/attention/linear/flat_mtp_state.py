# SPDX-License-Identifier: MIT

"""Commit accepted Flat MTP recurrent checkpoints to canonical State pages."""

from __future__ import annotations

from collections.abc import Sequence

import torch

try:
    from tokenspeed_kernel.ops.kvcache import flat_mtp_state_commit
except (ImportError, ModuleNotFoundError):  # CPU-only unit-test environments.
    flat_mtp_state_commit = None


def selected_checkpoint_steps(
    out_pages: Sequence[int], accepted_length: int
) -> tuple[int, ...]:
    """Return checkpoints that are the last accepted write to each out page.

    This is the CPU reference for the device-side commit predicate.  The
    production path does not use it to compress device metadata.
    """
    width = len(out_pages)
    if accepted_length < 1 or accepted_length > width:
        raise ValueError(
            f"accepted_length must be in [1, {width}], got {accepted_length}"
        )
    return tuple(
        step
        for step in range(accepted_length)
        if step == accepted_length - 1 or out_pages[step] != out_pages[step + 1]
    )


def commit_flat_mtp_state_rows(
    state: torch.Tensor,
    spec_pages: torch.Tensor,
    out_pages: torch.Tensor,
    accepted_lengths: torch.Tensor,
    *,
    validate_accepted_lengths: bool = True,
) -> None:
    """Commit selected checkpoints through the tokenspeed-kernel boundary.

    Args:
        state: State page rows shaped ``[num_pages, ...]``.
        spec_pages: Speculative source page ids shaped ``[batch, N]``.
        out_pages: Canonical destination page ids shaped ``[batch, N]``.
        accepted_lengths: Accepted verify width shaped ``[batch]``.
        validate_accepted_lengths: Whether the kernel boundary should schedule
            the device-side accepted-range assertion.

    Returns:
        None. The selected canonical rows are updated in place.
    """
    if flat_mtp_state_commit is None:
        raise RuntimeError("Flat MTP State commit requires tokenspeed-kernel")
    flat_mtp_state_commit(
        state,
        spec_pages,
        out_pages,
        accepted_lengths,
        validate_accepted_lengths=validate_accepted_lengths,
    )


def commit_flat_mtp_state_pages(
    kv_pool,
    spec_pages: torch.Tensor,
    out_pages: torch.Tensor,
    accepted_lengths: torch.Tensor,
) -> None:
    """Commit conv and SSM checkpoints for every Flat recurrent-state layer.

    Args:
        kv_pool: Flat KV pool exposing ``state_shard_view.state_layer_ids`` and
            ``get_state_buffers(layer_id)``.
        spec_pages: Distinct speculative page ids, shaped ``[shard, batch, N]``.
        out_pages: Canonical destination ids, shaped ``[shard, batch, N]``.
        accepted_lengths: Accepted verify width for each real request,
            shaped ``[batch]``.
    """
    request_count = accepted_lengths.shape[0]
    if spec_pages.ndim != 3 or out_pages.ndim != 3:
        raise ValueError("Flat MTP State pages must have shape [shard, batch, N]")
    if spec_pages.shape != out_pages.shape:
        raise ValueError("Flat MTP speculative and canonical page shapes differ")
    if accepted_lengths.ndim != 1 or request_count > spec_pages.shape[1]:
        raise ValueError("accepted_lengths exceed the available request rows")
    verify_width = spec_pages.shape[2]
    if request_count:
        torch._assert_async(
            ((accepted_lengths >= 1) & (accepted_lengths <= verify_width)).all(),
            f"accepted_lengths must be in [1, {verify_width}]",
        )
    for layer_id in sorted(kv_pool.state_shard_view.state_layer_ids):
        groups = kv_pool.get_state_buffers(layer_id)
        if not groups:
            raise ValueError(f"state layer {layer_id} has no state head groups")

        conv_group = groups[0]
        commit_flat_mtp_state_rows(
            conv_group.conv,
            spec_pages[conv_group.conv_shard, :request_count],
            out_pages[conv_group.conv_shard, :request_count],
            accepted_lengths,
            validate_accepted_lengths=False,
        )
        for group in groups:
            commit_flat_mtp_state_rows(
                group.ssm,
                spec_pages[group.shard, :request_count],
                out_pages[group.shard, :request_count],
                accepted_lengths,
                validate_accepted_lengths=False,
            )
