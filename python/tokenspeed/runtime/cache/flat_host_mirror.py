# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Byte-blind pinned-CPU mirror of a device KV pool for the flat L2 host
tier (M15 Phase D). Transport mechanism only; scheduler/engine wiring is D2.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


def _identity_dedup(
    tensors: Sequence[torch.Tensor | None],
) -> list[torch.Tensor]:
    """Distinct tensors in first-appearance order; None slots (flat GDN
    state layers carry no KV) are skipped."""
    seen: dict[int, torch.Tensor] = {}
    for t in tensors:
        if t is None:
            continue
        seen.setdefault(id(t), t)
    return list(seen.values())


def flat_bytes_per_host_page(device_kv_pool) -> int:
    """Bytes one host page occupies across all mirrors, computed from the
    device pool alone (no mirror allocation) -- the sizing side of
    ``FlatHostMirror.bytes_per_host_page`` for host-budget arithmetic.

    GDN state needs no extra term: its bytes alias segments inside the full
    layers' K/V slabs (M18c state binning), so the byte-blind slab copies
    below carry them for free.
    """
    tensors = _identity_dedup(device_kv_pool.k_buffer) + _identity_dedup(
        device_kv_pool.v_buffer
    )
    page_size = int(device_kv_pool.page_size)
    return sum(t.element_size() * t[0].numel() * page_size for t in tensors)


class FlatHostMirror:
    """One pinned CPU mirror per DISTINCT device KV tensor; a
    (device_page, host_page) pair copies that page's row range on every
    mirror pair.

    Slab tensors are enumerated once each -- a page's rows are exactly its
    owner group's layers, so byte copies are group-safe by id-exclusivity.
    GDN state bytes alias segments inside the full layers' K/V slabs
    (M18c state binning), so the blind slab copies cover them with no
    state-specific mirrors.

    ``tensor_pairs`` order (PINNED, D2 fencing indexes into it): K*, V*.
    Every mirror spans ``page_size`` token rows per page --
    ``row_spans[i]`` carries each pair's span.
    """

    def __init__(self, device_kv_pool, num_host_pages: int):
        self.page_size = int(device_kv_pool.page_size)
        self.num_host_pages = int(num_host_pages)

        # Slab layout dedups the per-layer entries to one K + one V slab per
        # paired layer set (layers-per-group slabs); legacy layout keeps all
        # per-layer buffers (dead-row copies are harmless).
        k_tensors = _identity_dedup(device_kv_pool.k_buffer)
        v_tensors = _identity_dedup(device_kv_pool.v_buffer)
        self.num_k_tensors = len(k_tensors)

        k_index = {id(t): i for i, t in enumerate(k_tensors)}
        v_index = {id(t): i for i, t in enumerate(v_tensors)}
        # None entries (flat GDN state layers, no KV) map to None: those
        # layers' bytes ride the full layers' slab copies (M18c aliasing)
        # and fence on the op's last per-tensor event.
        self._layer_to_k_index = [
            None if t is None else k_index[id(t)] for t in device_kv_pool.k_buffer
        ]
        # Invariant D2 relies on: a layer's V tensor sits at
        # tensor_index_of_layer(layer) + num_k_tensors.
        assert self._layer_to_k_index == [
            None if t is None else v_index[id(t)] for t in device_kv_pool.v_buffer
        ], "flat host mirror: K/V dedup orders diverge"

        pin = torch.cuda.is_available()
        self.tensor_pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...] = tuple(
            (
                dev,
                torch.zeros(
                    (self.num_host_pages * self.page_size, *dev.shape[1:]),
                    dtype=dev.dtype,
                    pin_memory=pin,
                ),
            )
            for dev in k_tensors + v_tensors
        )
        # Rows one page spans on each pair (uniform now that state carries
        # no page-indexed snapshot mirrors).
        self.row_spans: tuple[int, ...] = (self.page_size,) * len(self.tensor_pairs)

    def tensor_index_of_layer(self, layer_id: int) -> int:
        """Index of layer_id's K tensor in tensor_pairs (paired slab layers
        share the index); its V tensor is at index + num_k_tensors.
        Raises ValueError for flat GDN state layers (no KV tensor); their
        aliased state bytes ride the full layers' slab copies, so fence
        them on the op's last per-tensor event."""
        index = self._layer_to_k_index[layer_id]
        if index is None:
            raise ValueError(f"layer {layer_id} is a state layer; it has no KV mirror")
        return index

    def bytes_per_host_page(self) -> int:
        return sum(
            dev.element_size() * dev[0].numel() * span
            for (dev, _), span in zip(self.tensor_pairs, self.row_spans)
        )

    def _copy_pages(
        self,
        pairs: Iterable[tuple[int, int]],
        stream,
        to_host: bool,
        record_events: bool,
    ) -> list[torch.cuda.Event]:
        pairs = list(pairs)
        events: list[torch.cuda.Event] = []
        with torch.cuda.stream(stream):
            for (dev, mirror), p in zip(self.tensor_pairs, self.row_spans):
                for device_page, host_page in pairs:
                    dev_rows = dev[device_page * p : (device_page + 1) * p]
                    host_rows = mirror[host_page * p : (host_page + 1) * p]
                    if to_host:
                        host_rows.copy_(dev_rows, non_blocking=True)
                    else:
                        dev_rows.copy_(host_rows, non_blocking=True)
                if record_events:
                    event = torch.cuda.Event()
                    event.record()
                    events.append(event)
        return events

    def store_pages(self, pairs: Iterable[tuple[int, int]], stream) -> None:
        """Copy each (device_page, host_page) pair device -> host on stream."""
        self._copy_pages(pairs, stream, to_host=True, record_events=False)

    def load_pages(self, pairs: Iterable[tuple[int, int]], stream) -> None:
        """Copy each (device_page, host_page) pair host -> device on stream."""
        self._copy_pages(pairs, stream, to_host=False, record_events=False)

    def load_pages_with_events(
        self, pairs: Iterable[tuple[int, int]], stream
    ) -> list[torch.cuda.Event]:
        """load_pages, recording one event per device tensor (tensor_pairs
        order) after that tensor's copies -- D2's per-slab fencing hook."""
        return self._copy_pages(pairs, stream, to_host=False, record_events=True)
