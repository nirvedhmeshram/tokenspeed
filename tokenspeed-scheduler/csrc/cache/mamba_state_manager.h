// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "cache/swa_manager.h"

namespace tokenspeed {

// GDN/mamba semantics: matching and retention keep the last reachable aligned snapshot
// (= SwaManager(P, 2)), while registration can publish every complete page whose snapshot
// was emitted by the backend. Non-final prefill chunks stay on the shared checkpoint grid.
class MambaStateManager : public SwaManager {
public:
    explicit MambaStateManager(std::int32_t block_size) : SwaManager(block_size, /*sliding_window=*/2) {}

    bool CachesAlignedStateSnapshots() const override { return true; }
    bool RequiresAlignedPrefillChunks() const override { return true; }
    std::int32_t SafeReclaimBoundary(std::int32_t computed_end_tokens,
                                     std::int32_t protected_input_tokens) const override {
        return std::max(computed_end_tokens - protected_input_tokens, 0);
    }

    // Additional request-owned State pages required to reach target_num_blocks.
    // Existing pages remain stable for the request lifetime; a lower target does not shrink them.
    std::int32_t SpeculativeBlocksNeededFor(const BlockTable& table, std::int32_t target_num_blocks) const override {
        _assert(target_num_blocks >= 0, "target_num_blocks must be >= 0");
        const auto have = static_cast<std::int32_t>(table.speculative_blocks_.size());
        return std::max<std::int32_t>(target_num_blocks - have, 0);
    }

    // All-or-nothing growth of the request-owned speculative State segment. The base defaults
    // (0 needed / true) make this a no-op for full/SWA groups, so only State groups ever grow one.
    bool AcquireSpeculativeBlocks(BlockPool& pool, BlockTable& table, std::int32_t target_num_blocks) override {
        const std::int32_t needed = SpeculativeBlocksNeededFor(table, target_num_blocks);
        if (needed == 0) {
            return true;
        }
        std::vector<CacheBlock*> new_blocks = pool.AllocateBlocks(needed);
        if (static_cast<std::int32_t>(new_blocks.size()) < needed) {
            return false;
        }
        table.speculative_blocks_.reserve(table.speculative_blocks_.size() + new_blocks.size());
        for (CacheBlock* block : new_blocks) {
            table.speculative_blocks_.push_back(BlockRef::Adopt(pool, block));
        }
        return true;
    }
};

}  // namespace tokenspeed
